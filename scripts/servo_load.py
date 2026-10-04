#!/usr/bin/env python3
"""Measure servo load while a policy walks XgoDuck in sim.

Runs N parallel envs per fixed-command scenario (stand, walk, fast walk, turn,
walk+turn) and reports, per joint, the output torque the BAM actuator model
applied: RMS, p95, peak, the share of time above 30% / 50% of stall, and the
copper loss I^2*R it implies. The HLS1910's continuous rating is unpublished,
so loads are compared against stall torque (vin * kt / R).

It also reports how long each joint rests within 0.02 rad of a hard limit and
the torque it pushes there. Bracing against a stop is free stiffness in sim but
a stalled servo pressing on a printed stop on the robot.

  # a training checkpoint
  uv run python scripts/servo_load.py Mjlab-Velocity-Flat-XgoDuck \
      --checkpoint logs/rsl_rl/xgoduck_velocity/<run>/model_3000.pt
  # an exported runtime policy (batch-1 ONNX, e.g. xgoduck_runtime_arduino/python/xgoduck_walk.onnx)
  uv run python scripts/servo_load.py Mjlab-Velocity-Flat-XgoDuck --onnx xgoduck_walk.onnx
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import mjlab_microduck
import mjlab_microduck.tasks  # registers the tasks

DEFAULT_PARAMS = Path(mjlab_microduck.__file__).parent / "robot/xgoduck/params/1910_m6.json"
VIN_NOMINAL = 7.7  # middle of the actuator cfg's vin_range (7.4, 8.0)
STOP_MARGIN = 0.02  # rad from a joint limit that counts as resting on the stop

# name: (vx [m/s], vy [m/s], wz [rad/s])
SCENARIOS = {
    "stand": (0.0, 0.0, 0.0),
    "walk": (0.15, 0.0, 0.0),
    "walk_fast": (0.30, 0.0, 0.0),
    "turn": (0.0, 0.0, 0.8),
    "walk_turn": (0.15, 0.0, 0.5),
}


def fix_twist_command(env_cfg, vx: float, vy: float, wz: float) -> None:
    cmd = env_cfg.commands["twist"]
    cmd.ranges.lin_vel_x = (vx, vx)
    cmd.ranges.lin_vel_y = (vy, vy)
    cmd.ranges.ang_vel_z = (wz, wz)
    cmd.heading_command = False
    cmd.ranges.heading = None
    cmd.rel_heading_envs = 0.0
    cmd.rel_world_envs = 0.0
    cmd.rel_forward_envs = 0.0  # forward envs clamp vx up to >= 0.3
    cmd.rel_standing_envs = 1.0 if (vx, vy, wz) == (0.0, 0.0, 0.0) else 0.0
    if hasattr(cmd, "rel_turn_in_place_envs"):
        cmd.rel_turn_in_place_envs = 0.0
    # Loading a checkpoint restores the training step counter, so on the first
    # reset this curriculum jumps to its last stage and zeroes 25% of commands.
    if env_cfg.curriculum and "standing_envs" in env_cfg.curriculum:
        env_cfg.curriculum["standing_envs"] = None


class OnnxPolicy:
    """Batch-1 ONNX policy from the XGO runtime, run batched on CPU.

    The shipped policies carry their own default pose in metadata. When it
    differs from the env's home pose, joint_pos and last-action observations
    and the returned actions are shifted so targets land where the policy
    meant them.
    """

    JOINT_POS = slice(6, 20)
    LAST_ACTION = slice(34, 48)

    def __init__(self, path: str, robot):
        import onnx
        import onnxruntime as ort

        model = onnx.load(path)
        meta = {p.key: p.value for p in model.metadata_props}
        for t in list(model.graph.input) + list(model.graph.output):
            t.type.tensor_type.shape.dim[0].dim_param = "N"
        self.sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        names = meta["joint_names"].split(",")
        env_names = list(robot.actuator_names)
        assert names == env_names, f"joint order differs:\n{names}\n{env_names}"
        onnx_default = np.array([float(v) for v in meta["default_joint_pos"].split(",")])
        joint_ids = [robot.joint_names.index(n) for n in names]
        env_default = robot.data.default_joint_pos[0, joint_ids].cpu().numpy()
        self.delta = (env_default - onnx_default).astype(np.float32)
        print(f"[onnx] default-pose delta (deg): {np.round(np.degrees(self.delta), 2).tolist()}")

    def __call__(self, obs) -> torch.Tensor:
        actor = obs["actor"]
        x = actor.detach().cpu().numpy().astype(np.float32)
        assert x.shape[1] == 61, f"expected 61-d actor obs, got {x.shape[1]}"
        x[:, self.JOINT_POS] += self.delta
        x[:, self.LAST_ACTION] += self.delta
        (a,) = self.sess.run(None, {self.sess.get_inputs()[0].name: x})
        return torch.as_tensor(a - self.delta, device=actor.device)


def run_scenario(task, scenario, args, kt, R):
    vx, vy, wz = SCENARIOS[scenario]
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    fix_twist_command(env_cfg, vx, vy, wz)
    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    robot = env.scene["robot"]

    if args.onnx:
        policy = OnnxPolicy(args.onnx, robot)
    else:
        runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
        runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)

    steps = int(args.seconds / env.step_dt)
    warmup = int(args.warmup / env.step_dt)
    taus, poss, vels, speeds, yaw_rates, cmds = [], [], [], [], [], []
    falls = 0
    obs = wrapped.get_observations()
    with torch.inference_mode():
        for i in range(steps):
            obs, _, dones, extras = wrapped.step(policy(obs))
            time_outs = extras.get("time_outs", torch.zeros_like(dones))
            falls_now = dones.bool() & ~time_outs.bool()
            if i < warmup:
                continue
            falls += int(falls_now.sum())
            taus.append(robot.data.actuator_force.float().cpu())
            poss.append(robot.data.joint_pos.float().cpu())
            vels.append(robot.data.joint_vel.float().cpu())
            speeds.append(robot.data.root_link_lin_vel_b[:, 0].float().cpu())
            yaw_rates.append(robot.data.root_link_ang_vel_b[:, 2].float().cpu())
            cmds.append(env.command_manager.get_command("twist")[:, :3].float().cpu())

    tau = torch.stack(taus).numpy()  # (T, N, J)
    names = list(robot.actuator_names)
    joint_ids = [robot.joint_names.index(n) for n in names]
    omega = torch.stack(vels).numpy()[:, :, joint_ids]
    q = torch.stack(poss).numpy()[:, :, joint_ids]
    limits = robot.data.joint_pos_limits[0, joint_ids].cpu().numpy()  # (J, 2)
    at_stop = (q < limits[:, 0] + STOP_MARGIN) | (q > limits[:, 1] - STOP_MARGIN)
    env_seconds = tau.shape[0] * tau.shape[1] * env.step_dt
    env.close()

    stall = VIN_NOMINAL * kt / R
    abs_tau = np.abs(tau)
    current = abs_tau / kt
    joints = {}
    for j, name in enumerate(names):
        t = abs_tau[:, :, j]
        joints[name] = {
            "rms_nm": float(np.sqrt(np.mean(t**2))),
            "p95_nm": float(np.percentile(t, 95)),
            "peak_nm": float(t.max()),
            "pct_over_30_stall": float(100 * np.mean(t > 0.3 * stall)),
            "pct_over_50_stall": float(100 * np.mean(t > 0.5 * stall)),
            "copper_w": float(np.mean(current[:, :, j] ** 2) * R),
            "mech_w": float(np.mean(np.clip(tau[:, :, j] * omega[:, :, j], 0, None))),
            # Bracing: time spent within STOP_MARGIN of a hard stop, and the torque
            # pushed while there. Free stiffness in sim, a stalled servo on hardware.
            "pct_at_stop": float(100 * np.mean(at_stop[:, :, j])),
            "tau_at_stop_nm": float(t[at_stop[:, :, j]].mean()) if at_stop[:, :, j].any() else 0.0,
        }
    cmd = torch.stack(cmds).numpy().reshape(-1, 3).mean(axis=0)
    return {
        "scenario": scenario,
        "command_mean": cmd.tolist(),
        "forward_speed_mean": float(torch.stack(speeds).mean()),
        "yaw_rate_mean": float(torch.stack(yaw_rates).mean()),
        "falls_per_env_minute": falls / (env_seconds / 60.0),
        "stall_nm": stall,
        "total_copper_w": float(sum(v["copper_w"] for v in joints.values())),
        "joints": joints,
    }


def print_report(res):
    print(
        f"\n== {res['scenario']}: cmd={np.round(res['command_mean'], 3).tolist()} "
        f"vx={res['forward_speed_mean']:.3f} m/s wz={res['yaw_rate_mean']:.3f} rad/s "
        f"falls/env-min={res['falls_per_env_minute']:.3f} "
        f"copper={res['total_copper_w']:.2f} W (stall {res['stall_nm']:.3f} Nm)"
    )
    print(f"{'joint':16} {'rms':>6} {'p95':>6} {'peak':>6} {'>30%':>6} {'>50%':>6} {'Cu W':>6} "
          f"{'stop%':>6} {'stopNm':>6}")
    for name, v in res["joints"].items():
        print(
            f"{name:16} {v['rms_nm']:6.3f} {v['p95_nm']:6.3f} {v['peak_nm']:6.3f} "
            f"{v['pct_over_30_stall']:6.1f} {v['pct_over_50_stall']:6.1f} {v['copper_w']:6.2f} "
            f"{v['pct_at_stop']:6.1f} {v['tau_at_stop_nm']:6.3f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="rsl_rl model_*.pt")
    src.add_argument("--onnx", help="batch-1 ONNX policy from the XGO runtime")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--params", default=str(DEFAULT_PARAMS), help="BAM actuator JSON (kt, R) used for stall and copper loss")
    ap.add_argument("--out", help="JSON path (default logs/servo_load/<task>_<time>.json)")
    args = ap.parse_args()

    params = json.loads(Path(args.params).read_text())
    kt, R = params["kt"], params["R"]
    results = []
    for scenario in args.scenarios.split(","):
        res = run_scenario(args.task, scenario, args, kt, R)
        print_report(res)
        results.append(res)

    out = Path(args.out) if args.out else Path("logs/servo_load") / (
        f"{args.task}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "task": args.task,
        "policy": args.onnx or args.checkpoint,
        "num_envs": args.num_envs,
        "seconds": args.seconds,
        "kt": kt,
        "R": R,
        "vin_nominal": VIN_NOMINAL,
        "results": results,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
