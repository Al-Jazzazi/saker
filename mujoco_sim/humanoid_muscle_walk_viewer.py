import argparse
import os
import sys
import threading
from collections import Counter

import mujoco
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf

from fsr_sensors import bind_sensors, extract_virtual_fsr_sample


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "gait_detection"))

from algorithms import AffectedKneeFSMController


LEFT_KNEE_JOINT = "knee_angle_l"
RIGHT_KNEE_JOINT = "knee_angle_r"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def knee_target_to_joint_qpos(target_deg: float, joint_range: tuple[float, float]) -> float:
    # In this model, knee flexion is represented by negative hinge values.
    target_rad = -target_deg * 3.141592653589793 / 180.0
    return clamp(target_rad, joint_range[0], joint_range[1])


def replay_callback(args, controller, bound_sensors, diagnostics, joint_override):
    step_idx = 0

    def callback(env, model, data, traj_data_sample, carry):
        nonlocal step_idx

        data = env.set_sim_state_from_traj_data(data, traj_data_sample, carry)
        model, data, carry = env._simulation_pre_step(model, data, carry)

        if args.controller == "fsm":
            mujoco.mj_forward(model, data)
            sample = extract_virtual_fsr_sample(
                model,
                data,
                bound_sensors,
                affected_side=args.affected_side,
                unaffected_side=("right" if args.affected_side == "left" else "left"),
            )
            sample["t"] = step_idx * env.dt
            controller_output = controller.update(sample)

            data.qpos[joint_override["qpos_adr"]] = knee_target_to_joint_qpos(
                controller_output["A_knee_target_deg"],
                joint_override["joint_range"],
            )
            data.qvel[joint_override["qvel_adr"]] = 0.0
            diagnostics["state_counts"][controller_output["state"]] += 1
            if controller_output["state_changed"]:
                diagnostics["transitions"].append(
                    (
                        controller_output["t"],
                        controller_output["state"],
                        controller_output["A_knee_target_deg"],
                        controller_output["transition_reason"],
                    )
                )

        mujoco.mj_forward(model, data)
        data, carry = env._simulation_post_step(model, data, carry)
        step_idx += 1
        return model, data, carry

    return callback


def main() -> None:
    if threading.current_thread() is not threading.main_thread():
        raise SystemExit(
            "This viewer must be launched with `.venv/bin/python`, not `mjpython`. "
            "LocoMuJoCo creates its own GLFW window and macOS requires that window on the main thread."
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--controller", choices=("original", "fsm"), default="original")
    parser.add_argument("--affected-side", choices=("left", "right"), default="left")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    print("Loading SkeletonMuscle walking task...")
    env = ImitationFactory.make(
        "SkeletonMuscle",
        default_dataset_conf=DefaultDatasetConf(task="walk", dataset_type="mocap"),
        headless=args.headless,
        use_box_feet=False,
    )
    completed = False
    try:
        print(f"Walking task loaded. Controller mode: {args.controller}.")
        if not args.headless:
            print("Opening viewer...")

        controller = AffectedKneeFSMController() if args.controller == "fsm" else None
        bound_sensors = bind_sensors(env._model) if args.controller == "fsm" else None
        joint_name = LEFT_KNEE_JOINT if args.affected_side == "left" else RIGHT_KNEE_JOINT
        joint_override = None
        if args.controller == "fsm":
            joint_id = env._model.joint(joint_name).id
            joint_override = {
                "name": joint_name,
                "qpos_adr": int(env._model.jnt_qposadr[joint_id]),
                "qvel_adr": int(env._model.jnt_dofadr[joint_id]),
                "joint_range": tuple(float(x) for x in env._model.jnt_range[joint_id]),
            }
        diagnostics = {
            "state_counts": Counter(),
            "transitions": [],
        }

        env.play_trajectory(
            n_episodes=args.episodes,
            n_steps_per_episode=args.steps,
            render=not args.headless,
            quiet=True,
            callback_class=(
                replay_callback(args, controller, bound_sensors, diagnostics, joint_override)
                if args.controller == "fsm"
                else None
            ),
        )
        completed = True
        print("Replay complete.")
        if args.controller == "fsm":
            print("FSM state counts:", dict(diagnostics["state_counts"]))
            print("First transitions:")
            for t, state, knee_deg, reason in diagnostics["transitions"][:10]:
                print(f"  t={t:.3f}s state={state} knee={knee_deg:.1f}deg reason={reason}")
    finally:
        if not completed:
            env.stop()


if __name__ == "__main__":
    main()
