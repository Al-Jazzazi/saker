import argparse
import csv
from pathlib import Path

import mujoco

from loco_mujoco.task_factories import DefaultDatasetConf, ImitationFactory

from fsr_sensors import bind_sensors, extract_virtual_fsr_sample


def replay_callback(records, bound_sensors):
    step_idx = 0

    def callback(env, model, data, traj_data_sample, carry):
        nonlocal step_idx

        data = env.set_sim_state_from_traj_data(data, traj_data_sample, carry)
        model, data, carry = env._simulation_pre_step(model, data, carry)
        mujoco.mj_forward(model, data)
        data, carry = env._simulation_post_step(model, data, carry)

        row = {
            "step": step_idx,
            "time_s": step_idx * env.dt,
            "ncon": int(data.ncon),
        }
        row.update(extract_virtual_fsr_sample(model, data, bound_sensors))

        records.append(row)
        step_idx += 1
        return model, data, carry

    return callback


def write_csv(output_path: Path, records) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "time_s",
        "ncon",
        "A_heel_n",
        "A_mid_n",
        "A_toe_n",
        "U_heel_n",
        "U_mid_n",
        "U_toe_n",
        "A_total_n",
        "U_total_n",
        "A_heel",
        "A_mid",
        "A_toe",
        "U_heel",
        "U_mid",
        "U_toe",
        "A_contact",
        "U_contact",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mujoco_sim/output/skeleton_walk_fsr.csv"),
    )
    args = parser.parse_args()

    records = []
    env = ImitationFactory.make(
        "SkeletonMuscle",
        default_dataset_conf=DefaultDatasetConf(task="walk", dataset_type="mocap"),
        headless=True,
        use_box_feet=False,
    )
    try:
        bound_sensors = bind_sensors(env._model)
        env.play_trajectory(
            n_episodes=1,
            n_steps_per_episode=args.steps,
            render=False,
            quiet=True,
            callback_class=replay_callback(records, bound_sensors),
        )
    finally:
        env.stop()

    write_csv(args.output, records)

    print(f"Saved {len(records)} samples to {args.output}")
    print(f"dt: {env.dt}")
    print(
        "Max raw forces [N]:",
        {
            key: round(max(row[key] for row in records), 3)
            for key in ("A_heel_n", "A_mid_n", "A_toe_n", "U_heel_n", "U_mid_n", "U_toe_n")
        },
    )
    print(
        "Max FSR values [0-1024]:",
        {
            key: max(row[key] for row in records)
            for key in ("A_heel", "A_mid", "A_toe", "U_heel", "U_mid", "U_toe")
        },
    )


if __name__ == "__main__":
    main()
