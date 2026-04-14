import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from loco_mujoco.task_factories import DefaultDatasetConf, ImitationFactory


# These geoms are active when `use_box_feet=False`.
# `*_foot` is attached to the calcaneus/foot body and `*_bofoot` to the toes body,
# so these behave like hindfoot/forefoot channels during replay.
CHANNEL_GEOMS = {
    "r_hindfoot_n": "r_foot",
    "r_forefoot_n": "r_bofoot",
    "l_hindfoot_n": "l_foot",
    "l_forefoot_n": "l_bofoot",
}

CONTACT_THRESHOLD_N = 20.0


def pair_normal_force(model, data, geom_a: str, geom_b: str) -> float:
    geom_a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a)
    geom_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b)

    force = np.zeros(6, dtype=float)
    total_normal_force = 0.0

    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        if {contact.geom1, contact.geom2} == {geom_a_id, geom_b_id}:
            mujoco.mj_contactForce(model, data, contact_idx, force)
            total_normal_force += max(0.0, float(force[0]))

    return total_normal_force


def replay_callback(records):
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
        for channel, geom_name in CHANNEL_GEOMS.items():
            row[channel] = pair_normal_force(model, data, "floor", geom_name)

        row["r_total_n"] = row["r_hindfoot_n"] + row["r_forefoot_n"]
        row["l_total_n"] = row["l_hindfoot_n"] + row["l_forefoot_n"]
        row["r_contact"] = int(row["r_total_n"] >= CONTACT_THRESHOLD_N)
        row["l_contact"] = int(row["l_total_n"] >= CONTACT_THRESHOLD_N)

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
        "r_hindfoot_n",
        "r_forefoot_n",
        "l_hindfoot_n",
        "l_forefoot_n",
        "r_total_n",
        "l_total_n",
        "r_contact",
        "l_contact",
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
        default=Path("mujoco_sim/output/skeleton_walk_contacts.csv"),
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
        env.play_trajectory(
            n_episodes=1,
            n_steps_per_episode=args.steps,
            render=False,
            quiet=True,
            callback_class=replay_callback(records),
        )
    finally:
        env.stop()

    write_csv(args.output, records)

    print(f"Saved {len(records)} samples to {args.output}")
    print(f"dt: {env.dt}")
    print(
        "Max forces [N]:",
        {
            key: round(max(row[key] for row in records), 3)
            for key in ("r_hindfoot_n", "r_forefoot_n", "l_hindfoot_n", "l_forefoot_n")
        },
    )
    print(
        "Contact samples:",
        {
            key: sum(row[key] for row in records)
            for key in ("r_contact", "l_contact")
        },
    )


if __name__ == "__main__":
    main()
