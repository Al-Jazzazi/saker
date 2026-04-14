import argparse
import time
import threading

import numpy as np

from loco_mujoco.environments.humanoids import SkeletonMuscle


def main() -> None:
    if threading.current_thread() is not threading.main_thread():
        raise SystemExit(
            "This viewer must be launched with `.venv/bin/python`, not `mjpython`. "
            "LocoMuJoCo creates its own GLFW window and macOS requires that window on the main thread."
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    env = SkeletonMuscle()
    action = np.zeros(env.action_dim, dtype=np.float32)

    try:
        env.reset()
        for _ in range(args.steps):
            loop_start = time.time()
            env.render()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()

            remaining = env.dt - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.stop()


if __name__ == "__main__":
    main()
