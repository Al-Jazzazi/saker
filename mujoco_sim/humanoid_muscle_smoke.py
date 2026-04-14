import numpy as np

from loco_mujoco.environments.humanoids import SkeletonMuscle


def main() -> None:
    env = SkeletonMuscle()
    try:
        obs = env.reset()
        print(f"Environment: {type(env).__name__}")
        print(f"Observation shape: {obs.shape}")
        print(f"Action dimension: {env.action_dim}")
        print(f"Simulation dt: {env.dt}")

        action = np.zeros(env.action_dim, dtype=np.float32)
        for _ in range(100):
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        print(f"Final observation shape: {obs.shape}")
        print(f"Final reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Sim time: {env._data.time:.3f}")
    finally:
        env.stop()


if __name__ == "__main__":
    main()
