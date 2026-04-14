from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf


def main() -> None:
    env = ImitationFactory.make(
        "SkeletonMuscle",
        default_dataset_conf=DefaultDatasetConf(task="walk", dataset_type="mocap"),
        headless=True,
    )
    try:
        obs = env.reset()
        print(f"Environment: {type(env).__name__}")
        print(f"Observation shape: {obs.shape}")
        print(f"Action dimension: {env.action_dim}")
        print(f"Trajectory loaded: {env.th is not None}")
        print(f"Current sim time: {env._data.time:.3f}")

        env.play_trajectory(
            n_episodes=1,
            n_steps_per_episode=100,
            render=False,
            quiet=True,
        )

        print("Walking trajectory replay: ok")
        print(f"Replay sim time: {env._data.time:.3f}")
    finally:
        env.stop()


if __name__ == "__main__":
    main()
