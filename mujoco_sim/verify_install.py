from pathlib import Path

import mujoco


MODEL_PATH = Path(__file__).with_name("hello_world.xml")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    for _ in range(500):
        mujoco.mj_step(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe_body")
    xpos = data.xpos[body_id]

    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Simulated time: {data.time:.3f}s")
    print(f"Probe position: ({xpos[0]:.3f}, {xpos[1]:.3f}, {xpos[2]:.3f})")


if __name__ == "__main__":
    main()
