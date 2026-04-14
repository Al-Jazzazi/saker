from pathlib import Path
import time

import mujoco
import mujoco.viewer


MODEL_PATH = Path(__file__).with_name("hello_world.xml")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_started = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            remaining = model.opt.timestep - (time.time() - step_started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
