# MuJoCo Local Setup

This folder contains the local MuJoCo setup and smoke tests for this repo.

## Prerequisite

Create the repo-local virtual environment at the project root if needed:

```bash
python3.12 -m venv .venv
```

Then use it:

```bash
source .venv/bin/activate
```

## Verify MuJoCo Works

Run the headless smoke test:

```bash
.venv/bin/python mujoco_sim/verify_install.py
```

It should print the MuJoCo version, simulated time, and final body position.

## HumanoidMuscle Setup

For the musculoskeletal humanoid, use the source checkout of LocoMuJoCo rather than the PyPI wheel:

```bash
.venv/bin/python -m pip install mujoco
git clone https://github.com/robfiras/loco-mujoco.git external/loco-mujoco
.venv/bin/python -m pip install -e external/loco-mujoco
```

The PyPI wheel was missing model assets needed by `HumanoidMuscle` / `SkeletonMuscle`, while the source install pulls in the official model package.

## Verify HumanoidMuscle Works

Run the headless humanoid smoke test:

```bash
.venv/bin/python mujoco_sim/humanoid_muscle_smoke.py
```

This should report a `SkeletonMuscle` environment with a `65`-dimensional observation and `106` actions.

To open the humanoid viewer:

```bash
.venv/bin/python mujoco_sim/humanoid_muscle_viewer.py --steps 2000
```

Do not use `mjpython` for this script. LocoMuJoCo creates its own GLFW window, and on macOS that must happen on the real main thread.

`HumanoidMuscle` is a deprecated alias upstream; use `SkeletonMuscle` in code.

## Load The Walking Task

To load the built-in walking imitation task for the musculoskeletal humanoid:

```bash
.venv/bin/python mujoco_sim/humanoid_muscle_walk_smoke.py
```

This downloads the default walking dataset on first run if it is not cached yet.

To replay the walking trajectory in the viewer:

```bash
.venv/bin/python mujoco_sim/humanoid_muscle_walk_viewer.py --steps 500
```

To compare the untouched replay against the affected-knee FSM override:

```bash
.venv/bin/python mujoco_sim/humanoid_muscle_walk_viewer.py --controller original --steps 500
.venv/bin/python mujoco_sim/humanoid_muscle_walk_viewer.py --controller fsm --affected-side left --steps 500
```

`--controller original` leaves the mocap knee motion untouched.
`--controller fsm` overrides the affected knee joint each frame using the standalone FSR-driven FSM controller in `gait_detection/algorithms/knee_fsm_controller.py`.

To extract FSR-like foot-contact channels from the walking replay:

```bash
.venv/bin/python mujoco_sim/extract_walk_fsr.py --steps 1000
```

This writes a CSV with six virtual FSR channels:

- `A_heel`, `A_mid`, `A_toe`
- `U_heel`, `U_mid`, `U_toe`

Each FSR channel is an integer in `0-1024`, linearly mapped from `0-100 N`.
The raw Newton values are also written alongside them in the CSV.

The configurable sensor layout and scaling live in:

- `mujoco_sim/fsr_config.py`
- `mujoco_sim/fsr_sensors.py`

## Launch the Viewer

On macOS, launch the viewer with `mjpython`:

```bash
.venv/bin/mjpython mujoco_sim/hello_viewer.py
```

This opens a simple MuJoCo scene and continuously steps physics.

## Files

- `requirements.txt`: pinned bare-MuJoCo dependency for the basic smoke test.
- `hello_world.xml`: minimal MuJoCo scene used for verification.
- `verify_install.py`: headless smoke test for the MuJoCo install.
- `hello_viewer.py`: interactive viewer entrypoint.
- `humanoid_muscle_smoke.py`: headless smoke test for the musculoskeletal humanoid.
- `humanoid_muscle_viewer.py`: viewer runner for the musculoskeletal humanoid.
- `humanoid_muscle_walk_smoke.py`: headless smoke test for the walking imitation task.
- `humanoid_muscle_walk_viewer.py`: replay viewer for the walking imitation task.
- `extract_walk_contacts.py`: CSV extractor for FSR-like foot-contact channels during walking replay.
- `extract_walk_fsr.py`: CSV extractor for six virtual FSR channels in `A_/U_` format.
- `fsr_config.py`: configurable force scale, side mapping, and sensor layout.
- `fsr_sensors.py`: virtual sensor projection and force-to-FSR conversion logic.
