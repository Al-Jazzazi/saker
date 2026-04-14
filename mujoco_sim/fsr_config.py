from dataclasses import dataclass


# Linear full-scale conversion from Newtons to integer FSR units.
# Hardware-faithful mode uses a 100 N full-scale sensor range, which means
# stance loads will often saturate near 1024.
FORCE_FULL_SCALE_N = 100.0
FSR_FULL_SCALE = 1024

# Side mapping for downstream gait code.
# Existing detectors use A_* and U_* channels, so we map a simulation side
# to those labels here instead of hardcoding it in the extractor.
AFFECTED_SIDE = "right"
UNAFFECTED_SIDE = "left"

# Binary contact threshold on total per-foot load.
CONTACT_THRESHOLD_N = 20.0

# Gaussian distance scale for distributing a single MuJoCo contact force
# across nearby virtual sensors on the same foot.
DISTRIBUTION_SIGMA_M = 0.05


@dataclass(frozen=True)
class VirtualSensorSpec:
    foot_side: str
    channel_name: str
    body_name: str
    anchor_geom_name: str
    offset_m: tuple[float, float, float]
    source_geoms: tuple[str, ...]


# Three virtual sensors per foot:
# - heel and mid use the calcaneus/foot geom
# - toe uses the toe geom
# - mid is allowed to receive force from both geoms so it can bridge stance
#   between hindfoot and forefoot contacts
SENSOR_LAYOUT = (
    VirtualSensorSpec(
        foot_side="right",
        channel_name="heel",
        body_name="calcn_r",
        anchor_geom_name="r_foot",
        offset_m=(-0.055, 0.0, 0.0),
        source_geoms=("r_foot",),
    ),
    VirtualSensorSpec(
        foot_side="right",
        channel_name="mid",
        body_name="calcn_r",
        anchor_geom_name="r_foot",
        offset_m=(0.010, 0.0, 0.0),
        source_geoms=("r_foot", "r_bofoot"),
    ),
    VirtualSensorSpec(
        foot_side="right",
        channel_name="toe",
        body_name="toes_r",
        anchor_geom_name="r_bofoot",
        offset_m=(0.035, 0.0, 0.0),
        source_geoms=("r_bofoot",),
    ),
    VirtualSensorSpec(
        foot_side="left",
        channel_name="heel",
        body_name="calcn_l",
        anchor_geom_name="l_foot",
        offset_m=(-0.055, 0.0, 0.0),
        source_geoms=("l_foot",),
    ),
    VirtualSensorSpec(
        foot_side="left",
        channel_name="mid",
        body_name="calcn_l",
        anchor_geom_name="l_foot",
        offset_m=(0.010, 0.0, 0.0),
        source_geoms=("l_foot", "l_bofoot"),
    ),
    VirtualSensorSpec(
        foot_side="left",
        channel_name="toe",
        body_name="toes_l",
        anchor_geom_name="l_bofoot",
        offset_m=(0.035, 0.0, 0.0),
        source_geoms=("l_bofoot",),
    ),
)
