from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable

import mujoco
import numpy as np

from fsr_config import (
    AFFECTED_SIDE,
    CONTACT_THRESHOLD_N,
    DISTRIBUTION_SIGMA_M,
    FORCE_FULL_SCALE_N,
    FSR_FULL_SCALE,
    SENSOR_LAYOUT,
    UNAFFECTED_SIDE,
    VirtualSensorSpec,
)

@dataclass(frozen=True)
class BoundSensor:
    spec: VirtualSensorSpec
    body_id: int
    anchor_geom_id: int


def bind_sensors(model) -> tuple[BoundSensor, ...]:
    bound = []
    for spec in SENSOR_LAYOUT:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body_name)
        anchor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, spec.anchor_geom_name)
        bound.append(BoundSensor(spec=spec, body_id=body_id, anchor_geom_id=anchor_geom_id))
    return tuple(bound)


def sensor_world_position(model, data, sensor: BoundSensor) -> np.ndarray:
    body_pos = data.xpos[sensor.body_id]
    body_rot = data.xmat[sensor.body_id].reshape(3, 3)
    geom_local = model.geom_pos[sensor.anchor_geom_id]
    sensor_local = geom_local + np.asarray(sensor.spec.offset_m, dtype=float)
    return body_pos + body_rot @ sensor_local


def contact_normal_force(model, data, contact_index: int) -> float:
    wrench = np.zeros(6, dtype=float)
    mujoco.mj_contactForce(model, data, contact_index, wrench)
    return max(0.0, float(wrench[0]))


def candidate_sensors_for_geom(bound_sensors: Iterable[BoundSensor], geom_name: str) -> list[BoundSensor]:
    return [sensor for sensor in bound_sensors if geom_name in sensor.spec.source_geoms]


def side_to_label_map(affected_side: str = AFFECTED_SIDE, unaffected_side: str = UNAFFECTED_SIDE) -> dict[str, str]:
    if affected_side == unaffected_side:
        raise ValueError("Affected and unaffected sides must be different.")
    return {
        affected_side: "A",
        unaffected_side: "U",
    }


def distribute_contact_force(
    model,
    data,
    contact_pos_world: np.ndarray,
    force_n: float,
    candidates: list[BoundSensor],
    sensor_forces: Dict[str, float],
    side_to_label: dict[str, str],
) -> None:
    if not candidates or force_n <= 0.0:
        return

    if len(candidates) == 1:
        channel = prefixed_channel_name(candidates[0].spec, side_to_label)
        sensor_forces[channel] += force_n
        return

    sensor_positions = np.array([sensor_world_position(model, data, sensor) for sensor in candidates])
    distances = np.linalg.norm(sensor_positions - contact_pos_world, axis=1)
    weights = np.exp(-(distances ** 2) / (2 * DISTRIBUTION_SIGMA_M ** 2))

    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(candidates), dtype=float)

    weights /= weights.sum()
    for sensor, weight in zip(candidates, weights):
        channel = prefixed_channel_name(sensor.spec, side_to_label)
        sensor_forces[channel] += force_n * float(weight)


def prefixed_channel_name(spec: VirtualSensorSpec, side_to_label: dict[str, str]) -> str:
    side_prefix = side_to_label[spec.foot_side]
    return f"{side_prefix}_{spec.channel_name}"


def force_to_fsr(force_n: float) -> int:
    scaled = round((max(0.0, min(force_n, FORCE_FULL_SCALE_N)) / FORCE_FULL_SCALE_N) * FSR_FULL_SCALE)
    return int(max(0, min(scaled, FSR_FULL_SCALE)))


def extract_virtual_fsr_sample(
    model,
    data,
    bound_sensors: tuple[BoundSensor, ...],
    affected_side: str = AFFECTED_SIDE,
    unaffected_side: str = UNAFFECTED_SIDE,
) -> dict:
    side_to_label = side_to_label_map(affected_side=affected_side, unaffected_side=unaffected_side)
    sensor_forces = defaultdict(float)

    # Initialize all configured channels so CSV rows are stable.
    for sensor in bound_sensors:
        sensor_forces[prefixed_channel_name(sensor.spec, side_to_label)] += 0.0

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    geom_names = {sensor.spec.anchor_geom_name for sensor in bound_sensors}
    geom_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in geom_names
    }
    geom_id_to_name = {geom_id: name for name, geom_id in geom_ids.items()}

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom_pair = {contact.geom1, contact.geom2}
        if floor_id not in geom_pair:
            continue

        other_geom_id = contact.geom2 if contact.geom1 == floor_id else contact.geom1
        geom_name = geom_id_to_name.get(other_geom_id)
        if geom_name is None:
            continue

        force_n = contact_normal_force(model, data, contact_index)
        candidates = candidate_sensors_for_geom(bound_sensors, geom_name)
        distribute_contact_force(
            model,
            data,
            np.asarray(contact.pos),
            force_n,
            candidates,
            sensor_forces,
            side_to_label,
        )

    sample = {}
    for channel in ("A_heel", "A_mid", "A_toe", "U_heel", "U_mid", "U_toe"):
        force_n = float(sensor_forces[channel])
        sample[f"{channel}_n"] = force_n
        sample[channel] = force_to_fsr(force_n)

    sample["A_total_n"] = sample["A_heel_n"] + sample["A_mid_n"] + sample["A_toe_n"]
    sample["U_total_n"] = sample["U_heel_n"] + sample["U_mid_n"] + sample["U_toe_n"]
    sample["A_contact"] = int(sample["A_total_n"] >= CONTACT_THRESHOLD_N)
    sample["U_contact"] = int(sample["U_total_n"] >= CONTACT_THRESHOLD_N)
    return sample
