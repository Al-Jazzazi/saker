from dataclasses import dataclass, field
from math import radians
from typing import Any, Dict, Iterable, List


CONTACT_OFF = 0
CONTACT_ON = 1

NO_EVENT = 0
EVENT_DETECTED = 1

REQUIRED_CHANNELS = ("A_heel", "A_mid", "A_toe", "U_heel", "U_mid", "U_toe")

STATE_LOADING_RESPONSE = "LOADING_RESPONSE"
STATE_MID_STANCE = "MID_STANCE"
STATE_TERMINAL_STANCE = "TERMINAL_STANCE"
STATE_SWING_FLEXION = "SWING_FLEXION"
STATE_SWING_EXTENSION = "SWING_EXTENSION"


@dataclass(frozen=True)
class HysteresisThreshold:
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError("Hysteresis low threshold must be smaller than high threshold.")


@dataclass(frozen=True)
class KneeAngleTargets:
    loading_response_deg: float = 15.0
    mid_stance_deg: float = 10.0
    terminal_stance_deg: float = 5.0
    swing_flexion_deg: float = 55.0
    swing_extension_deg: float = 10.0

    def for_state(self, state: str) -> float:
        return {
            STATE_LOADING_RESPONSE: self.loading_response_deg,
            STATE_MID_STANCE: self.mid_stance_deg,
            STATE_TERMINAL_STANCE: self.terminal_stance_deg,
            STATE_SWING_FLEXION: self.swing_flexion_deg,
            STATE_SWING_EXTENSION: self.swing_extension_deg,
        }[state]


@dataclass(frozen=True)
class KneeFSMConfig:
    heel_threshold: HysteresisThreshold = field(
        default_factory=lambda: HysteresisThreshold(20.0, 40.0)
    )
    mid_threshold: HysteresisThreshold = field(
        default_factory=lambda: HysteresisThreshold(15.0, 35.0)
    )
    toe_threshold: HysteresisThreshold = field(
        default_factory=lambda: HysteresisThreshold(20.0, 40.0)
    )
    foot_threshold: HysteresisThreshold = field(
        default_factory=lambda: HysteresisThreshold(50.0, 90.0)
    )
    min_state_time_s: float = 0.05
    swing_flexion_hold_s: float = 0.12
    swing_timeout_s: float = 1.20
    max_knee_rate_deg_per_s: float = 250.0
    knee_angles: KneeAngleTargets = field(default_factory=KneeAngleTargets)


class HysteresisContact:
    def __init__(self, threshold: HysteresisThreshold):
        self.low = threshold.low
        self.high = threshold.high
        self.contact = CONTACT_OFF

    def reset(self) -> None:
        self.contact = CONTACT_OFF

    def update(self, value: float) -> int:
        if self.contact == CONTACT_OFF and value >= self.high:
            self.contact = CONTACT_ON
        elif self.contact == CONTACT_ON and value <= self.low:
            self.contact = CONTACT_OFF
        return self.contact


class EdgeEventDetector:
    def __init__(self) -> None:
        self.prev_contact = CONTACT_OFF

    def reset(self) -> None:
        self.prev_contact = CONTACT_OFF

    def update(self, contact: int) -> tuple[int, int]:
        ic = EVENT_DETECTED if self.prev_contact == CONTACT_OFF and contact == CONTACT_ON else NO_EVENT
        to = EVENT_DETECTED if self.prev_contact == CONTACT_ON and contact == CONTACT_OFF else NO_EVENT
        self.prev_contact = contact
        return ic, to


class AffectedKneeFSMController:
    """
    Bilateral FSR-driven finite-state controller for the affected knee.

    Inputs:
      - sample["t"] in seconds
      - six FSR channels: A_heel, A_mid, A_toe, U_heel, U_mid, U_toe

    Output:
      - affected-knee target angle in degrees and radians
      - gait state
      - contact/event diagnostics for both feet

    State intent:
      LOADING_RESPONSE: affected foot just touched down, absorb weight.
      MID_STANCE: affected foot is flat and supporting body weight.
      TERMINAL_STANCE: affected heel is unloading, prepare for push-off.
      SWING_FLEXION: affected foot is off the ground, flex knee for clearance.
      SWING_EXTENSION: late swing, extend knee in preparation for heel strike.
    """

    def __init__(self, config: KneeFSMConfig | None = None):
        self.config = config or KneeFSMConfig()

        self.a_heel = HysteresisContact(self.config.heel_threshold)
        self.a_mid = HysteresisContact(self.config.mid_threshold)
        self.a_toe = HysteresisContact(self.config.toe_threshold)
        self.u_heel = HysteresisContact(self.config.heel_threshold)
        self.u_mid = HysteresisContact(self.config.mid_threshold)
        self.u_toe = HysteresisContact(self.config.toe_threshold)

        self.a_foot = HysteresisContact(self.config.foot_threshold)
        self.u_foot = HysteresisContact(self.config.foot_threshold)
        self.a_events = EdgeEventDetector()
        self.u_events = EdgeEventDetector()

        self.state = STATE_SWING_EXTENSION
        self.last_t: float | None = None
        self.last_state_change_t: float | None = None
        self.knee_target_deg = self.config.knee_angles.for_state(self.state)

    def reset(self) -> None:
        for contact in (
            self.a_heel,
            self.a_mid,
            self.a_toe,
            self.u_heel,
            self.u_mid,
            self.u_toe,
            self.a_foot,
            self.u_foot,
        ):
            contact.reset()

        self.a_events.reset()
        self.u_events.reset()
        self.state = STATE_SWING_EXTENSION
        self.last_t = None
        self.last_state_change_t = None
        self.knee_target_deg = self.config.knee_angles.for_state(self.state)

    def update(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_sample(sample)

        t = float(sample["t"])

        a_total = float(sample["A_heel"]) + float(sample["A_mid"]) + float(sample["A_toe"])
        u_total = float(sample["U_heel"]) + float(sample["U_mid"]) + float(sample["U_toe"])

        a_heel_on = self.a_heel.update(float(sample["A_heel"]))
        a_mid_on = self.a_mid.update(float(sample["A_mid"]))
        a_toe_on = self.a_toe.update(float(sample["A_toe"]))
        u_heel_on = self.u_heel.update(float(sample["U_heel"]))
        u_mid_on = self.u_mid.update(float(sample["U_mid"]))
        u_toe_on = self.u_toe.update(float(sample["U_toe"]))

        a_contact = self.a_foot.update(a_total)
        u_contact = self.u_foot.update(u_total)

        a_ic, a_to = self.a_events.update(a_contact)
        u_ic, u_to = self.u_events.update(u_contact)

        initialized = self.last_t is not None
        if not initialized:
            self.state = self._bootstrap_state(
                a_contact=a_contact,
                u_contact=u_contact,
                a_heel_on=a_heel_on,
                a_mid_on=a_mid_on,
                a_toe_on=a_toe_on,
                u_heel_on=u_heel_on,
                u_mid_on=u_mid_on,
                u_toe_on=u_toe_on,
            )
            self.last_state_change_t = t

        state_before = self.state
        elapsed_in_state = 0.0 if self.last_state_change_t is None else max(0.0, t - self.last_state_change_t)

        a_flat = (a_mid_on == CONTACT_ON) or (a_heel_on == CONTACT_ON and a_toe_on == CONTACT_ON)
        a_toe_dominant = a_toe_on == CONTACT_ON and a_heel_on == CONTACT_OFF
        u_terminal_pattern = u_toe_on == CONTACT_ON and u_heel_on == CONTACT_OFF

        next_state, transition_reason = self._next_state(
            t=t,
            elapsed_in_state=elapsed_in_state,
            a_contact=a_contact,
            u_contact=u_contact,
            a_ic=a_ic,
            a_to=a_to,
            a_flat=a_flat,
            a_toe_dominant=a_toe_dominant,
            u_terminal_pattern=u_terminal_pattern,
        )

        if next_state != self.state:
            self.state = next_state
            self.last_state_change_t = t
            elapsed_in_state = 0.0
        else:
            transition_reason = ""

        state_target_deg = self.config.knee_angles.for_state(self.state)
        self.knee_target_deg = self._rate_limit_target(
            requested_deg=state_target_deg,
            t=t,
            initialized=initialized,
        )
        self.last_t = t

        return {
            "t": t,
            "state": self.state,
            "state_changed": 1 if self.state != state_before else 0,
            "transition_reason": transition_reason,
            "state_target_deg": state_target_deg,
            "A_knee_target_deg": self.knee_target_deg,
            "A_knee_target_rad": radians(self.knee_target_deg),
            "A_total": a_total,
            "U_total": u_total,
            "A_contact": a_contact,
            "U_contact": u_contact,
            "A_IC": a_ic,
            "A_TO": a_to,
            "U_IC": u_ic,
            "U_TO": u_to,
            "A_heel_contact": a_heel_on,
            "A_mid_contact": a_mid_on,
            "A_toe_contact": a_toe_on,
            "U_heel_contact": u_heel_on,
            "U_mid_contact": u_mid_on,
            "U_toe_contact": u_toe_on,
            "state_time_s": elapsed_in_state,
        }

    def _validate_sample(self, sample: Dict[str, Any]) -> None:
        if "t" not in sample:
            raise KeyError("Sample must include 't' in seconds.")
        for channel in REQUIRED_CHANNELS:
            if channel not in sample:
                raise KeyError(f"Missing FSR channel '{channel}'.")

    def _bootstrap_state(
        self,
        *,
        a_contact: int,
        u_contact: int,
        a_heel_on: int,
        a_mid_on: int,
        a_toe_on: int,
        u_heel_on: int,
        u_mid_on: int,
        u_toe_on: int,
    ) -> str:
        if a_contact == CONTACT_ON:
            if a_toe_on == CONTACT_ON and a_heel_on == CONTACT_OFF:
                return STATE_TERMINAL_STANCE
            if a_heel_on == CONTACT_ON and a_mid_on == CONTACT_OFF and a_toe_on == CONTACT_OFF:
                return STATE_LOADING_RESPONSE
            return STATE_MID_STANCE

        if u_contact == CONTACT_ON and u_toe_on == CONTACT_ON and u_heel_on == CONTACT_OFF:
            return STATE_SWING_EXTENSION

        return STATE_SWING_FLEXION

    def _next_state(
        self,
        *,
        t: float,
        elapsed_in_state: float,
        a_contact: int,
        u_contact: int,
        a_ic: int,
        a_to: int,
        a_flat: bool,
        a_toe_dominant: bool,
        u_terminal_pattern: bool,
    ) -> tuple[str, str]:
        if self.state == STATE_LOADING_RESPONSE:
            if a_to == EVENT_DETECTED:
                return STATE_SWING_FLEXION, "affected_foot_unloaded_early"
            if self._can_change(elapsed_in_state) and a_flat:
                return STATE_MID_STANCE, "affected_foot_flat"
            return self.state, ""

        if self.state == STATE_MID_STANCE:
            if a_to == EVENT_DETECTED:
                return STATE_SWING_FLEXION, "affected_toe_off"
            if self._can_change(elapsed_in_state) and a_toe_dominant:
                return STATE_TERMINAL_STANCE, "heel_unloaded_toe_dominant"
            return self.state, ""

        if self.state == STATE_TERMINAL_STANCE:
            if a_to == EVENT_DETECTED or (a_contact == CONTACT_OFF and u_contact == CONTACT_ON):
                return STATE_SWING_FLEXION, "affected_entered_swing"
            return self.state, ""

        if self.state == STATE_SWING_FLEXION:
            if a_ic == EVENT_DETECTED or a_contact == CONTACT_ON:
                return STATE_LOADING_RESPONSE, "affected_initial_contact"
            if self._can_change(elapsed_in_state) and (
                elapsed_in_state >= self.config.swing_flexion_hold_s or u_terminal_pattern
            ):
                return STATE_SWING_EXTENSION, "prepare_for_next_heel_strike"
            return self.state, ""

        if self.state == STATE_SWING_EXTENSION:
            if a_ic == EVENT_DETECTED or a_contact == CONTACT_ON:
                return STATE_LOADING_RESPONSE, "affected_initial_contact"
            if elapsed_in_state >= self.config.swing_timeout_s:
                return STATE_SWING_EXTENSION, "swing_timeout_hold"
            return self.state, ""

        raise ValueError(f"Unknown controller state '{self.state}' at time {t}.")

    def _can_change(self, elapsed_in_state: float) -> bool:
        return elapsed_in_state >= self.config.min_state_time_s

    def _rate_limit_target(self, *, requested_deg: float, t: float, initialized: bool) -> float:
        if not initialized or self.last_t is None:
            return requested_deg

        dt = max(0.0, t - self.last_t)
        max_step = self.config.max_knee_rate_deg_per_s * dt
        delta = requested_deg - self.knee_target_deg

        if delta > max_step:
            delta = max_step
        elif delta < -max_step:
            delta = -max_step

        return self.knee_target_deg + delta


def run(
    controller: AffectedKneeFSMController, samples: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    return [controller.update(sample) for sample in samples]
