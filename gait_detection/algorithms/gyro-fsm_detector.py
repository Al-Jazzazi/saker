from typing import Dict, Any, Tuple
import math

# Contact states
CONTACT_OFF = 0
CONTACT_ON = 1

# Event states
NO_EVENT = 0
EVENT_DETECTED = 1

# States (per foot)
STATE_SWING = "SWING"
STATE_EARLY_STANCE = "EARLY_STANCE"   # heel strike / loading response
STATE_MID_STANCE = "MID_STANCE"       # foot flat / mid-stance
STATE_LATE_STANCE = "LATE_STANCE"     # push-off / terminal stance

REQUIRED_CHANNELS = ("A_heel","A_mid","A_toe","U_heel","U_mid","U_toe")

# Defaults (tune)
DEFAULT_FSR_LOW = 900
DEFAULT_FSR_HIGH = 1100

DEFAULT_LOAD_LOW = 1200
DEFAULT_LOAD_HIGH = 1400

DEFAULT_GYRO_LOW = 40     # e.g., deg/s or rad/s magnitude threshold low
DEFAULT_GYRO_HIGH = 80    # high threshold for "moving"

DEFAULT_MIN_STATE_MS = 80
DEFAULT_TIMEOUT_MS = 2500

DEFAULT_HEEL_WEIGHT = 0.5
DEFAULT_MID_WEIGHT = 0.3
DEFAULT_TOE_WEIGHT = 0.2


class HysteresisContact:
    def __init__(self, t_low: float, t_high: float):
        if t_low >= t_high:
            raise ValueError("t_low must be < t_high")
        self.t_low = t_low
        self.t_high = t_high
        self.contact = CONTACT_OFF

    def update(self, x: float) -> int:
        if self.contact == CONTACT_OFF and x >= self.t_high:
            self.contact = CONTACT_ON
        elif self.contact == CONTACT_ON and x <= self.t_low:
            self.contact = CONTACT_OFF
        return self.contact


def gyro_mag(sample: Dict[str, Any], key: str = "gyro") -> float:
    """
    Accepts either:
      - sample["gyro"] as scalar
      - sample["gyro"] as (gx, gy, gz)
      - or sample has "gx","gy","gz"
    """
    if key in sample:
        g = sample[key]
        if isinstance(g, (tuple, list)) and len(g) == 3:
            return float(math.sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]))
        return float(abs(g))
    if all(k in sample for k in ("gx","gy","gz")):
        gx, gy, gz = float(sample["gx"]), float(sample["gy"]), float(sample["gz"])
        return float(math.sqrt(gx*gx + gy*gy + gz*gz))
    raise KeyError("Missing gyro signal: provide 'gyro' or ('gx','gy','gz').")


class OrderedFootFSM:
    """
    Order-aware foot phase detector:
      SWING -> (EARLY|MID|LATE)_STANCE based on first sensor(s)
      EARLY -> MID when mid appears
      MID   -> LATE when heel unloads while toe/mid remain
      Any STANCE -> SWING (TO) only when load drops AND gyro indicates movement
    """
    def __init__(
        self,
        fsr_thresholds: Dict[str, Tuple[float, float]],   # {"heel":(low,high), "mid":..., "toe":...}
        load_low: float = DEFAULT_LOAD_LOW,
        load_high: float = DEFAULT_LOAD_HIGH,
        gyro_low: float = DEFAULT_GYRO_LOW,
        gyro_high: float = DEFAULT_GYRO_HIGH,
        min_state_ms: int = DEFAULT_MIN_STATE_MS,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        weights: Dict[str, float] = None,
    ):
        self.heel_c = HysteresisContact(*fsr_thresholds["heel"])
        self.mid_c  = HysteresisContact(*fsr_thresholds["mid"])
        self.toe_c  = HysteresisContact(*fsr_thresholds["toe"])

        self.load_c = HysteresisContact(load_low, load_high)
        self.move_c = HysteresisContact(gyro_low, gyro_high)  # 1 => moving (swing-like)

        self.weights = weights or {"heel": DEFAULT_HEEL_WEIGHT, "mid": DEFAULT_MID_WEIGHT, "toe": DEFAULT_TOE_WEIGHT}

        self.state = STATE_SWING
        self.last_change_t = None  # timestamp units must match sample["t"]
        self.min_state_ms = int(min_state_ms)
        self.timeout_ms = int(timeout_ms)

    def reset(self) -> None:
        self.heel_c.contact = CONTACT_OFF
        self.mid_c.contact = CONTACT_OFF
        self.toe_c.contact = CONTACT_OFF
        self.load_c.contact = CONTACT_OFF
        self.move_c.contact = CONTACT_OFF
        self.state = STATE_SWING
        self.last_change_t = None

    def _can_change(self, t: float) -> bool:
        if self.last_change_t is None:
            return True
        return (t - self.last_change_t) >= self.min_state_ms

    def update(self, t: float, heel: float, mid: float, toe: float, gmag: float) -> Dict[str, Any]:
        # Per-sensor hysteresis contacts
        heel_on = self.heel_c.update(heel)
        mid_on  = self.mid_c.update(mid)
        toe_on  = self.toe_c.update(toe)

        # Weighted load + hysteresis contact
        load = heel * self.weights["heel"] + mid * self.weights["mid"] + toe * self.weights["toe"]
        foot_on = self.load_c.update(load)

        # Movement gate from gyro magnitude
        moving = self.move_c.update(gmag)  # 1=moving, 0=not moving

        ic = NO_EVENT
        to = NO_EVENT

        # Timeout recovery
        if self.last_change_t is not None and (t - self.last_change_t) > self.timeout_ms:
            self.state = STATE_SWING
            self.last_change_t = t

        if self.state == STATE_SWING:
            # Enter stance only when: not moving (swing ending) AND load says contact
            if self._can_change(t) and moving == CONTACT_OFF and foot_on == CONTACT_ON:
                # Choose entry state based on which sensor led (order cue)
                if heel_on == CONTACT_ON:
                    self.state = STATE_EARLY_STANCE
                elif mid_on == CONTACT_ON and toe_on == CONTACT_ON:
                    self.state = STATE_MID_STANCE  # flat-foot landing
                elif toe_on == CONTACT_ON:
                    self.state = STATE_LATE_STANCE  # toe-first landing (impaired pattern)
                else:
                    self.state = STATE_EARLY_STANCE
                ic = EVENT_DETECTED
                self.last_change_t = t

        elif self.state == STATE_EARLY_STANCE:
            if self._can_change(t):
                if mid_on == CONTACT_ON:
                    self.state = STATE_MID_STANCE
                    self.last_change_t = t
                elif toe_on == CONTACT_ON and heel_on == CONTACT_OFF:
                    self.state = STATE_LATE_STANCE
                    self.last_change_t = t

            # Toe-off: require BOTH contact loss and movement starting
            if self._can_change(t) and foot_on == CONTACT_OFF and moving == CONTACT_ON:
                to = EVENT_DETECTED
                self.state = STATE_SWING
                self.last_change_t = t

        elif self.state == STATE_MID_STANCE:
            if self._can_change(t):
                # heel unload while toe/mid remain => late stance
                if heel_on == CONTACT_OFF and (toe_on == CONTACT_ON or mid_on == CONTACT_ON):
                    self.state = STATE_LATE_STANCE
                    self.last_change_t = t

            if self._can_change(t) and foot_on == CONTACT_OFF and moving == CONTACT_ON:
                to = EVENT_DETECTED
                self.state = STATE_SWING
                self.last_change_t = t

        elif self.state == STATE_LATE_STANCE:
            # expect toe dominance; leave only when foot off AND moving
            if self._can_change(t) and foot_on == CONTACT_OFF and moving == CONTACT_ON:
                to = EVENT_DETECTED
                self.state = STATE_SWING
                self.last_change_t = t

        return {
            "state": self.state,
            "IC": ic,
            "TO": to,
            "contact": foot_on,
            "heel_on": heel_on,
            "mid_on": mid_on,
            "toe_on": toe_on,
            "load": load,
            "moving": moving,
            "gyro_mag": gmag,
        }


class OrderedGaitDetector:
    """
    Matches the other detectors' API: update(sample)->dict with A/U contact + IC/TO.
    Requires gyro in sample as 'gyro' or (gx,gy,gz) or 'gx','gy','gz'.
    """
    def __init__(
        self,
        fsr_thresholds: Dict[str, Dict[str, Tuple[float, float]]] = None,
        load_low: float = DEFAULT_LOAD_LOW,
        load_high: float = DEFAULT_LOAD_HIGH,
        gyro_low: float = DEFAULT_GYRO_LOW,
        gyro_high: float = DEFAULT_GYRO_HIGH,
        min_state_ms: int = DEFAULT_MIN_STATE_MS,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        weights: Dict[str, float] = None,
    ):
        # Default same hysteresis for all sensors if not provided
        fsr_thresholds = fsr_thresholds or {
            "heel": (DEFAULT_FSR_LOW, DEFAULT_FSR_HIGH),
            "mid":  (DEFAULT_FSR_LOW, DEFAULT_FSR_HIGH),
            "toe":  (DEFAULT_FSR_LOW, DEFAULT_FSR_HIGH),
        }

        self.a_fsm = OrderedFootFSM(
            fsr_thresholds=fsr_thresholds,
            load_low=load_low, load_high=load_high,
            gyro_low=gyro_low, gyro_high=gyro_high,
            min_state_ms=min_state_ms, timeout_ms=timeout_ms,
            weights=weights,
        )
        self.u_fsm = OrderedFootFSM(
            fsr_thresholds=fsr_thresholds,
            load_low=load_low, load_high=load_high,
            gyro_low=gyro_low, gyro_high=gyro_high,
            min_state_ms=min_state_ms, timeout_ms=timeout_ms,
            weights=weights,
        )

    def reset(self) -> None:
        self.a_fsm.reset()
        self.u_fsm.reset()

    def update(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        t = sample["t"]
        for ch in REQUIRED_CHANNELS:
            if ch not in sample:
                raise KeyError(f"Missing channel in sample: {ch}")

        gmag = gyro_mag(sample)

        a = self.a_fsm.update(t, sample["A_heel"], sample["A_mid"], sample["A_toe"], gmag)
        u = self.u_fsm.update(t, sample["U_heel"], sample["U_mid"], sample["U_toe"], gmag)

        return {
            "t": t,
            "A_contact": a["contact"],
            "U_contact": u["contact"],
            "A_IC": a["IC"],
            "A_TO": a["TO"],
            "U_IC": u["IC"],
            "U_TO": u["TO"],
            # optional debug/phase outputs (remove if you want identical shape)
            "A_state": a["state"],
            "U_state": u["state"],
            "gyro_mag": gmag,
        }