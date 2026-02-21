# gait_detector.py

from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, List, Any

Channel = str
Thresholds = Dict[Channel, Tuple[float, float]]  # (T_low, T_high)

REQUIRED_CHANNELS = ("A_heel","A_mid","A_toe","U_heel","U_mid","U_toe")

class HysteresisContact:
    def __init__(self, t_low: float, t_high: float):
        if t_low >= t_high:
            raise ValueError("t_low must be < t_high")
        self.t_low = t_low
        self.t_high = t_high
        self.contact = 0  # 0/1

    def update(self, x: float) -> int:
        if self.contact == 0 and x >= self.t_high:
            self.contact = 1
        elif self.contact == 1 and x <= self.t_low:
            self.contact = 0
        return self.contact


class EdgeEventDetector:
    def __init__(self):
        self.prev = 0

    def update(self, contact: int) -> Tuple[int, int]:
        ic = 1 if (self.prev == 0 and contact == 1) else 0
        to = 1 if (self.prev == 1 and contact == 0) else 0
        self.prev = contact
        return ic, to


class GaitDetector:
    def __init__(self, thresholds: Thresholds):
        # validate config
        for ch in REQUIRED_CHANNELS:
            if ch not in thresholds:
                raise KeyError(f"Missing threshold for channel: {ch}")

        self.contacts = {
            ch: HysteresisContact(*thresholds[ch])  # (t_low, t_high)
            for ch in REQUIRED_CHANNELS
        }
        self.a_events = EdgeEventDetector()
        self.u_events = EdgeEventDetector()

    def reset(self) -> None:
        for c in self.contacts.values():
            c.contact = 0
        self.a_events.prev = 0
        self.u_events.prev = 0

    def update(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # sample must provide t and required channels
        t = sample["t"]

        # per-channel contact
        a_heel_c = self.contacts["A_heel"].update(sample["A_heel"])
        a_mid_c  = self.contacts["A_mid"].update(sample["A_mid"])
        a_toe_c  = self.contacts["A_toe"].update(sample["A_toe"])
        u_heel_c = self.contacts["U_heel"].update(sample["U_heel"])
        u_mid_c  = self.contacts["U_mid"].update(sample["U_mid"])
        u_toe_c  = self.contacts["U_toe"].update(sample["U_toe"])

        # per-foot contact (OR)
        a_contact = 1 if (a_heel_c or a_mid_c or a_toe_c) else 0
        u_contact = 1 if (u_heel_c or u_mid_c or u_toe_c) else 0

        # events
        a_ic, a_to = self.a_events.update(a_contact)
        u_ic, u_to = self.u_events.update(u_contact)

        return {
            "t": t,
            "A_contact": a_contact,
            "U_contact": u_contact,
            "A_IC": a_ic,
            "A_TO": a_to,
            "U_IC": u_ic,
            "U_TO": u_to,
        }


def run(detector: GaitDetector, samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [detector.update(s) for s in samples]
