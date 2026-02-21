from typing import Dict, Any

# Contact states
CONTACT_OFF = 0
CONTACT_ON = 1

# Event states
NO_EVENT = 0
EVENT_DETECTED = 1

# Default threshold
DEFAULT_THRESHOLD = 1400

# Default weights for sensors
DEFAULT_HEEL_WEIGHT = 0.5
DEFAULT_MID_WEIGHT = 0.3
DEFAULT_TOE_WEIGHT = 0.2

class WeightedDetector:

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, weights: Dict[str, float] = None):
        self.threshold = threshold
        self.weights = weights or {
            "heel": DEFAULT_HEEL_WEIGHT,
            "mid": DEFAULT_MID_WEIGHT,
            "toe": DEFAULT_TOE_WEIGHT
        }
        self.prev_a_contact = CONTACT_OFF
        self.prev_u_contact = CONTACT_OFF

    def reset(self) -> None:
        self.prev_a_contact = CONTACT_OFF
        self.prev_u_contact = CONTACT_OFF

    def update(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        t = sample["t"]

        # Weighted sum for each foot
        a_sum = (sample["A_heel"] * self.weights["heel"] +
                 sample["A_mid"] * self.weights["mid"] +
                 sample["A_toe"] * self.weights["toe"])

        u_sum = (sample["U_heel"] * self.weights["heel"] +
                 sample["U_mid"] * self.weights["mid"] +
                 sample["U_toe"] * self.weights["toe"])

        # Contact based on weighted sum
        a_contact = CONTACT_ON if a_sum > self.threshold else CONTACT_OFF
        u_contact = CONTACT_ON if u_sum > self.threshold else CONTACT_OFF

        # Detect edges (IC/TO)
        a_ic = EVENT_DETECTED if (self.prev_a_contact == CONTACT_OFF and a_contact == CONTACT_ON) else NO_EVENT
        a_to = EVENT_DETECTED if (self.prev_a_contact == CONTACT_ON and a_contact == CONTACT_OFF) else NO_EVENT
        u_ic = EVENT_DETECTED if (self.prev_u_contact == CONTACT_OFF and u_contact == CONTACT_ON) else NO_EVENT
        u_to = EVENT_DETECTED if (self.prev_u_contact == CONTACT_ON and u_contact == CONTACT_OFF) else NO_EVENT

        # Update state
        self.prev_a_contact = a_contact
        self.prev_u_contact = u_contact

        return {
            "t": t,
            "A_contact": a_contact,
            "U_contact": u_contact,
            "A_IC": a_ic,
            "A_TO": a_to,
            "U_IC": u_ic,
            "U_TO": u_to,
        }
