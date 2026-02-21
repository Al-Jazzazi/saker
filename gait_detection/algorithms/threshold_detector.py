from typing import Dict, Any

# Contact states
CONTACT_OFF = 0
CONTACT_ON = 1

# Event states
NO_EVENT = 0
EVENT_DETECTED = 1

# Default threshold
DEFAULT_THRESHOLD = 1000

class ThresholdDetector:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.prev_a_contact = CONTACT_OFF
        self.prev_u_contact = CONTACT_OFF

    def reset(self) -> None:
        self.prev_a_contact = CONTACT_OFF
        self.prev_u_contact = CONTACT_OFF

    def update(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        t = sample["t"]

        # Simple threshold check (no hysteresis)
        a_heel = CONTACT_ON if sample["A_heel"] > self.threshold else CONTACT_OFF
        a_mid  = CONTACT_ON if sample["A_mid"] > self.threshold else CONTACT_OFF
        a_toe  = CONTACT_ON if sample["A_toe"] > self.threshold else CONTACT_OFF

        u_heel = CONTACT_ON if sample["U_heel"] > self.threshold else CONTACT_OFF
        u_mid  = CONTACT_ON if sample["U_mid"] > self.threshold else CONTACT_OFF
        u_toe  = CONTACT_ON if sample["U_toe"] > self.threshold else CONTACT_OFF

        # OR aggregation per foot
        a_contact = CONTACT_ON if (a_heel or a_mid or a_toe) else CONTACT_OFF
        u_contact = CONTACT_ON if (u_heel or u_mid or u_toe) else CONTACT_OFF

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
