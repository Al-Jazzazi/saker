"""
Interactive Gait Detector Simulator
Drag sliders to simulate FSR sensor readings and see detector behavior in real-time
Switch between different algorithms on-the-fly
"""

import tkinter as tk
from tkinter import ttk
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from algorithms import HysteresisDetector, ThresholdDetector, WeightedDetector

# ================= Flag Values ====================

# Sensor Configuration
SENSOR_MIN_VALUE = 0
SENSOR_MAX_VALUE = 4095  # ESP32 12-bit ADC
SENSOR_CHANNELS = ("A_heel", "A_mid", "A_toe", "U_heel", "U_mid", "U_toe")

# Algorithm Parameters - Hysteresis
HYSTERESIS_LOW_THRESHOLD = 1000
HYSTERESIS_HIGH_THRESHOLD = 2000

# Algorithm Parameters - Simple Threshold
SIMPLE_THRESHOLD = 1000

# Algorithm Parameters - Weighted
WEIGHTED_THRESHOLD = 1400
WEIGHTED_HEEL_WEIGHT = 0.5
WEIGHTED_MID_WEIGHT = 0.3
WEIGHTED_TOE_WEIGHT = 0.2



# ================ Simulation CONSTANTS ====================

# Timing
TIME_INCREMENT = 0.01  # seconds

# GUI Configuration
WINDOW_WIDTH = 850
WINDOW_HEIGHT = 750
WINDOW_TITLE = "FSR Gait Detector Simulator"

# GUI Fonts
FONT_TITLE = ("Arial", 16, "bold")
FONT_SECTION = ("Arial", 12, "bold")
FONT_LABEL = ("Arial", 11, "bold")
FONT_NORMAL = ("Arial", 10)
FONT_SMALL = ("Arial", 9)
FONT_MONOSPACE = ("Courier", 10)

# GUI Colors
COLOR_INFO = "blue"
COLOR_GRAY = "gray"

# GUI Layout
PADDING_LARGE = 20
PADDING_MEDIUM = 10
PADDING_SMALL = 5
SLIDER_LENGTH = 250
VALUE_LABEL_WIDTH = 5
EVENT_DISPLAY_HEIGHT = 6
MAX_EVENTS_DISPLAYED = 10

# ==================== SIMULATOR CLASS ====================

class GaitSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        # Available algorithms
        self.algorithms = {
            "Hysteresis Detector": self.create_hysteresis_detector,
            "Simple Threshold Detector": self.create_simple_detector,
            "Weighted Detector": self.create_weighted_detector,
        }

        # Current detector
        self.detector = None
        self.detector_name = tk.StringVar(value="Hysteresis Detector")

        # Sensor values
        self.sensor_values = {
            channel: tk.IntVar(value=SENSOR_MIN_VALUE)
            for channel in SENSOR_CHANNELS
        }

        # Time counter
        self.time = 0.0

        # Event history
        self.event_history = []

        # Initialize with default detector
        self.switch_algorithm("Hysteresis Detector")

        self.create_widgets()

    def create_hysteresis_detector(self):
        """Create hysteresis-based detector"""
        thresholds = {
            channel: (HYSTERESIS_LOW_THRESHOLD, HYSTERESIS_HIGH_THRESHOLD)
            for channel in SENSOR_CHANNELS
        }
        return HysteresisDetector(thresholds)

    def create_simple_detector(self):
        """Create simple threshold detector"""
        return ThresholdDetector(threshold=SIMPLE_THRESHOLD)

    def create_weighted_detector(self):
        """Create weighted detector"""
        return WeightedDetector(
            threshold=WEIGHTED_THRESHOLD,
            weights={
                "heel": WEIGHTED_HEEL_WEIGHT,
                "mid": WEIGHTED_MID_WEIGHT,
                "toe": WEIGHTED_TOE_WEIGHT
            }
        )

    def switch_algorithm(self, algorithm_name=None):
        """Switch to a different detection algorithm"""
        if algorithm_name is None:
            algorithm_name = self.detector_name.get()

        # Create new detector
        self.detector = self.algorithms[algorithm_name]()
        self.time = 0.0

        # Update display if widgets exist
        if hasattr(self, 'algo_info_label'):
            self.update_algorithm_info()

    def update_algorithm_info(self):
        """Update the algorithm information display"""
        algo_name = self.detector_name.get()
        info_text = ""

        if algo_name == "Hysteresis Detector":
            info_text = f"Two thresholds: Low={HYSTERESIS_LOW_THRESHOLD}, High={HYSTERESIS_HIGH_THRESHOLD} | Prevents noise/flickering"
        elif algo_name == "Simple Threshold Detector":
            info_text = f"Single threshold: {SIMPLE_THRESHOLD} | Simple but may flicker"
        elif algo_name == "Weighted Detector":
            heel_pct = int(WEIGHTED_HEEL_WEIGHT * 100)
            mid_pct = int(WEIGHTED_MID_WEIGHT * 100)
            toe_pct = int(WEIGHTED_TOE_WEIGHT * 100)
            info_text = f"Weighted sum: Heel={heel_pct}%, Mid={mid_pct}%, Toe={toe_pct}% | Threshold={WEIGHTED_THRESHOLD}"

        self.algo_info_label.config(text=info_text)

    def create_widgets(self):
        """Create all GUI widgets"""
        # Title
        title = tk.Label(self.root, text=WINDOW_TITLE, font=FONT_TITLE)
        title.pack(pady=PADDING_MEDIUM)

        # Algorithm selector
        algo_frame = tk.LabelFrame(self.root, text="Algorithm Selection",
                                   font=FONT_LABEL, padx=PADDING_MEDIUM, pady=PADDING_MEDIUM)
        algo_frame.pack(fill=tk.X, padx=PADDING_LARGE, pady=PADDING_SMALL)

        tk.Label(algo_frame, text="Algorithm:", font=FONT_NORMAL).pack(side=tk.LEFT, padx=PADDING_SMALL)

        algo_dropdown = ttk.Combobox(algo_frame, textvariable=self.detector_name,
                                     values=list(self.algorithms.keys()),
                                     state="readonly", width=30, font=FONT_NORMAL)
        algo_dropdown.pack(side=tk.LEFT, padx=PADDING_SMALL)
        algo_dropdown.bind("<<ComboboxSelected>>", lambda e: self.on_algorithm_change())

        # Algorithm info label
        self.algo_info_label = tk.Label(algo_frame, text="", font=FONT_SMALL, fg=COLOR_INFO)
        self.algo_info_label.pack(side=tk.LEFT, padx=PADDING_MEDIUM)
        self.update_algorithm_info()

        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=PADDING_LARGE, pady=PADDING_MEDIUM)

        # Left side: Affected foot (A)
        left_frame = tk.LabelFrame(main_frame, text="Affected Foot (A)",
                                   font=FONT_SECTION, padx=PADDING_MEDIUM, pady=PADDING_MEDIUM)
        left_frame.grid(row=0, column=0, padx=PADDING_MEDIUM, pady=PADDING_SMALL, sticky="nsew")

        self.create_sensor_slider(left_frame, "A_heel", "Heel", 0)
        self.create_sensor_slider(left_frame, "A_mid", "Midfoot", 1)
        self.create_sensor_slider(left_frame, "A_toe", "Toe", 2)

        # Right side: Unaffected foot (U)
        right_frame = tk.LabelFrame(main_frame, text="Unaffected Foot (U)",
                                    font=FONT_SECTION, padx=PADDING_MEDIUM, pady=PADDING_MEDIUM)
        right_frame.grid(row=0, column=1, padx=PADDING_MEDIUM, pady=PADDING_SMALL, sticky="nsew")

        self.create_sensor_slider(right_frame, "U_heel", "Heel", 0)
        self.create_sensor_slider(right_frame, "U_mid", "Midfoot", 1)
        self.create_sensor_slider(right_frame, "U_toe", "Toe", 2)

        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Output display
        output_frame = tk.LabelFrame(self.root, text="Detector Output",
                                     font=FONT_SECTION, padx=PADDING_MEDIUM, pady=PADDING_MEDIUM)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=PADDING_LARGE, pady=PADDING_MEDIUM)

        # Contact states
        self.contact_label = tk.Label(output_frame, text="", font=FONT_LABEL,
                                     justify=tk.LEFT, anchor="w")
        self.contact_label.pack(fill=tk.X, pady=PADDING_SMALL)

        # Events display
        events_label = tk.Label(output_frame, text="Recent Events:", font=FONT_NORMAL)
        events_label.pack(anchor="w")

        self.events_text = tk.Text(output_frame, height=EVENT_DISPLAY_HEIGHT, font=FONT_MONOSPACE)
        self.events_text.pack(fill=tk.BOTH, expand=True)

        # Control buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=PADDING_MEDIUM)

        reset_btn = tk.Button(button_frame, text="Reset Detector",
                             command=self.reset_detector, font=FONT_NORMAL)
        reset_btn.pack(side=tk.LEFT, padx=PADDING_SMALL)

        clear_btn = tk.Button(button_frame, text="Clear Events",
                             command=self.clear_events, font=FONT_NORMAL)
        clear_btn.pack(side=tk.LEFT, padx=PADDING_SMALL)

    def create_sensor_slider(self, parent, sensor_name, label_text, row):
        """Create a sensor slider widget"""
        # Label
        label = tk.Label(parent, text=label_text, font=FONT_NORMAL)
        label.grid(row=row, column=0, sticky="w", pady=PADDING_SMALL)

        # Slider
        slider = tk.Scale(parent, from_=SENSOR_MIN_VALUE, to=SENSOR_MAX_VALUE,
                         orient=tk.HORIZONTAL,
                         variable=self.sensor_values[sensor_name],
                         command=lambda v: self.update_detector(),
                         length=SLIDER_LENGTH)
        slider.grid(row=row, column=1, padx=PADDING_MEDIUM, pady=PADDING_SMALL)

        # Value display
        value_label = tk.Label(parent, textvariable=self.sensor_values[sensor_name],
                              font=FONT_NORMAL, width=VALUE_LABEL_WIDTH)
        value_label.grid(row=row, column=2, sticky="w")

    def on_algorithm_change(self):
        """Called when user changes algorithm from dropdown"""
        self.switch_algorithm()
        self.clear_events()
        self.update_algorithm_info()
        self.update_detector()

    def update_detector(self):
        """Update detector with current sensor values"""
        # Build sample
        sample = {
            "t": self.time,
            **{channel: self.sensor_values[channel].get() for channel in SENSOR_CHANNELS}
        }

        # Update detector
        output = self.detector.update(sample)

        # Update display
        self.display_output(output)

        # Increment time
        self.time += TIME_INCREMENT

    def display_output(self, output):
        """Display detector output"""
        # Contact states
        a_contact_str = "■ CONTACT" if output["A_contact"] else "□ No contact"
        u_contact_str = "■ CONTACT" if output["U_contact"] else "□ No contact"

        contact_text = f"Affected Foot:   {a_contact_str}\n"
        contact_text += f"Unaffected Foot: {u_contact_str}"

        self.contact_label.config(text=contact_text)

        # Events
        events = []
        if output["A_IC"]:
            events.append(f"[{output['t']:.2f}s] Affected foot: INITIAL CONTACT (IC)")
        if output["A_TO"]:
            events.append(f"[{output['t']:.2f}s] Affected foot: TOE OFF (TO)")
        if output["U_IC"]:
            events.append(f"[{output['t']:.2f}s] Unaffected foot: INITIAL CONTACT (IC)")
        if output["U_TO"]:
            events.append(f"[{output['t']:.2f}s] Unaffected foot: TOE OFF (TO)")

        # Add events to history
        for event in events:
            self.event_history.append(event)

        # Display last N events
        self.events_text.delete(1.0, tk.END)
        for event in self.event_history[-MAX_EVENTS_DISPLAYED:]:
            self.events_text.insert(tk.END, event + "\n")
        self.events_text.see(tk.END)

    def reset_detector(self):
        """Reset detector state"""
        self.detector.reset()
        self.time = 0.0
        self.update_detector()

    def clear_events(self):
        """Clear event history"""
        self.event_history = []
        self.events_text.delete(1.0, tk.END)


# ==================== MAIN ====================

def main():
    """Run the simulator"""
    root = tk.Tk()
    app = GaitSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
