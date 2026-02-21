# FSR Gait Detector Project

## Directory Structure

```
capstone/
├── algorithms/              # Detection algorithms
│   ├── __init__.py
│   ├── hysteresis_detector.py   # Dual-threshold detector (noise resistant)
│   ├── threshold_detector.py    # Simple single-threshold detector
│   └── weighted_detector.py     # Weighted sum detector
│
├── simulation/              # Interactive testing & simulation
│   ├── gait_simulator.py        # GUI simulator (algorithm-agnostic)
│   ├── test_algorithms.py       # Easy algorithm comparison
│   └── gait_example.py          # Basic usage example
│
├── esp32/                   # ESP32 hardware code
│   ├── fsr_sensor_reader.ino    # Read FSR sensors
│   └── fsr_sensors.h            # Header for external access
│
└── README.md
```

## Quick Start: Interactive Simulator

From the `simulation/` directory:

```bash
cd simulation
python gait_simulator.py
```

Or test different algorithms from a menu:

```bash
cd simulation
python test_algorithms.py
```

### How to Use the Simulator

1. **Select algorithm** from the dropdown menu
2. **Drag sliders** to simulate FSR sensor readings (0-4095)
3. **Watch the output** for:
   - Contact states (which foot is touching ground)
   - Events (IC = Initial Contact, TO = Toe Off)
4. **Switch algorithms** on-the-fly to compare behavior

### Test Scenario

Try this to understand the algorithms:
- Drag "Affected Heel" slider to 2500 → see IC event
- Drag "Affected Mid" slider to 2500 → foot stays in contact
- Drag both back to 500 → see TO event
- Switch algorithms and repeat to see different behaviors

## Available Algorithms

### 1. **Hysteresis Detector** (`hysteresis_detector.py`)
- **Two thresholds**: Low (1000) and High (2000)
- **Behavior**: Contact starts at high threshold, breaks at low threshold
- **Best for**: Real-world noisy sensor data
- **Advantage**: Prevents flickering from sensor noise

### 2. **Threshold Detector** (`threshold_detector.py`)
- **One threshold**: 1000
- **Behavior**: Contact ON if value > threshold, OFF otherwise
- **Best for**: Clean data or testing
- **Limitation**: May flicker if sensor hovers near threshold

### 3. **Weighted Detector** (`weighted_detector.py`)
- **Weighted sum**: Heel (50%), Mid (30%), Toe (20%)
- **Threshold**: 1400
- **Behavior**: Combines all sensors with different importance
- **Best for**: Prioritizing certain sensors or different gait patterns

## Sensor Placement

The detectors expect **6 FSR sensors**:

**Affected Foot (A):**
- A_heel - Back of foot
- A_mid - Middle/arch
- A_toe - Front/ball of foot

**Unaffected Foot (U):**
- U_heel - Back of foot
- U_mid - Middle/arch
- U_toe - Front/ball of foot

## Creating Your Own Algorithm

All algorithms follow a simple interface:

```python
class MyDetector:
    def update(self, sample: dict) -> dict:
        """
        Args:
            sample: {"t": float, "A_heel": int, "A_mid": int, ...}
        Returns:
            {"t": float, "A_contact": int, "U_contact": int,
             "A_IC": int, "A_TO": int, "U_IC": int, "U_TO": int}
        """
        # Your detection logic here
        pass

    def reset(self) -> None:  # Optional
        # Reset internal state
        pass
```

Save in `algorithms/my_detector.py` and add to `__init__.py`:

```python
from .my_detector import MyDetector
__all__ = ['HysteresisDetector', 'ThresholdDetector', 'WeightedDetector', 'MyDetector']
```

Then use in simulator:

```python
from simulation.gait_simulator import main
from algorithms import MyDetector

detector = MyDetector()
main(detector, "My Algorithm Name")
```

## Tuning Constants

All algorithms use constants defined at the top of each file:

**Hysteresis Detector:**
- Modify `HYSTERESIS_LOW_THRESHOLD` and `HYSTERESIS_HIGH_THRESHOLD` in `hysteresis_detector.py`

**Threshold Detector:**
- Modify `DEFAULT_THRESHOLD` in `threshold_detector.py`

**Weighted Detector:**
- Modify `DEFAULT_THRESHOLD`, `DEFAULT_HEEL_WEIGHT`, `DEFAULT_MID_WEIGHT`, `DEFAULT_TOE_WEIGHT` in `weighted_detector.py`

## ESP32 Integration

The current ESP32 code reads 3 sensors. To integrate:

1. **Update hardware**: Connect 6 FSR sensors (3 per foot)
2. **Update ESP32 code**: Read all 6 sensors
3. **Send data**: Transmit to Python via Serial/WiFi/Bluetooth
4. **Format**: Send as `{"t": timestamp, "A_heel": value, ...}`

## Next Steps

1. ✅ Test different algorithms in the simulator
2. ✅ Understand how each algorithm behaves
3. Tune thresholds based on your actual FSR sensors
4. Connect ESP32 and integrate real sensor data
5. Collect real gait data and validate algorithms
6. Create custom algorithms for your specific use case
