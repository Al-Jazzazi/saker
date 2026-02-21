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

### How to Use the Simulator

1. **Select algorithm** from the dropdown menu
2. **Drag sliders** to simulate FSR sensor readings (0-4095)
3. **Watch the output** for:
   - Contact states (which foot is touching ground)
   - Events (IC = Initial Contact, TO = Toe Off)
4. **Switch algorithms** on-the-fly to compare behavior


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


## Papers of interests 
(Gait phase analysis based on a Hidden Markov Model)[https://www.sciencedirect.com/science/article/pii/S0957415811000481]
(Unsupervised Square Finite State Machine for Gait Events Estimation from Instrumented Insoles)[https://link.springer.com/chapter/10.1007/978-3-031-47508-5_22]
(A Survey of Wearable Lower Extremity
Neurorehabilitation Exoskeleton: Sensing, Gait
Dynamics, and Human–Robot Collaboration)[https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10465662]