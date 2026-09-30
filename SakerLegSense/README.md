# SakerLegSense

Sensor/measurement library for the SAKER knee-exoskeleton prototype.

This library intentionally stops at the **measurement layer**. It does not make gait-phase or motor-control decisions.

## Inputs

- FSR ADC readings from each foot
- affected-thigh IMU:
  - 3-axis acceleration in g
  - 3-axis gyroscope in deg/s
- affected-shank IMU:
  - 3-axis acceleration in g
  - 3-axis gyroscope in deg/s

The IMU hardware driver is intentionally outside this library, so the same measurement code can be used with MPU6050, ISM330DHCX, or another 6-DoF IMU.

## Outputs

### Foot / ground interaction

For each foot:

- normalized load proxy for each FSR
- total load proxy
- load rate
- contact state with hysteresis
- longitudinal center-of-pressure proxy

Across both feet:

- healthy-side load share
- healthy-minus-affected load difference
- support configuration:
  - neither
  - healthy only
  - affected only
  - both

### Thigh and shank

For each segment:

- acceleration XYZ
- angular velocity XYZ
- acceleration magnitude
- angular-speed magnitude
- jerk XYZ
- angular acceleration XYZ
- quaternion orientation
- roll / pitch / yaw

### Knee

- relative thigh-shank angle
- knee angular velocity
- knee angular acceleration

## Important limitations

1. **FSR outputs are load proxies unless calibrated against a known force reference.**
   `totalLoad` and `healthyLoadShare` should not be called Newtons or body-weight percentages without calibration.

2. **The COP value is only a sparse longitudinal proxy.**
   A few discrete FSRs are not equivalent to a force plate or pressure insole.

3. **IMU mounting alignment matters.**
   `KneeAxis::PitchY` is only correct if your physical IMU mounting makes the local Y-relative rotation correspond to knee flexion/extension.

4. **Two 6-DoF IMUs have yaw-drift limitations.**
   Knee flexion in the sagittal plane can still be useful, but mounting calibration and validation against an encoder or goniometer are required.

5. **Derivative features are noisy.**
   Jerk and angular acceleration will usually need low-pass filtering before being used by control or ML code.

6. **This is not a safety controller.**
   Motor limits, emergency stop logic, fault detection, and powered-human testing safeguards belong in a separate control/safety layer.

## Arduino library installation

Copy the `SakerLegSense` directory into:

`Documents/Arduino/libraries/`

Then restart the Arduino IDE and open:

`File -> Examples -> SakerLegSense -> BasicMeasurement`

## Recommended software architecture

```
hardware drivers
    |
    +-- FSR analogRead
    +-- thigh IMU driver
    +-- shank IMU driver
            |
            v
      SakerLegSense
      measurement layer
            |
            v
       logged state
            |
            +--> offline analysis
            +--> future FSM
            +--> future ML
            +--> future motor controller
```

Keep gait-state labels and actuator commands outside this library until the sensing model has been validated.
