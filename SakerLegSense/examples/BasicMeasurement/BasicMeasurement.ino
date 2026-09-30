#include <Arduino.h>
#include <SakerLegSense.h>

using namespace saker;

SakerLegSense sensors;

// Example only. Replace with your actual ESP32 ADC pins.
const uint8_t healthyPins[]  = {34, 35, 32}; // heel, midfoot, toe
const uint8_t affectedPins[] = {33, 25, 26}; // heel, midfoot, toe

constexpr uint8_t HEALTHY_COUNT = 3;
constexpr uint8_t AFFECTED_COUNT = 3;

// Longitudinal positions are only used for the COP proxy.
// Calibrate adcMin/adcMax and thresholds for your installed sensors.
FSRCalibration healthyCal[HEALTHY_COUNT] = {
  {0, 4095, 0.15f, 0.08f, 0.00f}, // heel
  {0, 4095, 0.15f, 0.08f, 0.50f}, // midfoot
  {0, 4095, 0.15f, 0.08f, 1.00f}  // toe
};

FSRCalibration affectedCal[AFFECTED_COUNT] = {
  {0, 4095, 0.15f, 0.08f, 0.00f},
  {0, 4095, 0.15f, 0.08f, 0.50f},
  {0, 4095, 0.15f, 0.08f, 1.00f}
};

// Replace these two functions with calls to your actual IMU driver.
// IMPORTANT: return acceleration in g and gyro in deg/s.
ImuSample readThighIMU() {
  ImuSample s;
  // Example:
  // s.accelG = {ax, ay, az};
  // s.gyroDps = {gx, gy, gz};
  return s;
}

ImuSample readShankIMU() {
  ImuSample s;
  return s;
}

void setup() {
  Serial.begin(115200);

  sensors.configureHealthyFoot(HEALTHY_COUNT, healthyCal);
  sensors.configureAffectedFoot(AFFECTED_COUNT, affectedCal);

  // Change if your mounted IMUs use a different anatomical axis.
  sensors.setKneeAxis(KneeAxis::PitchY);

  // Tune after testing. Lower = trust gyro more, higher = trust accel more.
  sensors.setMadgwickBeta(0.08f);

  delay(1000);
}

void loop() {
  static uint32_t lastUs = micros();
  static bool neutralDone = false;
  static uint32_t startMs = millis();

  const uint32_t nowUs = micros();
  const float dt = (nowUs - lastUs) * 1e-6f;
  lastUs = nowUs;

  uint16_t healthyAdc[HEALTHY_COUNT];
  uint16_t affectedAdc[AFFECTED_COUNT];

  for (uint8_t i = 0; i < HEALTHY_COUNT; ++i) {
    healthyAdc[i] = analogRead(healthyPins[i]);
  }

  for (uint8_t i = 0; i < AFFECTED_COUNT; ++i) {
    affectedAdc[i] = analogRead(affectedPins[i]);
  }

  ImuSample thigh = readThighIMU();
  ImuSample shank = readShankIMU();

  MeasurementState s =
      sensors.update(healthyAdc, affectedAdc, thigh, shank, dt);

  // Let the filters settle, then establish the straight-knee reference.
  // In real use, trigger this only when the wearer is in a known neutral pose.
  if (!neutralDone && millis() - startMs > 3000) {
    sensors.calibrateKneeNeutral();
    neutralDone = true;
  }

  Serial.print("knee_deg=");
  Serial.print(s.knee.angleDeg, 2);

  Serial.print(", knee_vel_dps=");
  Serial.print(s.knee.angularVelocityDps, 2);

  Serial.print(", H_load=");
  Serial.print(s.healthyFoot.totalLoad, 3);

  Serial.print(", A_load=");
  Serial.print(s.affectedFoot.totalLoad, 3);

  Serial.print(", H_share=");
  Serial.print(s.bilateral.healthyLoadShare, 3);

  Serial.print(", A_COP=");
  Serial.print(s.affectedFoot.copLongitudinal, 3);

  Serial.print(", support=");
  Serial.println(static_cast<int>(s.bilateral.support));

  delay(5); // ~200 Hz loop for a first test
}
