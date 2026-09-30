#pragma once
#include <Arduino.h>

namespace saker {

static constexpr uint8_t MAX_FSR_PER_FOOT = 4;

struct Vec3 {
  float x;
  float y;
  float z;

  Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
  Vec3(float xValue, float yValue, float zValue)
      : x(xValue), y(yValue), z(zValue) {}
};

struct Quaternion {
  float w;
  float x;
  float y;
  float z;

  Quaternion() : w(1.0f), x(0.0f), y(0.0f), z(0.0f) {}
  Quaternion(float wValue, float xValue, float yValue, float zValue)
      : w(wValue), x(xValue), y(yValue), z(zValue) {}
};

struct EulerDeg {
  float roll = 0.0f;
  float pitch = 0.0f;
  float yaw = 0.0f;
};

struct ImuSample {
  // Acceleration in g. Gyroscope in deg/s.
  Vec3 accelG;
  Vec3 gyroDps;
};

enum class KneeAxis : uint8_t {
  RollX = 0,
  PitchY = 1,
  YawZ = 2
};

enum class SupportState : uint8_t {
  Neither = 0,
  HealthyOnly,
  AffectedOnly,
  Both
};

struct FSRCalibration {
  uint16_t adcMin;
  uint16_t adcMax;

  // Contact hysteresis thresholds on normalized [0,1] load.
  // "on" should be > "off".
  float contactOn;
  float contactOff;

  // Longitudinal location along foot:
  // 0.0 = heel, 1.0 = toe.
  float position;

  FSRCalibration()
      : adcMin(0), adcMax(4095), contactOn(0.15f),
        contactOff(0.08f), position(0.5f) {}

  FSRCalibration(uint16_t minValue, uint16_t maxValue,
                 float onThreshold, float offThreshold,
                 float longitudinalPosition)
      : adcMin(minValue), adcMax(maxValue), contactOn(onThreshold),
        contactOff(offThreshold), position(longitudinalPosition) {}
};

struct FootState {
  uint8_t count = 0;
  float sensorLoad[MAX_FSR_PER_FOOT] = {0};
  float totalLoad = 0.0f;
  float loadRate = 0.0f;        // normalized-load units / second
  float copLongitudinal = 0.5f; // proxy only; 0=heel, 1=toe
  bool contact = false;
};

struct SegmentState {
  Vec3 accelG;
  Vec3 gyroDps;
  Vec3 jerkGPerS;
  Vec3 angularAccelDps2;

  float accelMagnitudeG = 0.0f;
  float angularSpeedDps = 0.0f;

  Quaternion orientation;
  EulerDeg eulerDeg;
};

struct KneeState {
  float angleDeg = 0.0f;
  float angularVelocityDps = 0.0f;
  float angularAccelDps2 = 0.0f;
};

struct BilateralState {
  float healthyLoadShare = 0.5f; // H / (H + A), if denominator > 0
  float loadDifference = 0.0f;   // healthy - affected
  SupportState support = SupportState::Neither;
};

struct MeasurementState {
  FootState healthyFoot;
  FootState affectedFoot;
  SegmentState thigh;
  SegmentState shank;
  KneeState knee;
  BilateralState bilateral;
};

class Madgwick6DOF {
public:
  explicit Madgwick6DOF(float beta = 0.08f);
  void reset();
  void setBeta(float beta);
  void update(const Vec3& accelG, const Vec3& gyroDps, float dtSeconds);
  Quaternion quaternion() const;

private:
  float _beta;
  Quaternion _q;
};

class FSRFootModel {
public:
  FSRFootModel();

  void configure(uint8_t count, const FSRCalibration* calibration);
  void reset();

  // adcValues must contain at least count() elements.
  FootState update(const uint16_t* adcValues, float dtSeconds);

  uint8_t count() const { return _count; }

private:
  uint8_t _count = 0;
  FSRCalibration _cal[MAX_FSR_PER_FOOT];
  FootState _state;

  float normalize(uint16_t adc, const FSRCalibration& cal) const;
};

class LegKinematics {
public:
  LegKinematics();

  void setKneeAxis(KneeAxis axis);
  void reset();
  void calibrateNeutral(const Quaternion& thighQ, const Quaternion& shankQ);

  SegmentState updateThigh(const ImuSample& sample, float dtSeconds);
  SegmentState updateShank(const ImuSample& sample, float dtSeconds);
  KneeState updateKnee(const Quaternion& thighQ,
                       const Quaternion& shankQ,
                       float dtSeconds);

  const Quaternion& thighQuaternion() const { return _thighFilterQ; }
  const Quaternion& shankQuaternion() const { return _shankFilterQ; }

  Madgwick6DOF& thighFilter() { return _thighFilter; }
  Madgwick6DOF& shankFilter() { return _shankFilter; }

private:
  KneeAxis _axis = KneeAxis::PitchY;

  Madgwick6DOF _thighFilter;
  Madgwick6DOF _shankFilter;

  Quaternion _thighFilterQ;
  Quaternion _shankFilterQ;

  SegmentState _thighState;
  SegmentState _shankState;
  KneeState _kneeState;

  Vec3 _prevThighAccel;
  Vec3 _prevShankAccel;
  Vec3 _prevThighGyro;
  Vec3 _prevShankGyro;

  float _neutralAngleDeg = 0.0f;
  bool _neutralCalibrated = false;

  float relativeAxisAngleDeg(const Quaternion& thighQ,
                             const Quaternion& shankQ) const;
};

class SakerLegSense {
public:
  SakerLegSense();

  void configureHealthyFoot(uint8_t count, const FSRCalibration* calibration);
  void configureAffectedFoot(uint8_t count, const FSRCalibration* calibration);

  void setKneeAxis(KneeAxis axis);
  void setMadgwickBeta(float beta);
  void reset();

  // Call while the wearer is in the desired neutral/reference pose.
  void calibrateKneeNeutral();

  // Main update. FSR ADC arrays must match configured counts.
  // IMU accel units: g. Gyro units: deg/s.
  MeasurementState update(const uint16_t* healthyFsrAdc,
                          const uint16_t* affectedFsrAdc,
                          const ImuSample& thighImu,
                          const ImuSample& shankImu,
                          float dtSeconds);

  const MeasurementState& state() const { return _state; }

private:
  FSRFootModel _healthyFoot;
  FSRFootModel _affectedFoot;
  LegKinematics _kinematics;
  MeasurementState _state;

  BilateralState computeBilateral(const FootState& healthy,
                                  const FootState& affected) const;
};

// Utility functions
float magnitude(const Vec3& v);
Quaternion quatMultiply(const Quaternion& a, const Quaternion& b);
Quaternion quatConjugate(const Quaternion& q);
Quaternion quatNormalize(const Quaternion& q);
EulerDeg quaternionToEulerDeg(const Quaternion& q);

} // namespace saker
