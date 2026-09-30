#include "SakerLegSense.h"
#include <math.h>

namespace saker {

static constexpr float DEG_TO_RAD_F = 0.01745329251994329577f;
static constexpr float RAD_TO_DEG_F = 57.295779513082320876f;
static constexpr float EPS = 1e-9f;

static float clamp01(float v) {
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

static float safeDt(float dt) {
  return dt > 1e-6f ? dt : 1e-6f;
}

static float axisValue(const EulerDeg& e, KneeAxis axis) {
  switch (axis) {
    case KneeAxis::RollX:  return e.roll;
    case KneeAxis::PitchY: return e.pitch;
    case KneeAxis::YawZ:   return e.yaw;
  }
  return e.pitch;
}

float magnitude(const Vec3& v) {
  return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

Quaternion quatMultiply(const Quaternion& a, const Quaternion& b) {
  Quaternion r;
  r.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
  r.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
  r.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
  r.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
  return r;
}

Quaternion quatConjugate(const Quaternion& q) {
  Quaternion r;
  r.w = q.w;
  r.x = -q.x;
  r.y = -q.y;
  r.z = -q.z;
  return r;
}

Quaternion quatNormalize(const Quaternion& q) {
  const float n = sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
  if (n < EPS) return Quaternion{};
  Quaternion r;
  r.w = q.w / n;
  r.x = q.x / n;
  r.y = q.y / n;
  r.z = q.z / n;
  return r;
}

EulerDeg quaternionToEulerDeg(const Quaternion& qIn) {
  const Quaternion q = quatNormalize(qIn);
  EulerDeg e;

  const float sinr_cosp = 2.0f * (q.w*q.x + q.y*q.z);
  const float cosr_cosp = 1.0f - 2.0f * (q.x*q.x + q.y*q.y);
  e.roll = atan2f(sinr_cosp, cosr_cosp) * RAD_TO_DEG_F;

  const float sinp = 2.0f * (q.w*q.y - q.z*q.x);
  if (fabsf(sinp) >= 1.0f) {
    e.pitch = copysignf(90.0f, sinp);
  } else {
    e.pitch = asinf(sinp) * RAD_TO_DEG_F;
  }

  const float siny_cosp = 2.0f * (q.w*q.z + q.x*q.y);
  const float cosy_cosp = 1.0f - 2.0f * (q.y*q.y + q.z*q.z);
  e.yaw = atan2f(siny_cosp, cosy_cosp) * RAD_TO_DEG_F;

  return e;
}

// ---------------- Madgwick6DOF ----------------

Madgwick6DOF::Madgwick6DOF(float beta)
: _beta(beta) {
  reset();
}

void Madgwick6DOF::reset() {
  _q = Quaternion{};
}

void Madgwick6DOF::setBeta(float beta) {
  _beta = beta;
}

Quaternion Madgwick6DOF::quaternion() const {
  return _q;
}

void Madgwick6DOF::update(const Vec3& accelG,
                          const Vec3& gyroDps,
                          float dtSeconds) {
  const float dt = safeDt(dtSeconds);

  float q0 = _q.w;
  float q1 = _q.x;
  float q2 = _q.y;
  float q3 = _q.z;

  const float gx = gyroDps.x * DEG_TO_RAD_F;
  const float gy = gyroDps.y * DEG_TO_RAD_F;
  const float gz = gyroDps.z * DEG_TO_RAD_F;

  float qDot0 = 0.5f * (-q1*gx - q2*gy - q3*gz);
  float qDot1 = 0.5f * ( q0*gx + q2*gz - q3*gy);
  float qDot2 = 0.5f * ( q0*gy - q1*gz + q3*gx);
  float qDot3 = 0.5f * ( q0*gz + q1*gy - q2*gx);

  float ax = accelG.x;
  float ay = accelG.y;
  float az = accelG.z;

  const float aNorm = sqrtf(ax*ax + ay*ay + az*az);
  if (aNorm > EPS) {
    ax /= aNorm;
    ay /= aNorm;
    az /= aNorm;

    const float _2q0 = 2.0f*q0;
    const float _2q1 = 2.0f*q1;
    const float _2q2 = 2.0f*q2;
    const float _2q3 = 2.0f*q3;
    const float _4q0 = 4.0f*q0;
    const float _4q1 = 4.0f*q1;
    const float _4q2 = 4.0f*q2;
    const float _8q1 = 8.0f*q1;
    const float _8q2 = 8.0f*q2;
    const float q0q0 = q0*q0;
    const float q1q1 = q1*q1;
    const float q2q2 = q2*q2;
    const float q3q3 = q3*q3;

    float s0 = _4q0*q2q2 + _2q2*ax + _4q0*q1q1 - _2q1*ay;
    float s1 = _4q1*q3q3 - _2q3*ax + 4.0f*q0q0*q1
             - _2q0*ay - _4q1 + _8q1*q1q1 + _8q1*q2q2 + _4q1*az;
    float s2 = 4.0f*q0q0*q2 + _2q0*ax + _4q2*q3q3
             - _2q3*ay - _4q2 + _8q2*q1q1 + _8q2*q2q2 + _4q2*az;
    float s3 = 4.0f*q1q1*q3 - _2q1*ax + 4.0f*q2q2*q3 - _2q2*ay;

    const float sNorm = sqrtf(s0*s0 + s1*s1 + s2*s2 + s3*s3);
    if (sNorm > EPS) {
      s0 /= sNorm;
      s1 /= sNorm;
      s2 /= sNorm;
      s3 /= sNorm;

      qDot0 -= _beta*s0;
      qDot1 -= _beta*s1;
      qDot2 -= _beta*s2;
      qDot3 -= _beta*s3;
    }
  }

  q0 += qDot0*dt;
  q1 += qDot1*dt;
  q2 += qDot2*dt;
  q3 += qDot3*dt;

  _q = quatNormalize({q0, q1, q2, q3});
}

// ---------------- FSRFootModel ----------------

FSRFootModel::FSRFootModel() {
  reset();
}

void FSRFootModel::configure(uint8_t count,
                             const FSRCalibration* calibration) {
  _count = count > MAX_FSR_PER_FOOT ? MAX_FSR_PER_FOOT : count;
  for (uint8_t i = 0; i < _count; ++i) {
    _cal[i] = calibration[i];
  }
  reset();
  _state.count = _count;
}

void FSRFootModel::reset() {
  _state = FootState{};
  _state.count = _count;
}

float FSRFootModel::normalize(uint16_t adc,
                              const FSRCalibration& cal) const {
  if (cal.adcMax <= cal.adcMin) return 0.0f;
  const float v = (static_cast<float>(adc) - cal.adcMin) /
                  (static_cast<float>(cal.adcMax) - cal.adcMin);
  return clamp01(v);
}

FootState FSRFootModel::update(const uint16_t* adcValues,
                               float dtSeconds) {
  const float dt = safeDt(dtSeconds);
  const float previousTotal = _state.totalLoad;

  _state.totalLoad = 0.0f;
  float weightedPosition = 0.0f;

  bool anyAboveOn = false;
  bool allBelowOff = true;

  for (uint8_t i = 0; i < _count; ++i) {
    const float load = normalize(adcValues[i], _cal[i]);
    _state.sensorLoad[i] = load;
    _state.totalLoad += load;
    weightedPosition += load * _cal[i].position;

    if (load >= _cal[i].contactOn) anyAboveOn = true;
    if (load > _cal[i].contactOff) allBelowOff = false;
  }

  if (_state.totalLoad > EPS) {
    _state.copLongitudinal = weightedPosition / _state.totalLoad;
  }

  // Hysteresis: once contact is ON, it stays on until all sensors
  // are below their OFF thresholds.
  if (!_state.contact && anyAboveOn) {
    _state.contact = true;
  } else if (_state.contact && allBelowOff) {
    _state.contact = false;
  }

  _state.loadRate = (_state.totalLoad - previousTotal) / dt;
  return _state;
}

// ---------------- LegKinematics ----------------

LegKinematics::LegKinematics()
: _thighFilter(0.08f), _shankFilter(0.08f) {
  reset();
}

void LegKinematics::setKneeAxis(KneeAxis axis) {
  _axis = axis;
}

void LegKinematics::reset() {
  _thighFilter.reset();
  _shankFilter.reset();

  _thighFilterQ = Quaternion{};
  _shankFilterQ = Quaternion{};

  _thighState = SegmentState{};
  _shankState = SegmentState{};
  _kneeState = KneeState{};

  _prevThighAccel = Vec3{};
  _prevShankAccel = Vec3{};
  _prevThighGyro = Vec3{};
  _prevShankGyro = Vec3{};

  _neutralAngleDeg = 0.0f;
  _neutralCalibrated = false;
}

void LegKinematics::calibrateNeutral(const Quaternion& thighQ,
                                     const Quaternion& shankQ) {
  _neutralAngleDeg = relativeAxisAngleDeg(thighQ, shankQ);
  _neutralCalibrated = true;
  _kneeState = KneeState{};
}

static SegmentState updateSegmentCommon(const ImuSample& sample,
                                        const Quaternion& q,
                                        const Vec3& prevAccel,
                                        const Vec3& prevGyro,
                                        float dtSeconds) {
  const float dt = safeDt(dtSeconds);
  SegmentState s;
  s.accelG = sample.accelG;
  s.gyroDps = sample.gyroDps;
  s.accelMagnitudeG = magnitude(sample.accelG);
  s.angularSpeedDps = magnitude(sample.gyroDps);
  s.orientation = q;
  s.eulerDeg = quaternionToEulerDeg(q);

  s.jerkGPerS.x = (sample.accelG.x - prevAccel.x) / dt;
  s.jerkGPerS.y = (sample.accelG.y - prevAccel.y) / dt;
  s.jerkGPerS.z = (sample.accelG.z - prevAccel.z) / dt;

  s.angularAccelDps2.x = (sample.gyroDps.x - prevGyro.x) / dt;
  s.angularAccelDps2.y = (sample.gyroDps.y - prevGyro.y) / dt;
  s.angularAccelDps2.z = (sample.gyroDps.z - prevGyro.z) / dt;

  return s;
}

SegmentState LegKinematics::updateThigh(const ImuSample& sample,
                                        float dtSeconds) {
  _thighFilter.update(sample.accelG, sample.gyroDps, dtSeconds);
  _thighFilterQ = _thighFilter.quaternion();

  _thighState = updateSegmentCommon(sample, _thighFilterQ,
                                    _prevThighAccel, _prevThighGyro,
                                    dtSeconds);
  _prevThighAccel = sample.accelG;
  _prevThighGyro = sample.gyroDps;
  return _thighState;
}

SegmentState LegKinematics::updateShank(const ImuSample& sample,
                                        float dtSeconds) {
  _shankFilter.update(sample.accelG, sample.gyroDps, dtSeconds);
  _shankFilterQ = _shankFilter.quaternion();

  _shankState = updateSegmentCommon(sample, _shankFilterQ,
                                    _prevShankAccel, _prevShankGyro,
                                    dtSeconds);
  _prevShankAccel = sample.accelG;
  _prevShankGyro = sample.gyroDps;
  return _shankState;
}

float LegKinematics::relativeAxisAngleDeg(const Quaternion& thighQ,
                                          const Quaternion& shankQ) const {
  // Relative orientation from thigh frame to shank frame.
  const Quaternion relative =
      quatNormalize(quatMultiply(quatConjugate(thighQ), shankQ));

  const EulerDeg e = quaternionToEulerDeg(relative);
  return axisValue(e, _axis);
}

KneeState LegKinematics::updateKnee(const Quaternion& thighQ,
                                    const Quaternion& shankQ,
                                    float dtSeconds) {
  const float dt = safeDt(dtSeconds);
  const float rawAngle = relativeAxisAngleDeg(thighQ, shankQ);
  const float angle = rawAngle - (_neutralCalibrated ? _neutralAngleDeg : 0.0f);

  const float previousAngle = _kneeState.angleDeg;
  const float previousVelocity = _kneeState.angularVelocityDps;

  _kneeState.angleDeg = angle;
  _kneeState.angularVelocityDps = (angle - previousAngle) / dt;
  _kneeState.angularAccelDps2 =
      (_kneeState.angularVelocityDps - previousVelocity) / dt;

  return _kneeState;
}

// ---------------- SakerLegSense ----------------

SakerLegSense::SakerLegSense() {
  reset();
}

void SakerLegSense::configureHealthyFoot(
    uint8_t count,
    const FSRCalibration* calibration) {
  _healthyFoot.configure(count, calibration);
}

void SakerLegSense::configureAffectedFoot(
    uint8_t count,
    const FSRCalibration* calibration) {
  _affectedFoot.configure(count, calibration);
}

void SakerLegSense::setKneeAxis(KneeAxis axis) {
  _kinematics.setKneeAxis(axis);
}

void SakerLegSense::setMadgwickBeta(float beta) {
  _kinematics.thighFilter().setBeta(beta);
  _kinematics.shankFilter().setBeta(beta);
}

void SakerLegSense::reset() {
  _healthyFoot.reset();
  _affectedFoot.reset();
  _kinematics.reset();
  _state = MeasurementState{};
}

void SakerLegSense::calibrateKneeNeutral() {
  _kinematics.calibrateNeutral(
      _kinematics.thighQuaternion(),
      _kinematics.shankQuaternion());
}

BilateralState SakerLegSense::computeBilateral(
    const FootState& healthy,
    const FootState& affected) const {
  BilateralState b;
  const float total = healthy.totalLoad + affected.totalLoad;

  if (total > EPS) {
    b.healthyLoadShare = healthy.totalLoad / total;
  } else {
    b.healthyLoadShare = 0.5f;
  }

  b.loadDifference = healthy.totalLoad - affected.totalLoad;

  if (healthy.contact && affected.contact) {
    b.support = SupportState::Both;
  } else if (healthy.contact) {
    b.support = SupportState::HealthyOnly;
  } else if (affected.contact) {
    b.support = SupportState::AffectedOnly;
  } else {
    b.support = SupportState::Neither;
  }

  return b;
}

MeasurementState SakerLegSense::update(
    const uint16_t* healthyFsrAdc,
    const uint16_t* affectedFsrAdc,
    const ImuSample& thighImu,
    const ImuSample& shankImu,
    float dtSeconds) {

  _state.healthyFoot =
      _healthyFoot.update(healthyFsrAdc, dtSeconds);
  _state.affectedFoot =
      _affectedFoot.update(affectedFsrAdc, dtSeconds);

  _state.thigh =
      _kinematics.updateThigh(thighImu, dtSeconds);
  _state.shank =
      _kinematics.updateShank(shankImu, dtSeconds);

  _state.knee =
      _kinematics.updateKnee(_kinematics.thighQuaternion(),
                             _kinematics.shankQuaternion(),
                             dtSeconds);

  _state.bilateral =
      computeBilateral(_state.healthyFoot, _state.affectedFoot);

  return _state;
}

} // namespace saker
