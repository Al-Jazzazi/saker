#pragma once
// Minimal MPU6050 driver: +-4 g, +-500 deg/s, 44 Hz low-pass filter.
#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include "Config.h"

struct Imu {
  uint8_t address;
  float accel[3] = {0, 0, 1};  // g
  float gyro[3] = {0, 0, 0};   // deg/s, bias removed
  float bias[3] = {0, 0, 0};   // deg/s, measured by the stationary CAL
  float gyroMag = 0, accelMag = 1;
  // Which gyro axis is sagittal and which sign is "swinging forward" (set by WALK).
  uint8_t sagAxis = 0;
  int8_t sagSign = 1;

  float sagittalRate() const { return sagSign * gyro[sagAxis]; }
};

namespace mpu {
constexpr uint8_t REG_CONFIG = 0x1A, REG_GYRO_CONFIG = 0x1B, REG_ACCEL_CONFIG = 0x1C;
constexpr uint8_t REG_DATA = 0x3B, REG_PWR_MGMT_1 = 0x6B, REG_WHO_AM_I = 0x75;
constexpr float ACCEL_LSB_PER_G = 8192.0f;  // +-4 g
constexpr float GYRO_LSB_PER_DPS = 65.5f;   // +-500 deg/s
}  // namespace mpu

bool imuWriteReg(uint8_t address, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool imuInit(uint8_t address) {
  Wire.beginTransmission(address);
  Wire.write(mpu::REG_WHO_AM_I);
  if (Wire.endTransmission(false) != 0 || Wire.requestFrom((int)address, 1) != 1) return false;
  // 0x68 is a genuine MPU6050; 0x70/0x72/0x98 are common clone chips sold as MPU6050.
  const uint8_t id = Wire.read();
  if (id != 0x68 && id != 0x70 && id != 0x72 && id != 0x98) return false;
  return imuWriteReg(address, mpu::REG_PWR_MGMT_1, 0x01)      // wake up, clock from gyro PLL
         && imuWriteReg(address, mpu::REG_CONFIG, 0x03)       // 44 Hz low-pass
         && imuWriteReg(address, mpu::REG_GYRO_CONFIG, 0x08)  // +-500 deg/s
         && imuWriteReg(address, mpu::REG_ACCEL_CONFIG, 0x08);  // +-4 g
}

bool imuRead(Imu &s) {
  Wire.beginTransmission(s.address);
  Wire.write(mpu::REG_DATA);
  if (Wire.endTransmission(false) != 0 || Wire.requestFrom((int)s.address, 14) != 14) return false;
  auto next = []() -> int16_t {
    const int hi = Wire.read(), lo = Wire.read();
    return (int16_t)((hi << 8) | lo);
  };
  int16_t raw[7];  // ax ay az temperature gx gy gz
  for (int16_t &v : raw) v = next();
  for (int i = 0; i < 3; i++) {
    s.accel[i] = raw[i] / mpu::ACCEL_LSB_PER_G;
    s.gyro[i] = raw[i + 4] / mpu::GYRO_LSB_PER_DPS - s.bias[i];
  }
  s.gyroMag = sqrtf(s.gyro[0] * s.gyro[0] + s.gyro[1] * s.gyro[1] + s.gyro[2] * s.gyro[2]);
  s.accelMag = sqrtf(s.accel[0] * s.accel[0] + s.accel[1] * s.accel[1] + s.accel[2] * s.accel[2]);
  return isfinite(s.gyroMag) && isfinite(s.accelMag);
}

// (Re)starts the I2C bus and configures both sensors. Also used to recover
// after a wire comes loose.
bool imuStartBus(Imu &a, Imu &b) {
  Wire.end();
  Wire.begin(SDA_PIN, SCL_PIN, I2C_HZ);
  Wire.setTimeOut(20);
  return imuInit(a.address) && imuInit(b.address);
}
