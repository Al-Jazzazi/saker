#pragma once
// Everything you are likely to tune lives here.
#include <stdint.h>
#include "GaitFsm.h"

// ---- Network ----
// The sender and your laptop join the same hotspot. Its name and password are in
// secrets.h (not in git; copy secrets.example.h).
#include "secrets.h"
constexpr uint8_t WIFI_SSID_COUNT = sizeof(WIFI_SSIDS) / sizeof(WIFI_SSIDS[0]);
constexpr const char *MDNS_NAME = "saker";      // dashboard also at http://saker.local
constexpr uint16_t STREAM_PORT = 5005;          // UDP port used by laptop/saker_logger.py
// Also broadcast every sample to the whole hotspot, so the laptop only has to listen
// (saker_logger.py --listen). iPhone hotspots can drop laptop -> ESP traffic.
constexpr bool STREAM_BROADCAST = true;
constexpr uint16_t STREAM_BROADCAST_PORT = 5006;

// ---- IMUs: two MPU6050 on one I2C bus ----
constexpr uint8_t THIGH_ADDR = 0x69;            // AD0 -> 3V3
constexpr uint8_t SHANK_ADDR = 0x68;            // AD0 -> GND
constexpr uint8_t SDA_PIN = 21, SCL_PIN = 22;
constexpr uint32_t I2C_HZ = 100000;             // 100 kHz copes with long leg wires better than 400 kHz
constexpr uint32_t SAMPLE_MS = 10;              // 100 Hz
constexpr uint8_t IMU_FAILS_BEFORE_LOST = 5;    // consecutive bad reads (50 ms) before giving up
constexpr uint32_t IMU_RETRY_MS = 1000;         // how often to try reconnecting lost IMUs

// ---- Calibration ----
constexpr uint32_t STILL_CAL_SAMPLES = 300;     // 3 s at 100 Hz
constexpr float STILL_CAL_MAX_SD = 2.5f;        // deg/s; more spread than this means the leg moved
constexpr uint32_t WALK_CAL_MS = 12000;

// ---- Assist motion (the receiver also clamps to its own 45..135 limits) ----
constexpr uint16_t REST_ANGLE = 90;
constexpr int ASSIST_DELTA_DEG = -30;           // 30 degrees, toward 60
constexpr uint16_t ASSIST_MOVE_MS = 1500;       // time to move out; the return takes the same
// Healthy-leg event that starts an assist. At healthy heel strike the other leg
// is about to swing. Any GaitEvent except EVENT_NONE / EVENT_TIMEOUT works.
constexpr GaitEvent ASSIST_TRIGGER_EVENT = EVENT_HEEL_STRIKE;
