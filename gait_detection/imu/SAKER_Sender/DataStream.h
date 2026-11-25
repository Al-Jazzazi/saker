#pragma once
// Streams every IMU sample to the laptop over UDP (laptop/saker_logger.py).
// The logger sends "SUB" about once a second and gets the CSV header back;
// samples flow to it until 3 s pass without a SUB, or it sends "UNSUB".
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "Config.h"
#include "Imu.h"
#include "Log.h"

constexpr uint32_t STREAM_SUB_TIMEOUT_MS = 3000;
constexpr uint8_t STREAM_SAMPLES_PER_PACKET = 5;  // 20 packets/s at 100 Hz

// One CSV row per sample. Gyro in deg/s (bias removed), accel in g.
// *_sag = signed sagittal rate, empty until WALK has found the axes.
const char STREAM_HEADER[] =
    "t_ms,seq,mode,phase,event,assist,"
    "thigh_ax,thigh_ay,thigh_az,thigh_gx,thigh_gy,thigh_gz,"
    "shank_ax,shank_ay,shank_az,shank_gx,shank_gy,shank_gz,"
    "thigh_sag,shank_sag";

struct DataStream {
  bool serial = false;  // laptop logger on the USB cable sent "SUB"
  WiFiUDP udp;
  bool started = false;
  IPAddress client;
  uint16_t clientPort = 0;
  uint32_t lastSubMs = 0;
  uint32_t lastBroadcastHeaderMs = 0;
  uint32_t seq = 0;
  char buf[1400];
  size_t len = 0;
  uint8_t rows = 0;
};
DataStream stream;

void streamBegin() { stream.started = stream.udp.begin(STREAM_PORT); }

bool streamActive(uint32_t now) {
  return stream.clientPort != 0 && now - stream.lastSubMs < STREAM_SUB_TIMEOUT_MS;
}

bool streamBroadcasting(uint32_t now) {
  return STREAM_BROADCAST && !streamActive(now) && WiFi.status() == WL_CONNECTED;
}

String streamClientText(uint32_t now) {
  if (streamActive(now)) return stream.client.toString();
  return streamBroadcasting(now) ? String("broadcast") : String("");
}

// Subscribed laptop if there is one, otherwise everyone on the hotspot.
void streamBeginPacket(uint32_t now) {
  if (streamActive(now)) stream.udp.beginPacket(stream.client, stream.clientPort);
  else stream.udp.beginPacket(WiFi.broadcastIP(), STREAM_BROADCAST_PORT);
}

// Handles SUB / UNSUB messages from the laptop.
void streamPoll(uint32_t now) {
  if (!stream.started) return;
  for (int size = stream.udp.parsePacket(); size > 0; size = stream.udp.parsePacket()) {
    char msg[8] = {};
    stream.udp.read(msg, sizeof(msg) - 1);
    if (strncmp(msg, "SUB", 3) == 0) {
      const bool isNew = !streamActive(now) || stream.udp.remoteIP() != stream.client;
      stream.client = stream.udp.remoteIP();
      stream.clientPort = stream.udp.remotePort();
      stream.lastSubMs = now;
      stream.udp.beginPacket(stream.client, stream.clientPort);
      stream.udp.print("#HDR,");
      stream.udp.print(STREAM_HEADER);
      stream.udp.endPacket();
      if (isNew) logMsg("Laptop logger connected: " + stream.client.toString());
    } else if (strncmp(msg, "UNSUB", 5) == 0) {
      stream.clientPort = 0;
      logMsg("Laptop logger disconnected");
    }
  }
}

void streamFlush(uint32_t now) {
  if (stream.len == 0) return;
  streamBeginPacket(now);
  stream.udp.write((const uint8_t *)stream.buf, stream.len);
  stream.udp.endPacket();
  stream.len = 0;
  stream.rows = 0;
}

// The logger can also read the stream over the USB cable (saker_logger.py --serial).
void streamSerialStart() {
  stream.serial = true;
  Serial.print("#HDR,");
  Serial.println(STREAM_HEADER);
}

void streamSample(uint32_t now, const char *mode, const char *phase, const char *event,
                  bool assist, const Imu &thigh, const Imu &shank, bool sagittalKnown) {
  stream.seq++;  // counts every sample, so the logger can spot gaps
  const bool wifi = streamActive(now) || streamBroadcasting(now);
  if (!wifi && !stream.serial) {
    stream.len = stream.rows = 0;
    return;
  }
  char sag[24] = ",";
  if (sagittalKnown) snprintf(sag, sizeof(sag), "%.2f,%.2f", thigh.sagittalRate(), shank.sagittalRate());

  char row[260];
  const int n = snprintf(
      row, sizeof(row),
      "%lu,%lu,%s,%s,%s,%d,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%s\n",
      (unsigned long)now, (unsigned long)stream.seq, mode, phase, event, assist ? 1 : 0,
      thigh.accel[0], thigh.accel[1], thigh.accel[2], thigh.gyro[0], thigh.gyro[1], thigh.gyro[2],
      shank.accel[0], shank.accel[1], shank.accel[2], shank.gyro[0], shank.gyro[1], shank.gyro[2],
      sag);
  if (n <= 0 || n >= (int)sizeof(row)) return;
  if (stream.serial) Serial.write((const uint8_t *)row, n);
  if (!wifi) return;
  if (streamBroadcasting(now) && now - stream.lastBroadcastHeaderMs >= 1000) {
    // Listeners can join at any time, so repeat the header once a second.
    stream.lastBroadcastHeaderMs = now;
    streamBeginPacket(now);
    stream.udp.print("#HDR,");
    stream.udp.print(STREAM_HEADER);
    stream.udp.endPacket();
  }
  if (stream.len + n > sizeof(stream.buf)) streamFlush(now);
  memcpy(stream.buf + stream.len, row, n);
  stream.len += n;
  if (++stream.rows >= STREAM_SAMPLES_PER_PACKET) streamFlush(now);
}
