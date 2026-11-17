// SAKER sender: thigh + shank MPU6050s on the healthy leg, gait-phase state
// machine, ESP-NOW commands to the receiver's servo, a web dashboard, and a
// UDP data stream to the laptop.
// ESP32 Dev Module / Arduino-ESP32 core 3.3.x. Bench servo only.
//
// Tabs:  Config.h     settings to tune
//        GaitFsm.h    gait phases + WALK calibration (tested in ../tests)
//        Imu.h        MPU6050 driver
//        Radio.h      ESP-NOW pairing and commands
//        DataStream.h UDP stream to laptop/saker_logger.py
//        WebUi.h      dashboard
//        Protocol.h   packet format (identical copy in SAKER_Receiver)
//
// Flow: CAL (still 3 s) -> WALK (12 s of steps) -> TRACK (phases only) or
// START (phases + motor). STOP at any time.
#include <Arduino.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <esp_wifi.h>
#include "Config.h"
#include "Log.h"
#include "GaitFsm.h"
#include "Imu.h"
#include "Radio.h"
#include "DataStream.h"
#include "WebUi.h"

enum Mode : uint8_t { STANDBY, STILL_CAL, WALK_CAL, TRACKING, RUNNING };
Mode mode = STANDBY;

const char *modeName() {
  switch (mode) {
    case STANDBY: return "STANDBY";
    case STILL_CAL: return "CAL";
    case WALK_CAL: return "WALK";
    case TRACKING: return "TRACK";
    case RUNNING: return "RUN";
  }
  return "?";
}

Imu thigh{THIGH_ADDR};
Imu shank{SHANK_ADDR};
GaitFsm fsm;

bool imuOk = false;       // both IMUs answering
uint8_t imuFailStreak = 0;
uint32_t lastImuRetryMs = 0, lastSampleMs = 0, lastWiFiTryMs = 0;
bool gyroReady = false;   // stationary CAL done
bool walkReady = false;   // WALK done: sagittal axes and thresholds known
uint32_t heelStrikes = 0, assistsSent = 0, assistBusyUntilMs = 0;
WalkCalResult walkResult;

bool gaitActive() { return mode == TRACKING || mode == RUNNING; }

// ---------------------------------------------------------------------------
// Motor commands

void stopAll() {
  mode = STANDBY;
  if (radio.paired) radioSendCommand(CMD_STOP);
  assistBusyUntilMs = millis() + 1000;  // receiver needs ~0.8 s to ease back to rest
}

bool sendAssist(uint32_t now) {
  if ((int32_t)(now - assistBusyUntilMs) < 0) return false;  // previous assist still moving
  if (!radioSendCommand(CMD_ASSIST, REST_ANGLE + ASSIST_DELTA_DEG, ASSIST_MOVE_MS)) return false;
  assistBusyUntilMs = now + 2u * ASSIST_MOVE_MS + 200;  // out + back + margin
  assistsSent++;
  return true;
}

// ---------------------------------------------------------------------------
// Stationary CAL: gyro bias, only accepted if the sensors were truly still.

float calSum[6], calSumSq[6];
uint32_t calCount = 0;

void startStillCal() {
  stopAll();
  gyroReady = walkReady = false;
  for (int i = 0; i < 3; i++) thigh.bias[i] = shank.bias[i] = 0;
  memset(calSum, 0, sizeof(calSum));
  memset(calSumSq, 0, sizeof(calSumSq));
  calCount = 0;
  mode = STILL_CAL;
  logMsg("CAL: keep both sensors still for 3 seconds");
}

void stillCalSample() {
  const float v[6] = {thigh.gyro[0], thigh.gyro[1], thigh.gyro[2],
                      shank.gyro[0], shank.gyro[1], shank.gyro[2]};
  for (int i = 0; i < 6; i++) {
    calSum[i] += v[i];
    calSumSq[i] += v[i] * v[i];
  }
  if (++calCount < STILL_CAL_SAMPLES) return;

  mode = STANDBY;
  float mean[6];
  for (int i = 0; i < 6; i++) {
    mean[i] = calSum[i] / calCount;
    const float sd = sqrtf(fmaxf(0, calSumSq[i] / calCount - mean[i] * mean[i]));
    if (sd > STILL_CAL_MAX_SD) {
      logMsg("CAL failed: sensors moved. Try again");
      return;
    }
  }
  for (int i = 0; i < 3; i++) {
    thigh.bias[i] = mean[i];
    shank.bias[i] = mean[i + 3];
  }
  gyroReady = true;
  logMsg("CAL done. Next: WALK");
}

// Slow bias correction while the leg rests in STANDBY; never while walking.
uint32_t stillSinceMs = 0;
void updateBiasIfStill(uint32_t now) {
  if (!gyroReady || mode != STANDBY) return;
  const bool quiet = thigh.gyroMag < 2.0f && shank.gyroMag < 2.0f &&
                     fabsf(thigh.accelMag - 1) < 0.04f && fabsf(shank.accelMag - 1) < 0.04f;
  if (!quiet) {
    stillSinceMs = 0;
    return;
  }
  if (!stillSinceMs) stillSinceMs = now;
  if (now - stillSinceMs < 3000) return;
  constexpr float alpha = 0.0005f;
  for (int i = 0; i < 3; i++) {
    thigh.bias[i] += alpha * thigh.gyro[i];
    shank.bias[i] += alpha * shank.gyro[i];
  }
}

// ---------------------------------------------------------------------------
// WALK: record 12 s of walking, then let analyseWalk() pick axes and thresholds.

constexpr size_t WALK_BUF = WALK_CAL_MS / SAMPLE_MS + 100;
float walkShank[WALK_BUF][3], walkThigh[WALK_BUF][3], walkScratch[WALK_BUF];
size_t walkCount = 0;
uint32_t walkStartMs = 0;

void startWalkCal() {
  if (!gyroReady) {
    logMsg("Run CAL first");
    return;
  }
  stopAll();
  walkReady = false;
  walkCount = 0;
  walkStartMs = millis();
  mode = WALK_CAL;
  logMsg("WALK: take at least 4 normal steps in the next 12 seconds");
}

void walkCalSample(uint32_t now) {
  if (walkCount < WALK_BUF) {
    memcpy(walkShank[walkCount], shank.gyro, sizeof(shank.gyro));
    memcpy(walkThigh[walkCount], thigh.gyro, sizeof(thigh.gyro));
    walkCount++;
  }
  if (now - walkStartMs < WALK_CAL_MS) return;

  mode = STANDBY;
  walkResult = analyseWalk(walkShank, walkThigh, walkCount, SAMPLE_MS, walkScratch);
  Serial.printf("WALK: %u samples, %d steps, shank axis %c%c, thigh axis %c%c, "
                "swing peak %.0f, push-off peak %.0f\n",
                (unsigned)walkCount, walkResult.steps, "XYZ"[walkResult.shank.axis],
                walkResult.shank.sign > 0 ? '+' : '-', "XYZ"[walkResult.thigh.axis],
                walkResult.thigh.sign > 0 ? '+' : '-', walkResult.swingPeak, walkResult.pushOffPeak);
  if (!walkResult.ok) {
    logMsg(String("WALK failed (") + walkResult.message + "). Repeat with clearer steps");
    return;
  }
  shank.sagAxis = walkResult.shank.axis;
  shank.sagSign = walkResult.shank.sign;
  thigh.sagAxis = walkResult.thigh.axis;
  thigh.sagSign = walkResult.thigh.sign;
  fsm.thresholds = walkResult.thresholds;
  walkReady = true;
  logMsg("WALK done: " + String(walkResult.steps) + " steps. Ready for TRACK or START");
}

// ---------------------------------------------------------------------------
// Gait tracking

void startGait(bool withMotor) {
  if (!gyroReady || !walkReady) {
    logMsg("Run CAL and WALK first");
    return;
  }
  if (withMotor && !radio.paired) {
    logMsg("Receiver not paired");
    return;
  }
  fsm.reset(millis());
  heelStrikes = 0;
  mode = withMotor ? RUNNING : TRACKING;
  logMsg(withMotor ? "RUNNING: gait phases + motor" : "TRACKING: gait phases, motor off");
}

// Returns true if an assist was sent on this sample.
bool gaitSample(uint32_t now, GaitEvent &event) {
  event = fsm.update(shank.sagittalRate(), thigh.gyroMag, now);
  if (event == EVENT_NONE) return false;
  if (event == EVENT_HEEL_STRIKE) heelStrikes++;
  if (mode != RUNNING || event != ASSIST_TRIGGER_EVENT) return false;
  if (!sendAssist(now)) return false;
  logMsg("Assist #" + String(assistsSent) + " at " + gaitEventName(event));
  return true;
}

// ---------------------------------------------------------------------------
// 100 Hz sample

void onImuLost(uint32_t now) {
  imuOk = false;
  lastImuRetryMs = now;
  if (mode != STANDBY) stopAll();
  logMsg("IMU lost: check wiring. Retrying every second");
}

void onSample(uint32_t now) {
  if (!imuOk) {
    if (now - lastImuRetryMs < IMU_RETRY_MS) return;
    lastImuRetryMs = now;
    if (!imuStartBus(thigh, shank)) return;
    imuOk = true;
    imuFailStreak = 0;
    logMsg("IMUs connected");
    return;
  }
  if (!imuRead(thigh) || !imuRead(shank)) {
    if (++imuFailStreak >= IMU_FAILS_BEFORE_LOST) onImuLost(now);
    return;  // skip one bad sample, keep going
  }
  imuFailStreak = 0;

  GaitEvent event = EVENT_NONE;
  bool assist = false;
  switch (mode) {
    case STILL_CAL: stillCalSample(); break;
    case WALK_CAL: walkCalSample(now); break;
    case TRACKING:
    case RUNNING: assist = gaitSample(now, event); break;
    case STANDBY: updateBiasIfStill(now); break;
  }
  const bool phaseKnown = gaitActive();
  streamSample(now, modeName(), phaseKnown ? gaitPhaseName(fsm.phase()) : "", gaitEventName(event),
               assist, thigh, shank, walkReady);
}

// ---------------------------------------------------------------------------
// Commands (dashboard, Serial) and status

String runCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  if (cmd == "SUB") {  // USB logger asking for the data stream
    streamSerialStart();
    return lastMessage;
  }
  if (cmd == "UNSUB") {
    stream.serial = false;
    return lastMessage;
  }
  if (cmd == "STOP") {
    stopAll();
    logMsg("Stopped");
  } else if (mode == STILL_CAL || mode == WALK_CAL) {
    logMsg("Calibration running. STOP to cancel");
  } else if (cmd == "TEST") {
    if (mode != STANDBY || !radio.paired) logMsg("TEST needs STANDBY and a paired receiver");
    else if (sendAssist(millis())) logMsg("Servo TEST: 30 degrees out and back");
    else logMsg("Servo still moving: try again in a moment");
  } else if (cmd != "CAL" && cmd != "WALK" && cmd != "TRACK" && cmd != "START") {
    logMsg("Unknown command: " + cmd);
  } else if (!imuOk) {
    logMsg("IMUs not connected");
  } else if (cmd == "CAL") {
    startStillCal();
  } else if (cmd == "WALK") {
    startWalkCal();
  } else {
    startGait(cmd == "START");
  }
  return lastMessage;
}

String buildStatusJson() {
  const uint32_t now = millis();
  String axes = "";
  if (walkReady) {
    axes = String("shank ") + "XYZ"[shank.sagAxis] + (shank.sagSign > 0 ? "+" : "-") +
           ", thigh " + "XYZ"[thigh.sagAxis] + (thigh.sagSign > 0 ? "+" : "-");
  }
  String j;
  j.reserve(480);
  j += "{\"mode\":\"" + String(modeName()) + "\"";
  j += ",\"phase\":\"" + String(gaitActive() ? gaitPhaseName(fsm.phase()) : "--") + "\"";
  j += ",\"imu\":" + String(imuOk ? "true" : "false");
  j += ",\"peer\":" + String(radio.paired ? "true" : "false");
  j += ",\"logger\":\"" + streamClientText(now) + "\"";
  j += ",\"channel\":" + String(WiFi.channel());
  j += ",\"gyro\":" + String(gyroReady ? "true" : "false");
  j += ",\"walk\":" + String(walkReady ? "true" : "false");
  j += ",\"axes\":\"" + axes + "\"";
  j += ",\"steps\":" + String(heelStrikes) + ",\"assists\":" + String(assistsSent);
  j += ",\"sr\":" + String(shank.sagittalRate(), 1) + ",\"tr\":" + String(thigh.sagittalRate(), 1);
  j += ",\"thr\":[" + String(fsm.thresholds.swingEnter, 0) + "," +
       String(fsm.thresholds.pushOffEnter, 0) + "," + String(fsm.thresholds.heelStrike, 0) + "]";
  j += ",\"event\":\"" + jsonEscape(lastMessage) + "\"}";
  return j;
}

String serialLine;
void readSerialCommands() {
  while (Serial.available()) {
    const char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialLine.length()) runCommand(serialLine);
      serialLine = "";
    } else if (serialLine.length() < 20) {
      serialLine += c;
    }
  }
}

// ---------------------------------------------------------------------------
// Wi-Fi

bool mdnsStarted = false;
uint8_t ssidIndex = 0;

String scannedSsid;  // exact hotspot name found by the last scan, if any

// Letters and digits only, lowercase: "Yousef’s iPhone" and "Yousef's iPhone" match.
String simplifySsid(const String &ssid) {
  String out;
  for (unsigned int i = 0; i < ssid.length(); i++) {
    const unsigned char c = ssid[i];
    if (c < 128 && isalnum(c)) out += (char)tolower(c);
  }
  return out;
}

// Lists nearby networks on Serial and remembers the one that matches the hotspot.
void scanForHotspot() {
  const int n = WiFi.scanNetworks();
  const String wanted = simplifySsid(WIFI_SSIDS[0]);
  Serial.printf("Wi-Fi scan: %d networks (looking for \"%s\")\n", n, WIFI_SSIDS[0]);
  for (int i = 0; i < n; i++) {
    const String ssid = WiFi.SSID(i);
    Serial.printf("  \"%s\"  %d dBm  channel %d\n", ssid.c_str(), (int)WiFi.RSSI(i), (int)WiFi.channel(i));
    if (simplifySsid(ssid) == wanted) scannedSsid = ssid;
  }
  WiFi.scanDelete();
}

void beginWiFi() {
  const char *ssid = scannedSsid.length() ? scannedSsid.c_str() : WIFI_SSIDS[ssidIndex];
  Serial.printf("Connecting to hotspot \"%s\"...\n", ssid);
  WiFi.begin(ssid, WIFI_PASSWORD);
}

void maintainWiFi(uint32_t now) {
  static bool wasConnected = false;
  const bool connected = WiFi.status() == WL_CONNECTED;
  if (connected && !wasConnected) {
    // Power save makes the ESP sleep through packets from the laptop (ping and the
    // dashboard fail while its own messages still go out), so force it off once connected.
    esp_wifi_set_ps(WIFI_PS_NONE);
    wifi_ps_type_t ps;
    esp_wifi_get_ps(&ps);
    Serial.printf("Wi-Fi power save: %s\n", ps == WIFI_PS_NONE ? "off" : "STILL ON");
    Serial.printf("Connected to \"%s\"\n", WiFi.SSID().c_str());
    Serial.printf("Dashboard: http://%s  or  http://%s.local\n",
                  WiFi.localIP().toString().c_str(), MDNS_NAME);
    Serial.printf("Hotspot Wi-Fi channel: %d\n", (int)WiFi.channel());
    if (!mdnsStarted && MDNS.begin(MDNS_NAME)) {
      MDNS.addService("http", "tcp", 80);
      mdnsStarted = true;
    }
    if (!stream.started) streamBegin();
  }
  wasConnected = connected;

  // Reconnecting can change the channel; the receiver then finds it again by itself.
  // Each retry tries the next hotspot spelling.
  if (connected || mode == RUNNING || now - lastWiFiTryMs < 12000) return;
  lastWiFiTryMs = now;
  WiFi.disconnect(false, false);
  ssidIndex = (ssidIndex + 1) % WIFI_SSID_COUNT;
  scanForHotspot();
  beginWiFi();
}

// ---------------------------------------------------------------------------

void setup() {
  Serial.setTxBufferSize(8192);  // the USB data stream must never block the 100 Hz loop
  Serial.begin(921600);
  delay(200);

  // The MPU6050s can miss the very first transaction after power-up; give them a few tries.
  for (int attempt = 0; attempt < 3 && !imuOk; attempt++) {
    if (attempt) delay(100);
    imuOk = imuStartBus(thigh, shank);
  }
  if (!imuOk) Serial.println("MPU6050 not found at 0x68 / 0x69: check wiring. Retrying in the background.");

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  beginWiFi();
  lastWiFiTryMs = millis();

  if (!radioBegin()) logMsg("ESP-NOW failed to start: reset the board");
  setupWeb();
  Serial.println("Sender MAC: " + WiFi.macAddress());
  logMsg(imuOk ? "Ready. CAL, WALK, TRACK, START, STOP, TEST" : "IMU not found: check wiring");
}

void loop() {
  const uint32_t now = millis();
  server.handleClient();
  readSerialCommands();
  maintainWiFi(now);
  streamPoll(now);
  radioUpdate(now);

  if (mode == RUNNING && !radio.paired) {
    stopAll();
    logMsg("Receiver lost: stopped");
  }
  if (mode == RUNNING && WiFi.status() != WL_CONNECTED) {
    stopAll();
    logMsg("Hotspot lost: stopped, dashboard unreachable");
  }

  if (now - lastSampleMs >= SAMPLE_MS) {
    // Keep a steady 100 Hz; if the loop stalled, resync instead of bursting.
    lastSampleMs = (now - lastSampleMs > 5 * SAMPLE_MS) ? now : lastSampleMs + SAMPLE_MS;
    onSample(now);
  }
  delay(1);
}
