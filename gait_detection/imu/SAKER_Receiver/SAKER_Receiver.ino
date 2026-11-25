// SAKER receiver: moves the assist servo when the sender asks, along a smooth
// (minimum-jerk) path, returns it on its own timer, and returns it to rest if
// the link drops.
// ESP32 Dev Module / Arduino-ESP32 core 3.3.x. BENCH SERVO ONLY.
//
// Tabs: Protocol.h (packet format, identical copy in SAKER_Sender)
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <string.h>
#include "Protocol.h"

// ---- Servo ----
constexpr uint8_t SERVO_PIN = 18;
constexpr uint32_t PWM_HZ = 50;                   // 20 ms servo frame
constexpr uint8_t PWM_BITS = 16;
constexpr float SERVO_MIN_US = 1000;              // pulse at 0 degrees: check your servo's datasheet
constexpr float SERVO_MAX_US = 2000;              // pulse at 180 degrees
constexpr float REST_ANGLE = 90;
constexpr float MIN_ANGLE = 45, MAX_ANGLE = 135;  // every command is clamped to this range
constexpr uint32_t SERVO_UPDATE_MS = 20;          // one new position per PWM frame
constexpr uint32_t ASSIST_HOLD_MS = 0;            // pause at the target before returning
constexpr uint16_t MIN_MOVE_MS = 300, MAX_MOVE_MS = 3000;
constexpr uint32_t STOP_RETURN_MS = 800;          // smooth return used by STOP and link loss

// ---- Link ----
constexpr uint32_t LINK_TIMEOUT_MS = 1250;        // no packet for this long: return to rest
constexpr uint32_t SCAN_DWELL_MS = 250;           // sender says HELLO every 100 ms while searching
constexpr uint8_t MAX_CHANNEL = 11;

// Received packet waiting to be handled in loop(). Declared up here so the
// Arduino builder's auto-generated function prototypes can see it.
struct RxItem {
  Packet packet;
  uint8_t mac[6];
};

// ---------------------------------------------------------------------------
// Motion: MOVING_OUT -> HOLDING -> RETURNING -> AT_REST

enum MotionState : uint8_t { AT_REST, MOVING_OUT, HOLDING, RETURNING };

struct Motion {
  MotionState state = AT_REST;
  float angle = REST_ANGLE;  // last angle written to the servo
  float fromAngle = REST_ANGLE, toAngle = REST_ANGLE;
  uint32_t startMs = 0, durationMs = 1;
  uint32_t holdUntilMs = 0;
  uint32_t returnMs = STOP_RETURN_MS;  // duration of the return leg
};
Motion motion;

// Maps 0..1 to 0..1 with zero speed and zero acceleration at both ends, so the
// servo eases in and out instead of starting and stopping abruptly.
float minimumJerk(float t) {
  t = constrain(t, 0.0f, 1.0f);
  return t * t * t * (10 + t * (6 * t - 15));
}

void writeAngle(float angle) {
  angle = constrain(angle, MIN_ANGLE, MAX_ANGLE);
  const float periodUs = 1e6f / PWM_HZ;
  const float pulseUs = SERVO_MIN_US + (SERVO_MAX_US - SERVO_MIN_US) * angle / 180.0f;
  ledcWrite(SERVO_PIN, (uint32_t)lroundf(pulseUs / periodUs * ((1u << PWM_BITS) - 1)));
  motion.angle = angle;
}

void beginMove(MotionState state, float target, uint32_t durationMs) {
  motion.state = state;
  motion.fromAngle = motion.angle;
  motion.toAngle = constrain(target, MIN_ANGLE, MAX_ANGLE);
  motion.startMs = millis();
  motion.durationMs = max(durationMs, (uint32_t)1);
}

void returnToRest() {
  if (motion.state == AT_REST || motion.state == RETURNING) return;
  beginMove(RETURNING, REST_ANGLE, STOP_RETURN_MS);
}

void updateMotion(uint32_t now) {
  static uint32_t lastUpdateMs = 0;
  if (now - lastUpdateMs < SERVO_UPDATE_MS) return;
  lastUpdateMs = now;

  switch (motion.state) {
    case AT_REST:
      return;
    case HOLDING:
      if ((int32_t)(now - motion.holdUntilMs) >= 0) beginMove(RETURNING, REST_ANGLE, motion.returnMs);
      return;
    case MOVING_OUT:
    case RETURNING: {
      const float t = float(now - motion.startMs) / motion.durationMs;
      writeAngle(motion.fromAngle + (motion.toAngle - motion.fromAngle) * minimumJerk(t));
      if (t < 1) return;
      if (motion.state == MOVING_OUT) {
        motion.state = HOLDING;
        motion.holdUntilMs = now + ASSIST_HOLD_MS;
      } else {
        motion.state = AT_REST;
        Serial.println("Back at rest");
      }
      return;
    }
  }
}

// ---------------------------------------------------------------------------
// ESP-NOW

QueueHandle_t rxQueue = nullptr;
volatile bool versionMismatch = false;

bool paired = false;
uint8_t senderMac[6] = {};
uint8_t channel = 1;
uint32_t lastScanMs = 0, lastReceivedMs = 0, lastMismatchLogMs = 0;
bool seenCommand = false;
uint32_t lastCommandSeq = 0;
uint16_t lastCommandResult = ACK_OK;

// Runs in the Wi-Fi task: only validate and queue.
void onReceive(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  if (!info || !info->src_addr || len != (int)sizeof(Packet)) return;
  RxItem item;
  memcpy(&item.packet, data, sizeof(Packet));
  if (item.packet.magic != PROTOCOL_MAGIC) return;
  if (item.packet.version != PROTOCOL_VERSION) {
    versionMismatch = true;
    return;
  }
  memcpy(item.mac, info->src_addr, 6);
  xQueueSend(rxQueue, &item, 0);
}

void addPeer(const uint8_t *mac) {
  if (esp_now_is_peer_exist(mac)) return;
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, mac, 6);
  peer.channel = 0;
  peer.ifidx = WIFI_IF_STA;
  peer.encrypt = false;
  if (esp_now_add_peer(&peer) != ESP_OK) Serial.println("Add ESP-NOW peer failed");
}

void reply(uint8_t type, uint32_t seq = 0, uint16_t code = ACK_OK) {
  if (!paired) return;
  const Packet p = makePacket(type, code, seq, (uint16_t)lroundf(motion.angle));
  esp_now_send(senderMac, (const uint8_t *)&p, sizeof(p));
}

uint16_t runCommand(const Packet &p) {
  if (p.code == CMD_STOP) {
    returnToRest();
    Serial.println("STOP: easing back to rest");
    return ACK_OK;
  }
  if (p.code != CMD_ASSIST) return ACK_REJECTED;
  if (p.angle < MIN_ANGLE || p.angle > MAX_ANGLE || p.durationMs < MIN_MOVE_MS ||
      p.durationMs > MAX_MOVE_MS) {
    return ACK_REJECTED;
  }
  if (motion.state != AT_REST) return ACK_BUSY;
  motion.returnMs = p.durationMs;
  beginMove(MOVING_OUT, p.angle, p.durationMs);
  Serial.printf("ASSIST: to %u deg over %u ms, then back\n", p.angle, p.durationMs);
  return ACK_OK;
}

void handlePacket(const RxItem &in, uint32_t now) {
  const Packet &p = in.packet;
  if (p.type == HELLO) {
    if (!paired || memcmp(senderMac, in.mac, 6) != 0) {
      memcpy(senderMac, in.mac, 6);
      addPeer(senderMac);
      paired = esp_now_is_peer_exist(senderMac);
      if (paired) {
        Serial.printf("Sender paired on channel %u\n", channel);
        seenCommand = false;
      }
    }
    lastReceivedMs = now;
    reply(HELLO_ACK);
    return;
  }
  if (!paired || memcmp(senderMac, in.mac, 6) != 0) return;
  lastReceivedMs = now;

  if (p.type == HEARTBEAT) {
    reply(ACK, p.seq);
  } else if (p.type == COMMAND) {
    // The sender resends until it hears an ACK; answer repeats without re-running them.
    if (!seenCommand || p.seq != lastCommandSeq) {
      lastCommandResult = runCommand(p);
      lastCommandSeq = p.seq;
      seenCommand = true;
    }
    reply(ACK, p.seq, lastCommandResult);
  }
}

// While unpaired, hop through channels until the sender's HELLO is heard.
void scanIfNeeded(uint32_t now) {
  if (paired && now - lastReceivedMs > LINK_TIMEOUT_MS) {
    paired = false;
    seenCommand = false;
    returnToRest();
    Serial.println("Link lost: easing back to rest, rescanning channels");
  }
  if (paired || now - lastScanMs < SCAN_DWELL_MS) return;
  lastScanMs = now;
  // The sender sits on its hotspot's channel; ESP-NOW only works on the same one.
  channel = (channel >= MAX_CHANNEL) ? 1 : channel + 1;
  const esp_err_t err = esp_wifi_set_channel(channel, WIFI_SECOND_CHAN_NONE);
  if (err != ESP_OK) Serial.printf("Channel %u set error %d\n", channel, (int)err);
}

// ---------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(200);
  if (!ledcAttach(SERVO_PIN, PWM_HZ, PWM_BITS)) {
    Serial.println("PWM attach failed");
    while (true) delay(1000);
  }
  writeAngle(REST_ANGLE);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect(false, false);
  WiFi.setSleep(false);
  if (esp_wifi_set_channel(channel, WIFI_SECOND_CHAN_NONE) != ESP_OK) {
    Serial.println("Initial channel set failed");
  }
  rxQueue = xQueueCreate(8, sizeof(RxItem));
  if (!rxQueue || esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW initialization failed");
    while (true) delay(1000);
  }
  esp_now_register_recv_cb(onReceive);
  Serial.println("SAKER receiver ready, servo on GPIO18");
  Serial.println("Receiver MAC: " + WiFi.macAddress());
  Serial.println("Scanning channels 1-11 for the sender");
}

void loop() {
  const uint32_t now = millis();
  RxItem item;
  while (xQueueReceive(rxQueue, &item, 0) == pdTRUE) handlePacket(item, now);
  if (versionMismatch && now - lastMismatchLogMs > 5000) {
    versionMismatch = false;
    lastMismatchLogMs = now;
    Serial.println("Sender runs different code: upload SAKER_Sender again");
  }
  scanIfNeeded(now);
  updateMotion(now);
  delay(1);
}
