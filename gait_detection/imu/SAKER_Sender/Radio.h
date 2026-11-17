#pragma once
// ESP-NOW link to the receiver. Until a receiver answers, the sender broadcasts
// HELLO; after pairing it sends heartbeats, and commands are resent until the
// receiver acknowledges them.
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <string.h>
#include "Protocol.h"
#include "Log.h"

constexpr uint32_t HELLO_INTERVAL_MS = 100;      // while searching
constexpr uint32_t HEARTBEAT_INTERVAL_MS = 300;
constexpr uint32_t PEER_TIMEOUT_MS = 1500;       // no reply for this long = receiver lost
constexpr uint32_t COMMAND_RETRY_MS = 150;
constexpr uint8_t COMMAND_MAX_TRIES = 6;

const uint8_t BROADCAST_MAC[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

struct RxItem {
  Packet packet;
  uint8_t mac[6];
};

struct Radio {
  bool ready = false;   // ESP-NOW started
  bool paired = false;
  uint8_t peerMac[6] = {};
  uint32_t nextSeq = 1;
  uint32_t lastPeerSeenMs = 0, lastHelloMs = 0, lastHeartbeatMs = 0, lastMismatchLogMs = 0;
  // Command waiting for its ACK.
  bool commandPending = false;
  Packet command = {};
  uint32_t commandSentMs = 0;
  uint8_t commandTries = 0;
  QueueHandle_t rxQueue = nullptr;
  volatile bool versionMismatch = false;
};
Radio radio;

// Runs in the Wi-Fi task: only validate and queue. Never touch Serial, web or I2C here.
void radioOnReceive(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  if (!info || !info->src_addr || len != (int)sizeof(Packet)) return;
  RxItem item;
  memcpy(&item.packet, data, sizeof(Packet));
  if (item.packet.magic != PROTOCOL_MAGIC) return;
  if (item.packet.version != PROTOCOL_VERSION) {
    radio.versionMismatch = true;
    return;
  }
  memcpy(item.mac, info->src_addr, 6);
  xQueueSend(radio.rxQueue, &item, 0);  // if the queue is full, drop rather than block Wi-Fi
}

void radioAddPeer(const uint8_t *mac) {
  if (esp_now_is_peer_exist(mac)) return;
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, mac, 6);
  peer.channel = 0;  // follow whatever channel the hotspot uses
  peer.ifidx = WIFI_IF_STA;
  peer.encrypt = false;
  if (esp_now_add_peer(&peer) != ESP_OK) logMsg("ESP-NOW add peer failed");
}

bool radioSend(const Packet &p, const uint8_t *mac) {
  return radio.ready && esp_now_send(mac, (const uint8_t *)&p, sizeof(p)) == ESP_OK;
}

// Call after WiFi.mode(WIFI_STA).
bool radioBegin() {
  radio.rxQueue = xQueueCreate(8, sizeof(RxItem));
  if (!radio.rxQueue || esp_now_init() != ESP_OK) return false;
  esp_now_register_recv_cb(radioOnReceive);
  radioAddPeer(BROADCAST_MAC);
  radio.ready = true;
  return true;
}

// Sends now and keeps resending until ACKed. A new command replaces an unanswered one.
bool radioSendCommand(uint16_t code, uint16_t angle = 0, uint16_t durationMs = 0) {
  if (!radio.paired) return false;
  radio.command = makePacket(COMMAND, code, radio.nextSeq++, angle, durationMs);
  radio.commandPending = true;
  radio.commandTries = 1;
  radio.commandSentMs = millis();
  radioSend(radio.command, radio.peerMac);
  return true;
}

void radioHandlePacket(const RxItem &in, uint32_t now) {
  const Packet &p = in.packet;
  if (p.type == HELLO_ACK) {
    if (!radio.paired || memcmp(in.mac, radio.peerMac, 6) != 0) {
      memcpy(radio.peerMac, in.mac, 6);
      radioAddPeer(radio.peerMac);
      radio.paired = esp_now_is_peer_exist(radio.peerMac);
      if (radio.paired) logMsg("Receiver paired");
    }
    radio.lastPeerSeenMs = now;
    return;
  }
  if (!radio.paired || memcmp(in.mac, radio.peerMac, 6) != 0 || p.type != ACK) return;
  radio.lastPeerSeenMs = now;
  if (!radio.commandPending || p.seq != radio.command.seq) return;  // heartbeat ACK
  radio.commandPending = false;
  if (p.code == ACK_BUSY) logMsg("Receiver still moving: assist skipped");
  else if (p.code != ACK_OK) logMsg("Receiver rejected the command");
}

void radioUpdate(uint32_t now) {
  if (!radio.ready) return;

  RxItem item;
  while (xQueueReceive(radio.rxQueue, &item, 0) == pdTRUE) radioHandlePacket(item, now);

  if (radio.versionMismatch && now - radio.lastMismatchLogMs > 5000) {
    radio.versionMismatch = false;
    radio.lastMismatchLogMs = now;
    logMsg("Receiver runs older code: upload SAKER_Receiver again");
  }

  if (radio.paired && now - radio.lastPeerSeenMs > PEER_TIMEOUT_MS) {
    radio.paired = false;
    radio.commandPending = false;
    logMsg("Receiver lost: searching");
  }

  if (!radio.paired && now - radio.lastHelloMs >= HELLO_INTERVAL_MS) {
    radio.lastHelloMs = now;
    radioSend(makePacket(HELLO), BROADCAST_MAC);
  }

  if (radio.paired && now - radio.lastHeartbeatMs >= HEARTBEAT_INTERVAL_MS) {
    radio.lastHeartbeatMs = now;
    radioSend(makePacket(HEARTBEAT, 0, radio.nextSeq++), radio.peerMac);
  }

  if (radio.commandPending && radio.paired && now - radio.commandSentMs >= COMMAND_RETRY_MS) {
    if (radio.commandTries >= COMMAND_MAX_TRIES) {
      radio.commandPending = false;
      logMsg("Receiver did not acknowledge the command");
    } else {
      radioSend(radio.command, radio.peerMac);
      radio.commandTries++;
      radio.commandSentMs = now;
    }
  }
}
