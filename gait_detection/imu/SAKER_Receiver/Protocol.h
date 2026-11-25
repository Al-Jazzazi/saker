#pragma once
// ESP-NOW packets shared by SAKER_Sender and SAKER_Receiver.
// This file must be identical in both sketch folders. Bump PROTOCOL_VERSION
// whenever the packet layout or meaning changes, then re-upload BOTH boards.
#include <stdint.h>

constexpr uint32_t PROTOCOL_MAGIC = 0x53414B52;  // "SAKR"
constexpr uint8_t PROTOCOL_VERSION = 2;

enum PacketType : uint8_t { HELLO = 1, HELLO_ACK = 2, COMMAND = 3, ACK = 4, HEARTBEAT = 5 };
enum CommandCode : uint16_t { CMD_ASSIST = 1, CMD_STOP = 2 };
enum AckCode : uint16_t { ACK_OK = 0, ACK_REJECTED = 1, ACK_BUSY = 2 };

struct __attribute__((packed)) Packet {
  uint32_t magic;
  uint8_t version;
  uint8_t type;         // PacketType
  uint16_t code;        // COMMAND: CommandCode, ACK: AckCode
  uint32_t seq;         // ACK echoes the seq it answers
  uint16_t angle;       // ASSIST: target angle in degrees. ACK: receiver's current angle
  uint16_t durationMs;  // ASSIST: time to move out; the return takes the same time
};
static_assert(sizeof(Packet) == 16, "Packet layout changed");

inline Packet makePacket(uint8_t type, uint16_t code = 0, uint32_t seq = 0,
                         uint16_t angle = 0, uint16_t durationMs = 0) {
  return Packet{PROTOCOL_MAGIC, PROTOCOL_VERSION, type, code, seq, angle, durationMs};
}
