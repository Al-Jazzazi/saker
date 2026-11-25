#pragma once
#include <Arduino.h>

// Latest status line, shown on the dashboard and printed to Serial.
String lastMessage = "Starting";

void logMsg(const String &msg) {
  lastMessage = msg;
  Serial.println("[SAKER] " + msg);
}
