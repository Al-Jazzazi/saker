#pragma once
// Dashboard served by the sender: status, calibration, controls, live values.
#include <Arduino.h>
#include <WebServer.h>

// Implemented in SAKER_Sender.ino.
String buildStatusJson();
String runCommand(String cmd);

WebServer server(80);

const char PAGE[] PROGMEM = R"HTML(<!doctype html>
<html><head>
<meta name="viewport" content="width=device-width,initial-scale=1"><meta charset="utf-8">
<title>SAKER</title>
<style>
body{font:16px -apple-system,Arial,sans-serif;max-width:780px;margin:auto;padding:18px;background:#111a29;color:#edf3fa}
h1{margin-bottom:4px}.muted,small{color:#aab9ca}
.card{background:#1b293d;border:1px solid #354962;border-radius:14px;padding:17px;margin:12px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.value{font-size:24px;font-weight:700}
.phase{font-size:34px;font-weight:800;color:#7cc4ff}
button{background:#276cbd;color:#fff;border:0;border-radius:9px;padding:15px;margin:4px;font-weight:700;min-width:125px}
button.stop{background:#c44848}button.alt{background:#3d566f}#event{overflow-wrap:anywhere}
</style></head><body>
<h1>SAKER Control</h1><p class="muted">Healthy-leg IMUs / gait phases / ESP-NOW bench servo</p>

<div class="card"><h3>Gait phase</h3><div class="phase" id="phase">--</div>
<div class="grid"><div>Heel strikes<div class="value" id="steps">--</div></div>
<div>Assists sent<div class="value" id="assists">--</div></div></div></div>

<div class="card"><h3>System</h3><div class="grid">
<div>Mode<div class="value" id="mode">--</div></div>
<div>IMUs<div class="value" id="imu">--</div></div>
<div>Receiver<div class="value" id="peer">--</div></div>
<div>Laptop logger<div class="value" id="logger">--</div></div>
</div></div>

<div class="card"><h3>Calibration</h3><p id="cal">--</p>
<button onclick="cmd('CAL')">Stationary CAL</button><button onclick="cmd('WALK')">Walking CAL</button>
<p><small>CAL: keep both sensors still for 3 s. WALK: take at least four normal steps during the 12 s.</small></p></div>

<div class="card"><h3>Run</h3>
<button class="alt" onclick="cmd('TRACK')">TRACK (motor off)</button><button onclick="cmd('START')">START (motor on)</button>
<button class="stop" onclick="cmd('STOP')">STOP</button><button class="alt" onclick="cmd('TEST')">TEST SERVO</button>
<p><small>TRACK runs the phase detection and streams data without moving the motor.</small></p></div>

<div class="card"><h3>Live</h3><div class="grid">
<div>Shank sagittal (deg/s)<div class="value" id="sr">--</div></div>
<div>Thigh sagittal (deg/s)<div class="value" id="tr">--</div></div>
<div>Swing / push-off / heel thresholds<div class="value" id="thr">--</div></div>
<div>Wi-Fi channel<div class="value" id="channel">--</div></div>
</div></div>

<div class="card"><h3>Latest event</h3><p id="event">--</p></div>
<p><small>Experimental movement detector; no ground-force measurement. Bench servo only.</small></p>
<script>
const e = id => document.getElementById(id);
async function cmd(c) {
  try {
    const r = await fetch('/cmd?c=' + encodeURIComponent(c), {method: 'POST'});
    e('event').textContent = (await r.json()).message;
  } catch (x) { e('event').textContent = 'Command failed: ' + x.message; }
}
async function refresh() {
  try {
    const d = await (await fetch('/status', {cache: 'no-store'})).json();
    e('phase').textContent = d.phase;
    e('steps').textContent = d.steps;
    e('assists').textContent = d.assists;
    e('mode').textContent = d.mode;
    e('imu').textContent = d.imu ? 'OK' : 'Not found';
    e('peer').textContent = d.peer ? 'Paired' : 'Searching';
    e('logger').textContent = d.logger || 'None';
    e('channel').textContent = d.channel;
    e('cal').textContent = 'Gyro: ' + (d.gyro ? 'Ready' : 'Not ready') + ' | WALK: ' +
      (d.walk ? 'Ready (' + d.axes + ')' : 'Not ready');
    e('sr').textContent = d.walk ? d.sr.toFixed(0) : '--';
    e('tr').textContent = d.walk ? d.tr.toFixed(0) : '--';
    e('thr').textContent = d.thr.map(v => v.toFixed(0)).join(' / ');
    e('event').textContent = d.event;
  } catch (x) { e('mode').textContent = 'Offline'; }
}
setInterval(refresh, 500); refresh();
</script></body></html>)HTML";

String jsonEscape(String v) {
  v.replace("\\", "\\\\");
  v.replace("\"", "\\\"");
  v.replace("\n", " ");
  return v;
}

void setupWeb() {
  server.on("/", HTTP_GET, [] { server.send_P(200, "text/html; charset=utf-8", PAGE); });
  server.on("/status", HTTP_GET, [] { server.send(200, "application/json", buildStatusJson()); });
  server.on("/cmd", HTTP_POST, [] {
    const String message = runCommand(server.arg("c"));
    server.send(200, "application/json", "{\"message\":\"" + jsonEscape(message) + "\"}");
  });
  server.begin();
}
