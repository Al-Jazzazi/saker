#!/usr/bin/env python3
"""Live view + guided recording session at http://localhost:8000

Listens to the sender's Wi-Fi broadcast (same data as saker_logger.py --listen),
plots the thigh and shank signals live, and on START runs a guided session:
the Mac says each instruction out loud and every sample is saved to
data/<date-time>_session.csv with an extra 'task' column (stand, walk_normal, ...).

    python3 live_view.py        then open http://localhost:8000
"""
import collections
import csv
import datetime
import http.server
import json
import pathlib
import socket
import subprocess
import threading
import time

BROADCAST_PORT = 5006
HTTP_PORT = 8000
DATA_DIR = pathlib.Path(__file__).parent / "data"

# (task column value, seconds, what to say / show)
SESSION = [
    ("stand", 10, "Stand still"),
    ("walk_normal", 120, "Walk at your normal speed"),
    ("stand", 10, "Stop and stand still"),
    ("walk_slow", 90, "Walk slowly"),
    ("walk_fast", 90, "Walk fast"),
    ("turns", 60, "Walk in circles and turn around"),
    ("stop_go", 60, "Walk, and stop every few steps"),
    ("stand", 10, "Stand still. Almost done"),
]
GET_READY_S = 5


def say(text):
    subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.header = None
        self.recent = collections.deque(maxlen=600)  # (thigh gyro xyz, shank gyro xyz), 6 s
        self.last_data = 0.0
        self.count_window = collections.deque()      # arrival times for the rate
        # session
        self.running = False
        self.session_start = 0.0
        self.step = -1
        self.file = None
        self.writer = None
        self.path = None
        self.rows = 0

    def step_at(self, now):
        """(-1 = get ready, index, or len(SESSION) = finished, seconds left in it)."""
        t = now - self.session_start - GET_READY_S
        if t < 0:
            return -1, -t
        for i, (_, secs, _) in enumerate(SESSION):
            if t < secs:
                return i, secs - t
            t -= secs
        return len(SESSION), 0

    def start(self):
        DATA_DIR.mkdir(exist_ok=True)
        self.path = DATA_DIR / f"{datetime.datetime.now():%Y%m%d-%H%M%S}_session.csv"
        self.file = open(self.path, "w", newline="")
        self.writer = None  # created when the header is known
        self.rows = 0
        self.running = True
        self.session_start = time.monotonic()
        self.step = -1
        say(f"Starting in {GET_READY_S} seconds. Stand still.")

    def stop(self, finished=False):
        if self.file:
            self.file.close()
        self.running = False
        self.file = self.writer = None
        say("Session complete. Well done." if finished else "Recording stopped.")


S = State()


def receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", BROADCAST_PORT))
    sock.settimeout(0.1)
    while True:
        try:
            data, _ = sock.recvfrom(4096)
        except socket.timeout:
            data = b""
        now = time.monotonic()
        with S.lock:
            # Advance the guided session and announce each new step.
            if S.running:
                step, _ = S.step_at(now)
                if step != S.step:
                    S.step = step
                    if step == len(SESSION):
                        S.stop(finished=True)
                    elif step >= 0:
                        say(SESSION[step][2])
            for line in data.decode("ascii", "replace").splitlines():
                if line.startswith("#HDR,"):
                    S.header = line[5:].split(",")
                    continue
                if not S.header:
                    continue
                f = line.split(",")
                if len(f) != len(S.header):
                    continue
                S.last_data = now
                S.count_window.append(now)
                try:
                    row = dict(zip(S.header, f))
                    S.recent.append([float(row[k]) for k in
                                     ("thigh_gx", "thigh_gy", "thigh_gz", "shank_gx", "shank_gy", "shank_gz")])
                except ValueError:
                    pass
                if S.running and 0 <= S.step < len(SESSION):
                    if S.writer is None:
                        S.writer = csv.writer(S.file)
                        S.writer.writerow(S.header + ["task"])
                    S.writer.writerow(f + [SESSION[S.step][0]])
                    S.rows += 1
            while S.count_window and now - S.count_window[0] > 2:
                S.count_window.popleft()
            if S.file and S.rows % 100 == 0:
                S.file.flush()


def signed_main_axis(values):
    """The axis that moves most, signed so its biggest swing is positive (looks like the gait wave)."""
    if not values:
        return []
    cols = list(zip(*values))
    var = [sum((v - sum(c) / len(c)) ** 2 for v in c) for c in cols]
    c = cols[var.index(max(var))]
    sign = 1 if max(c) >= -min(c) else -1
    return [round(sign * v, 1) for v in c]


def status():
    now = time.monotonic()
    with S.lock:
        recent = list(S.recent)
        out = {
            "connected": now - S.last_data < 2,
            "rate": round(len(S.count_window) / 2),
            "running": S.running,
            "rows": S.rows,
            "file": S.path.name if S.path else "",
        }
        if S.running:
            step, left = S.step_at(now)
            out["left"] = round(left)
            out["step"] = "Get ready: stand still" if step < 0 else SESSION[step][2]
            nxt = step + 1
            out["next"] = SESSION[nxt][2] if 0 <= nxt < len(SESSION) else ("Walk at your normal speed" if step < 0 else "Done")
            out["progress"] = round(100 * max(0, now - S.session_start - GET_READY_S) / sum(s for _, s, _ in SESSION))
    out["thigh"] = signed_main_axis([r[:3] for r in recent])
    out["shank"] = signed_main_axis([r[3:] for r in recent])
    return out


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>SAKER Live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font:16px -apple-system,Arial,sans-serif;background:#111a29;color:#edf3fa;max-width:900px;margin:auto;padding:18px}
.card{background:#1b293d;border:1px solid #354962;border-radius:14px;padding:18px;margin:12px 0}
#step{font-size:40px;font-weight:800;color:#7cc4ff}#left{font-size:64px;font-weight:800}
button{background:#276cbd;color:#fff;border:0;border-radius:10px;padding:16px 28px;font-size:18px;font-weight:700;margin-right:8px}
button.stop{background:#c44848}.muted{color:#aab9ca}.ok{color:#5fd38d}.bad{color:#ff7b7b}
canvas{width:100%;height:220px;background:#0d1522;border-radius:10px}
.bar{height:10px;background:#0d1522;border-radius:5px;overflow:hidden}.bar div{height:100%;background:#7cc4ff;width:0}
</style></head><body>
<h1>SAKER Live</h1>
<div class="card"><span id="conn">...</span> <span class="muted" id="info"></span></div>
<div class="card" id="session">
  <div class="muted">Now</div><div id="step">Press START when the sensors are on</div>
  <div id="left"></div><div class="muted" id="next"></div><div class="bar"><div id="prog"></div></div><br>
  <button onclick="post('/start')">START</button><button class="stop" onclick="post('/stop')">STOP</button>
  <p class="muted">The Mac reads each instruction out loud. Session: stand 10 s, normal 2 min, stand, slow 1.5 min,
  fast 1.5 min, turns 1 min, stop-and-go 1 min, stand. About 8 minutes.</p>
</div>
<div class="card"><div class="muted">Shank (below knee), last 6 s</div><canvas id="shank"></canvas>
<div class="muted">Thigh (above knee), last 6 s</div><canvas id="thigh"></canvas></div>
<script>
const $ = id => document.getElementById(id);
async function post(p){ await fetch(p,{method:'POST'}); }
function plot(id, v, color){
  const c=$(id), w=c.width=c.clientWidth*2, h=c.height=c.clientHeight*2, g=c.getContext('2d');
  g.strokeStyle='#354962'; g.beginPath(); g.moveTo(0,h/2); g.lineTo(w,h/2); g.stroke();
  if(!v.length) return; const m=Math.max(150,...v.map(Math.abs));
  g.strokeStyle=color; g.lineWidth=3; g.beginPath();
  v.forEach((y,i)=>{const x=i*w/599, yy=h/2-y/m*(h/2-8); i?g.lineTo(x,yy):g.moveTo(x,yy);}); g.stroke();
}
async function tick(){
  try{
    const d = await (await fetch('/status')).json();
    $('conn').innerHTML = d.connected ? '<b class="ok">ESP connected</b>' : '<b class="bad">No data from ESP</b>';
    $('info').textContent = d.connected ? ` ${d.rate} samples/s` : ' (check it is powered and on the Wi-Fi)';
    if(d.running){ $('step').textContent=d.step; $('left').textContent=d.left+' s'; $('next').textContent='Next: '+d.next;
      $('prog').style.width=d.progress+'%'; }
    else { $('step').textContent = d.rows ? `Saved ${d.rows} samples (${(d.rows/6000).toFixed(1)} min) to ${d.file}` :
      'Press START when the sensors are on'; $('left').textContent=''; $('next').textContent=''; $('prog').style.width='0'; }
    plot('shank', d.shank, '#7cc4ff'); plot('thigh', d.thigh, '#f5b942');
  }catch(e){ $('conn').innerHTML='<b class="bad">Live view stopped</b>'; }
}
setInterval(tick, 250); tick();
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, kind):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        if self.path == "/status":
            self.send(200, json.dumps(status()), "application/json")
        else:
            self.send(200, PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        with S.lock:
            if self.path == "/start" and not S.running:
                S.start()
            elif self.path == "/stop" and S.running:
                S.stop()
        self.send(200, "{}", "application/json")


if __name__ == "__main__":
    threading.Thread(target=receiver, daemon=True).start()
    print(f"Live view at http://localhost:{HTTP_PORT}")
    http.server.ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), Handler).serve_forever()
