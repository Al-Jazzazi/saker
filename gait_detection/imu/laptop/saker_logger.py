#!/usr/bin/env python3
"""Record the SAKER sender's 100 Hz IMU stream to a CSV file.

Over the USB cable (most reliable; close the Arduino Serial Monitor first):
    python3 saker_logger.py --serial /dev/cu.usbserial-0001 --label walk1
    python3 saker_logger.py --serial /dev/cu.usbserial-0001 --track --label walk1
        --track also runs CAL -> WALK -> TRACK for you and tells you when to stand / walk.
Over Wi-Fi, just listening to the sender's broadcast (works even when the hotspot
drops laptop -> ESP traffic):
    python3 saker_logger.py --listen --label walk1
Over Wi-Fi, asking the sender directly:
    python3 saker_logger.py --host 172.20.10.2 --label walk1
Stop with Ctrl+C (or --seconds N). Files land in ./data/. No extra packages needed.
"""
import argparse
import array
import csv
import datetime
import fcntl
import os
import pathlib
import socket
import sys
import termios
import time

SUBSCRIBE_EVERY_S = 1.0   # Wi-Fi: the sender stops streaming 3 s after the last SUB
SERIAL_BAUD = 921600
IOSSIOSPEED = 0x80045402  # macOS ioctl for baud rates above 230400


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--serial", help="USB port of the sender, e.g. /dev/cu.usbserial-0001")
    p.add_argument("--listen", action="store_true", help="Wi-Fi: only listen to the sender's broadcast")
    p.add_argument("--host", default="saker.local", help="Wi-Fi: sender IP or name (default saker.local)")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--out", default=str(pathlib.Path(__file__).parent / "data"), help="output folder")
    p.add_argument("--label", default="", help="text added to the file name, e.g. walk_slow")
    p.add_argument("--seconds", type=float, default=0, help="stop by itself after this long (0 = until Ctrl+C)")
    p.add_argument("--track", action="store_true", help="run CAL, WALK and TRACK on the sender with prompts")
    return p.parse_args()


class SerialLink:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        a = termios.tcgetattr(self.fd)
        a[0] = 0
        a[1] = 0
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[3] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, a)
        fcntl.ioctl(self.fd, IOSSIOSPEED, array.array("i", [SERIAL_BAUD]), 1)
        termios.tcflush(self.fd, termios.TCIFLUSH)
        self.pending = b""

    def send(self, text):
        os.write(self.fd, (text + "\n").encode())

    def subscribe(self):
        self.send("SUB")

    def read(self):
        try:
            self.pending += os.read(self.fd, 65536)
        except BlockingIOError:
            time.sleep(0.01)
        *lines, self.pending = self.pending.split(b"\n")
        return [l.decode("ascii", "replace").strip() for l in lines]

    def close(self):
        try:
            self.send("UNSUB")
        finally:
            os.close(self.fd)


class UdpLink:
    def __init__(self, host, port):
        try:
            self.addr = (socket.gethostbyname(host), port)
        except socket.gaierror:
            sys.exit(f"Can't find '{host}'. Use --host <IP from the serial monitor>, or --serial.")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.2)

    def send(self, text):
        # Commands go to the dashboard's HTTP endpoint, not the UDP port.
        import urllib.request
        urllib.request.urlopen(urllib.request.Request(
            f"http://{self.addr[0]}/cmd?c={text}", method="POST"), timeout=3).read()

    def subscribe(self):
        self.sock.sendto(b"SUB", self.addr)

    def read(self):
        try:
            data, _ = self.sock.recvfrom(4096)
        except socket.timeout:
            return []
        return data.decode("ascii", "replace").splitlines()

    def close(self):
        try:
            self.sock.sendto(b"UNSUB", self.addr)
        except OSError:
            pass


class ListenLink:
    """Receives the sender's broadcast; never sends anything to it."""

    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", port))
        self.sock.settimeout(0.2)

    def send(self, text):
        print(f"\n(--listen can't send {text}; use the dashboard or --serial)")

    def subscribe(self):
        pass

    def read(self):
        try:
            data, _ = self.sock.recvfrom(4096)
        except socket.timeout:
            return []
        return data.decode("ascii", "replace").splitlines()

    def close(self):
        self.sock.close()


# (seconds after the header arrives, command, what to tell the person)
TRACK_SCRIPT = [
    (1, "CAL", ">>> STAND STILL (3 s calibration)"),
    (6, "WALK", ">>> WALK NORMALLY NOW (12 s)"),
    (20, "TRACK", ">>> KEEP WALKING: recording gait phases. Vary speed, turn, stop and go."),
]


def main():
    args = parse_args()
    if args.serial:
        link, where = SerialLink(args.serial), args.serial
    elif args.listen:
        link, where = ListenLink(5006), "the sender's broadcast (port 5006)"
    else:
        link, where = UdpLink(args.host, args.port), f"{args.host}:{args.port}"

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_{args.label}" if args.label else "")
    path = out_dir / f"{name}.csv"

    header = None
    out_file = writer = None
    rows = dropped = 0
    last_seq = None
    last_row = {}
    last_sub = last_print = 0.0
    last_data = start = time.monotonic()
    header_at = None
    script = list(TRACK_SCRIPT) if args.track else []
    rows_at_print = 0
    stop_at = start + args.seconds if args.seconds else None
    print(f"Asking {where} for data...  Ctrl+C to stop")

    try:
        while True:
            now = time.monotonic()
            if stop_at and now >= stop_at:
                break
            if now - last_sub >= SUBSCRIBE_EVERY_S * (2 if args.serial else 1):
                link.subscribe()
                last_sub = now
            if header_at is not None and script and now - header_at >= script[0][0]:
                _, command, prompt = script.pop(0)
                link.send(command)
                print(f"\n{prompt}")

            lines = link.read()
            if lines:
                last_data = now
            for line in lines:
                if line.startswith("#HDR,"):
                    columns = line[5:].split(",")
                    if header is None:
                        header = columns
                        header_at = now
                        out_file = open(path, "w", newline="")
                        writer = csv.writer(out_file)
                        writer.writerow(header)
                        print(f"Recording to {path}")
                    elif columns != header:
                        print("\nSender's columns changed (new firmware?). Start a new recording.")
                    continue
                if line.startswith("[SAKER]"):
                    print(f"\n{line}")  # sender status messages (CAL done, WALK failed, ...)
                    continue
                if writer is None:
                    continue
                fields = line.split(",")
                if len(fields) != len(header) or not fields[1].isdigit():
                    continue
                seq = int(fields[1])
                if last_seq is not None and seq > last_seq + 1:
                    dropped += seq - last_seq - 1
                last_seq = seq  # a smaller seq means the sender rebooted
                writer.writerow(fields)
                rows += 1
                last_row = dict(zip(header, fields))

            if now - last_print >= 1.0:
                if out_file:
                    out_file.flush()  # a crash or closed terminal loses at most 1 s
                rate = (rows - rows_at_print) / (now - last_print) if last_print else 0
                last_print, rows_at_print = now, rows
                if now - last_data > 3:
                    print("\rNo data: check the cable / Wi-Fi, and that the Serial Monitor is closed.   ", end="")
                else:
                    print(f"\r{rows} samples | {rate:4.0f} Hz | {dropped} lost | "
                          f"mode {last_row.get('mode', '-'):7} phase {last_row.get('phase') or '-':15}", end="")
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            link.close()
        except OSError:
            pass
        if out_file:
            out_file.close()
            print(f"\nSaved {rows} samples ({rows / 6000:.1f} min, {dropped} lost) to {path}")
        else:
            print("\nNothing recorded.")


if __name__ == "__main__":
    main()
