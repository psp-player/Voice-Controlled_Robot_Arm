#!/usr/bin/env python3
"""
Voice + vision pick pipeline for the 3-DoF robot arm  (laptop side).

Say a color ("red", "black", ...).  Vosk recognizes it (grammar-constrained),
OpenCV finds the largest block of that color in the camera frame, the two-stage
transform maps pixel -> table -> robot coordinates, and the pick is sent as an
ASCII line over serial to the STM32 with an ACK handshake.  The MCU receives
coordinates + does IK.

Backbone is the chessboard/affine vision+serial pipeline; the only thing voice
adds is *which* color the vision stage looks for on each command.

Keys (preview window):
    w - calibrate workspace homography (chessboard flat in view)
    q - quit
Voice:
    say an enabled color to pick the largest block of that color.

Deps:
    pip install vosk sounddevice pyserial opencv-python numpy
    Model: download e.g. vosk-model-small-en-us-0.15 and point MODEL_PATH at it.
"""

import os
import json
import time
import queue
import threading
import numpy as np
import cv2 as cv
import sounddevice as sd
import serial                       # pip install pyserial
from vosk import Model, KaldiRecognizer

# ----------------------------- CONFIG -----------------------------
# Which colors are live this run. Drives both the voice grammar and which
# masks get computed. Must be keys in COLOR_RANGES below.
ENABLED_COLORS = ["RED", "BLACK"]

CHESSBOARD     = (8, 6)
SQUARE_SIZE_MM = 25.0
MIN_AREA       = 800
KERNEL         = np.ones((5, 5), np.uint8)
PICK_Z_MM      = 5.0                # approach height in robot frame

SERIAL_PORT    = "COM5"             # "/dev/ttyACM0" on Linux/Mac
BAUD           = 115200
CAM_INDEX      = 0
ACK_TIMEOUT_S  = 15.0               # how long to wait for the MCU to report DONE

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(SCRIPT_DIR, "vosk-model-small-en-us-0.15")
SAMPLE_RATE    = 16000

CALIB_FILE      = "camera_calib.npz"
HOMOGRAPHY_FILE = "workspace_homography.npz"
ROBOT_TF_FILE   = "robot_transform.npz"

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# HSV ranges. OpenCV: H 0-179, S/V 0-255.
# RED wraps the hue circle -> two bands. BLACK is not a hue -> low-V band
# (any H/S, V capped); tune the V ceiling (60) to your lighting. Same
# cv.inRange logic handles both, so black is just another entry here.
COLOR_RANGES = {
    "RED": {
        "ranges": [
            (np.array([0,   120, 70]),  np.array([10,  255, 255])),
            (np.array([170, 120, 70]),  np.array([179, 255, 255])),
        ],
        "bgr": (0, 0, 255),
    },
    "GREEN": {
        "ranges": [(np.array([40, 80, 70]), np.array([80, 255, 255]))],
        "bgr": (0, 255, 0),
    },
    "BLUE": {
        "ranges": [(np.array([100, 120, 70]), np.array([130, 255, 255]))],
        "bgr": (255, 0, 0),
    },
    "YELLOW": {
        "ranges": [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
        "bgr": (0, 255, 255),
    },
    "BLACK": {
        "ranges": [(np.array([0, 0, 0]), np.array([179, 90, 60]))],
        "bgr": (0, 0, 0),
    },
}

# Serial framing. PC->MCU 'M,<COLOR>,<X>,<Y>,<Z>\n'  |  MCU->PC 'DONE'/'ERR'.
# <-- if your firmware speaks the fire-and-forget 'GOTO x y\n' format instead,
#     swap send_pick() for a plain ser.write() and drop the ACK wait.
# ------------------------------------------------------------------

# Shared state between the voice thread and the vision loop
audio_q      = queue.Queue()
target_color = None                 # set by voice thread, e.g. "RED"
new_command  = threading.Event()
stop_flag    = threading.Event()


# -----------------------------
# Voice (Vosk) thread
# -----------------------------
def audio_cb(indata, frames, time_, status):
    if status:
        print(status, flush=True)
    audio_q.put(bytes(indata))


def recognizer_loop():
    """Vosk with a color-only grammar; updates target_color on each command."""
    global target_color
    spoken = [c.lower() for c in ENABLED_COLORS]
    grammar = json.dumps(spoken + ["[unk]"])        # [unk] lets it reject non-colors
    if not os.path.isdir(os.path.join(MODEL_PATH, "am")):
        print(f"[voice] ERROR: no Vosk model at {MODEL_PATH}")
        print("        That folder must directly contain am/ conf/ graph/ ...")
        print("        Check the name matches exactly and isn't double-nested")
        print("        (e.g. vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15/).")
        return
    model = Model(MODEL_PATH)
    rec   = KaldiRecognizer(model, SAMPLE_RATE, grammar)
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000,
                           dtype="int16", channels=1, callback=audio_cb):
        print("[voice] listening… say a color:", ", ".join(spoken))
        while not stop_flag.is_set():
            data = audio_q.get()
            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get("text", "").strip()
                if text in spoken:
                    target_color = text.upper()
                    new_command.set()
                    print(f"[voice] -> {text}")


# -----------------------------
# Load saved calibration
# -----------------------------
mtx = dist = None
if os.path.exists(CALIB_FILE):
    data = np.load(CALIB_FILE)
    mtx, dist = data["mtx"], data["dist"]

H = None
if os.path.exists(HOMOGRAPHY_FILE):
    H = np.load(HOMOGRAPHY_FILE)["H"]

M_robot = None
if os.path.exists(ROBOT_TF_FILE):
    M_robot = np.load(ROBOT_TF_FILE)["M"]


# -----------------------------
# Geometry helpers
# -----------------------------
def compute_workspace_homography(gray):
    found, corners = cv.findChessboardCorners(gray, CHESSBOARD, None)
    if not found:
        return None
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    world = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 2), np.float32)
    world[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
    world *= SQUARE_SIZE_MM
    homography, _ = cv.findHomography(corners.reshape(-1, 2), world)
    return homography


def pixel_to_table(u, v, H):
    w = H @ np.array([u, v, 1.0])
    w /= w[2]
    return float(w[0]), float(w[1])


def table_to_robot(x, y, M):
    p = M @ np.array([x, y, 1.0])
    return float(p[0]), float(p[1])


# -----------------------------
# Detection
# -----------------------------
def make_mask(hsv, color_def):
    mask = None
    for lower, upper in color_def["ranges"]:
        part = cv.inRange(hsv, lower, upper)
        mask = part if mask is None else cv.bitwise_or(mask, part)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, KERNEL)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, KERNEL)
    return mask


def detect_blocks(frame):
    """Detect every enabled color in the frame (used for the live overlay)."""
    blurred = cv.GaussianBlur(frame, (5, 5), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    detections = []
    for name in ENABLED_COLORS:
        color_def = COLOR_RANGES[name]
        mask = make_mask(hsv, color_def)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < MIN_AREA:
                continue
            mm = cv.moments(cnt)
            if mm["m00"] == 0:
                continue
            cx = int(mm["m10"] / mm["m00"])
            cy = int(mm["m01"] / mm["m00"])
            x, y, w, h = cv.boundingRect(cnt)
            detections.append({
                "name": name, "centroid": (cx, cy), "area": area,
                "bbox": (x, y, w, h), "bgr": color_def["bgr"],
            })
    return detections


def best_detection(detections, color):
    """Largest detected blob of the given color, or None."""
    hits = [d for d in detections if d["name"] == color]
    return max(hits, key=lambda d: d["area"]) if hits else None


# -----------------------------
# Serial link to STM32
# -----------------------------
def send_pick(ser, color, x, y, z):
    """PC->MCU 'M,<COLOR>,<X>,<Y>,<Z>\\n'  |  MCU->PC 'DONE'/'ERR'."""
    line = f"M,{color},{x:.2f},{y:.2f},{z:.2f}\n"
    ser.reset_input_buffer()
    ser.write(line.encode("ascii"))
    print(f"-> {line.strip()}")

    deadline = time.time() + ACK_TIMEOUT_S
    while time.time() < deadline:
        reply = ser.readline().decode("ascii", errors="ignore").strip()
        if not reply:
            continue
        print(f"<- {reply}")
        if reply == "DONE":
            return True
        if reply == "ERR":
            return False
    print("   (timed out waiting for MCU)")
    return False


# -----------------------------
# Main
# -----------------------------
def main():
    global H, M_robot

    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.5)
        time.sleep(2)               # let the STM32 reset/boot after the port opens
        print(f"Connected to STM32 on {SERIAL_PORT}")
    except serial.SerialException as e:
        print(f"No serial link ({e}). Running VISION-ONLY - picks are disabled.")

    cap = cv.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"camera {CAM_INDEX} not opening")

    threading.Thread(target=recognizer_loop, daemon=True).start()

    print("\n'w' - calibrate workspace (chessboard flat in view)")
    print("'q' - quit")
    print("say a color to pick it\n")

    last = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if mtx is not None:
                frame = cv.undistort(frame, mtx, dist)

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            detections = detect_blocks(frame)

            # One detection + report per spoken command
            if new_command.is_set():
                color = target_color
                new_command.clear()
                d = best_detection(detections, color)
                if d is None:
                    print(f"[vision] no {color} block found")
                    last = None
                else:
                    cx, cy = d["centroid"]
                    msg = f"[vision] {color} px({cx},{cy}) area={d['area']:.0f}"
                    if H is not None and M_robot is not None:
                        # Calibrated: report robot coords, send if serial is up
                        tx, ty = pixel_to_table(cx, cy, H)
                        rx, ry = table_to_robot(tx, ty, M_robot)
                        msg += f" -> robot({rx:.1f},{ry:.1f})"
                        print(msg)
                        if ser is not None:
                            send_pick(ser, color, rx, ry, PICK_Z_MM)
                        else:
                            print("   (vision-only, not sent)")
                    else:
                        # Not calibrated: pixel centroid only — enough to test
                        # voice + detection standalone
                        print(msg + "   (not calibrated, pixel only)")
                    last = d

            # Live overlay: all detections, with the last picked one highlighted
            for d in detections:
                x, y, w, h = d["bbox"]
                cx, cy = d["centroid"]
                bgr = d["bgr"]
                thick = 3 if d is last else 2
                cv.rectangle(frame, (x, y), (x + w, y + h), bgr, thick)
                cv.circle(frame, (cx, cy), 4, bgr, -1)
                label = d["name"]
                if H is not None:
                    tx, ty = pixel_to_table(cx, cy, H)
                    if M_robot is not None:
                        rx, ry = table_to_robot(tx, ty, M_robot)
                        label = f"{d['name']} R({rx:.0f},{ry:.0f})"
                    else:
                        label = f"{d['name']} T({tx:.0f},{ty:.0f})"
                cv.putText(frame, label, (x, y - 8),
                           cv.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2)

            cv.imshow("voice + vision pick (q to quit)", frame)
            key = cv.waitKey(1) & 0xFF

            if key == ord('w'):
                new_H = compute_workspace_homography(gray)
                if new_H is not None:
                    H = new_H
                    np.savez(HOMOGRAPHY_FILE, H=H)
                    print("Workspace homography saved.")
                else:
                    print("Chessboard not detected.")
            elif key == ord('q'):
                break
    finally:
        stop_flag.set()
        cap.release()
        cv.destroyAllWindows()
        if ser is not None:
            ser.close()


if __name__ == "__main__":
    main()