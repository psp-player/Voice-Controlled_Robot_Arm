#!/usr/bin/env python3
## @defgroup python_host Python Host Pipeline
#  @brief Laptop-side voice recognition, computer vision, and the two-stage
#         pixel -> table -> robot calibration that feeds Cartesian pick targets
#         to the STM32 over serial. Comprises three scripts: live voice+vision
#         picking, vision-only detect/sort, and camera intrinsic calibration.

## @file voiceplusvision.py
#  @brief Voice + vision pick pipeline (host side): say a color, and the largest
#         block of that color is located, transformed into the robot frame, and
#         sent to the arm with an ACK handshake.
#  @ingroup python_host
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
## Colors live this run; drives both the voice grammar and which masks run.
#  Must be keys in COLOR_RANGES.
ENABLED_COLORS = ["RED", "BLUE"]

CHESSBOARD     = (8, 6)              ##< Inner-corner count of the calibration board.
SQUARE_SIZE_MM = 25.0               ##< Physical chessboard square size (mm).
MIN_AREA       = 800                ##< Minimum contour area (px) to count as a block.
KERNEL         = np.ones((5, 5), np.uint8)  ##< Morphological kernel for mask cleanup.
PICK_Z_MM      = 100.0              ##< Approach height in the robot frame (mm).

SERIAL_PORT    = "COM8"             ##< STM32 virtual COM port ("/dev/ttyACM0" on Linux/Mac).
BAUD           = 115200             ##< Serial baud rate.
CAM_INDEX      = 0                  ##< OpenCV camera index.
ACK_TIMEOUT_S  = 15.0               ##< Seconds to wait for the MCU to report DONE.

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))  ##< Folder of this script.
MODEL_PATH     = os.path.join(SCRIPT_DIR, "vosk-model-small-en-us-0.15")  ##< Vosk model dir.
SAMPLE_RATE    = 16000             ##< Microphone sample rate (Hz) for Vosk.

CALIB_FILE      = "camera_calib.npz"        ##< Saved camera intrinsics (mtx, dist).
HOMOGRAPHY_FILE = "workspace_homography.npz"  ##< Saved pixel->table homography (H).
ROBOT_TF_FILE   = "robot_transform.npz"     ##< Saved table->robot affine (M).

## Sub-pixel corner-refinement termination criteria for the chessboard.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

## HSV color definitions (OpenCV: H 0-179, S/V 0-255).
#  RED wraps the hue circle, so it uses two bands. BLACK (commented out) is a
#  low-V band rather than a hue. The same cv.inRange logic handles every entry.
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
    # "BLACK": {
    #     "ranges": [(np.array([0, 0, 0]), np.array([179, 90, 60]))],
    #     "bgr": (0, 0, 0),
    # },
}

# Serial framing. PC->MCU 'M,<COLOR>,<X>,<Y>,<Z>\n'  |  MCU->PC 'DONE'/'ERR'.
# <-- if your firmware speaks the fire-and-forget 'GOTO x y\n' format instead,
#     swap send_pick() for a plain ser.write() and drop the ACK wait.
# ------------------------------------------------------------------

# Shared state between the voice thread and the vision loop
audio_q      = queue.Queue()        ##< Thread-safe queue of raw mic audio blocks.
target_color = None                 ##< Color requested by the voice thread (e.g. "RED").
new_command  = threading.Event()    ##< Set when a fresh voice command is ready.
stop_flag    = threading.Event()    ##< Set on shutdown to stop the voice thread.


# -----------------------------
# Voice (Vosk) thread
# -----------------------------
def audio_cb(indata, frames, time_, status):
    """! @brief sounddevice callback: push raw mic blocks onto the audio queue.
    @param indata  Raw audio buffer from the input stream.
    @param frames  Number of frames in this block (unused).
    @param time_   Stream timestamps (unused).
    @param status  Stream status flags; printed if non-zero.
    """
    if status:
        print(status, flush=True)
    audio_q.put(bytes(indata))


def recognizer_loop():
    """! @brief Vosk recognition thread with a color-only grammar.

    Runs until ::stop_flag is set, pulling audio from ::audio_q. On a recognized
    color it updates the global ::target_color and sets ::new_command so the
    vision loop performs one detection. The grammar is constrained to the enabled
    colors plus "[unk]" so non-color speech is rejected.
    """
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
mtx = dist = None                   ##< Camera intrinsics (matrix, distortion) once loaded.
if os.path.exists(CALIB_FILE):
    data = np.load(CALIB_FILE)
    mtx, dist = data["mtx"], data["dist"]

H = None                            ##< Pixel->table homography once loaded/calibrated.
if os.path.exists(HOMOGRAPHY_FILE):
    H = np.load(HOMOGRAPHY_FILE)["H"]

M_robot = None                      ##< Table->robot affine transform once loaded/calibrated.
if os.path.exists(ROBOT_TF_FILE):
    M_robot = np.load(ROBOT_TF_FILE)["M"]


# -----------------------------
# Geometry helpers
# -----------------------------
def compute_workspace_homography(gray):
    """! @brief Estimate the pixel -> table-mm homography from a chessboard.
    @param gray  Grayscale frame containing the calibration board lying flat.
    @return 3x3 homography matrix, or None if the board was not found.
    """
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
    """! @brief Map an image pixel to table-frame millimetres via homography.
    @param u  Pixel column.
    @param v  Pixel row.
    @param H  Pixel->table homography (3x3).
    @return (x, y) in table millimetres.
    """
    w = H @ np.array([u, v, 1.0])
    w /= w[2]
    return float(w[0]), float(w[1])


def table_to_robot(x, y, M):
    """! @brief Map a table-frame point into the robot base frame via affine M.
    @param x  Table-frame X (mm).
    @param y  Table-frame Y (mm).
    @param M  Table->robot affine (3x3, last row [0,0,1]).
    @return (x, y) in robot-frame millimetres.
    """
    p = M @ np.array([x, y, 1.0])
    return float(p[0]), float(p[1])


# -----------------------------
# Detection
# -----------------------------
def make_mask(hsv, color_def):
    """! @brief Build a cleaned binary mask for one color definition.

    ORs together each HSV band in @p color_def (so multi-band colors like red
    work), then applies morphological open+close to remove speckle and fill gaps.

    @param hsv        Frame already converted to HSV.
    @param color_def  Entry from ::COLOR_RANGES (its "ranges" list is used).
    @return Single-channel binary mask.
    """
    mask = None
    for lower, upper in color_def["ranges"]:
        part = cv.inRange(hsv, lower, upper)
        mask = part if mask is None else cv.bitwise_or(mask, part)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, KERNEL)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, KERNEL)
    return mask


def detect_blocks(frame):
    """! @brief Detect every enabled color in a frame (for the live overlay).

    Blurs and converts to HSV, masks each enabled color, and returns one record
    per contour above ::MIN_AREA with its centroid, area, bounding box, and color.

    @param frame  BGR camera frame.
    @return List of detection dicts (name, centroid, area, bbox, bgr).
    """
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
    """! @brief Pick the largest detected blob of a given color.
    @param detections  Output of ::detect_blocks.
    @param color       Color name to filter on.
    @return The largest-area detection of that color, or None.
    """
    hits = [d for d in detections if d["name"] == color]
    return max(hits, key=lambda d: d["area"]) if hits else None


# -----------------------------
# Serial link to STM32
# -----------------------------
def send_pick(ser, color, x, y, z):
    """! @brief Send a pick command and wait for the MCU's ACK.

    Frames 'M,<COLOR>,<X>,<Y>,<Z>\\n', flushes input, transmits, and waits up to
    ::ACK_TIMEOUT_S for 'DONE' (success) or 'ERR' (unreachable / rejected).

    @param ser    Open pyserial port.
    @param color  Color label sent with the command.
    @param x      Robot-frame X (mm).
    @param y      Robot-frame Y (mm).
    @param z      Robot-frame Z (mm).
    @return True on 'DONE', False on 'ERR' or timeout.
    """
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

## Robot-frame (x,y) targets the arm is driven to during transform calibration.
ROBOT_CALIB_POINTS = [
    (250.0,  30.0),
    (250.0,  50.0),
    (300.0,  30.0),
    (300.0,  50.0),
]

def query_robot_pos(ser):
    """! @brief Ask the MCU for its current tip position via the 'WHERE' command.
    @param ser  Open pyserial port.
    @return (x, y) in robot millimetres, or None on timeout.
    """
    ser.reset_input_buffer()
    ser.write(b"WHERE\n")
    deadline = time.time() + 3.0
    while time.time() < deadline:
        reply = ser.readline().decode("ascii", errors="ignore").strip()
        if reply.startswith("POS,"):
            _, sx, sy, sz = reply.split(",")
            return float(sx), float(sy)
    return None

def calibrate_robot_transform(ser, cap):
    """! @brief Interactively build the table->robot affine from correspondences.

    For each point in ::ROBOT_CALIB_POINTS the arm drives to a known robot (x,y);
    the user places a block under the tip and presses SPACE, and vision reads the
    block's table coordinate. The collected (table, robot) pairs are fit with
    cv.estimateAffine2D, saved to ::ROBOT_TF_FILE, and the residual error is
    reported so a poor calibration is obvious.

    @param ser  Open pyserial port (required to drive the arm).
    @param cap  Open cv.VideoCapture for reading the block position.
    """
    global M_robot
    if H is None:
        print("[calib] Need workspace homography first — press 'w'.")
        return
    if ser is None:
        print("[calib] Need serial link to drive the arm.")
        return

    table_pts, robot_pts = [], []
    print("\n=== ROBOT TRANSFORM CALIBRATION ===")
    print("For each point: arm moves, place a BLACK block under the tip, press SPACE.")
    print("Press ESC to abort.\n")

    for i, (rx, ry) in enumerate(ROBOT_CALIB_POINTS):
        # drive the arm to the known robot point (z high enough to clear, then it holds)
        line = f"M,CAL,{rx:.2f},{ry:.2f},{PICK_Z_MM:.2f}\n"
        ser.reset_input_buffer()
        ser.write(line.encode("ascii"))
        # wait for the move to finish
        deadline = time.time() + ACK_TIMEOUT_S
        ok = False
        while time.time() < deadline:
            r = ser.readline().decode("ascii", errors="ignore").strip()
            if r == "DONE": ok = True; break
            if r == "ERR":  break
        if not ok:
            print(f"[calib] point {i+1}: arm couldn't reach ({rx},{ry}), skipping.")
            continue

        print(f"[calib] point {i+1}/{len(ROBOT_CALIB_POINTS)}: "
              f"arm at robot({rx},{ry}). Place a block under the tip, press SPACE.")

        # let the user place the block; grab the table coord on SPACE
        captured = None
        while True:
            okf, frame = cap.read()
            if not okf: continue
            if mtx is not None: frame = cv.undistort(frame, mtx, dist)
            dets = detect_blocks(frame)
            # show what we see
            for d in dets:
                bx, by, bw, bh = d["bbox"]
                cv.rectangle(frame, (bx,by), (bx+bw,by+bh), d["bgr"], 2)
            cv.putText(frame, f"point {i+1}: place block, SPACE to capture",
                       (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv.imshow("voice + vision pick (q to quit)", frame)
            k = cv.waitKey(1) & 0xFF
            if k == 27:   # ESC
                print("[calib] aborted."); return
            if k == 32:   # SPACE
                if not dets:
                    print("   no block detected — adjust and try again."); continue
                d = max(dets, key=lambda d: d["area"])   # largest blob = the block
                cx, cy = d["centroid"]
                tx, ty = pixel_to_table(cx, cy, H)
                captured = (tx, ty)
                break

        table_pts.append(captured)
        robot_pts.append((rx, ry))
        print(f"   captured table({captured[0]:.1f},{captured[1]:.1f}) "
              f"<-> robot({rx},{ry})")

    if len(table_pts) < 3:
        print("[calib] need at least 3 points — got "
              f"{len(table_pts)}. Aborting."); return

    table_np = np.array(table_pts, dtype=np.float32)
    robot_np = np.array(robot_pts, dtype=np.float32)
    M2x3, inliers = cv.estimateAffine2D(table_np, robot_np)
    if M2x3 is None:
        print("[calib] affine solve failed."); return

    M = np.vstack([M2x3, [0, 0, 1]]).astype(np.float64)
    np.savez(ROBOT_TF_FILE, M=M)
    M_robot = M

    # report residual error so you know if it's any good
    errs = []
    for (tx,ty),(rx,ry) in zip(table_pts, robot_pts):
        px = M @ np.array([tx,ty,1.0]); 
        errs.append(np.hypot(px[0]-rx, px[1]-ry))
    print(f"[calib] SAVED. mean residual {np.mean(errs):.1f} mm, "
          f"max {np.max(errs):.1f} mm ({len(table_pts)} pts).")
    if np.max(errs) > 10:
        print("   WARNING: >10mm error — re-check block placement accuracy.")

def run_ik_demo(ser):
    """! @brief Run a scripted sequence of fixed reachable poses (no vision).

    Streams known-good Cartesian waypoints to the arm to show smooth coordinated
    motion and the gripper without needing calibration. Each move waits for DONE
    before the next. Bound to the 'd' key in the main loop.

    @param ser  Open pyserial port.
    """
    if ser is None:
        print("[demo] No serial link — can't run demo.")
        return

    # Known-reachable robot-frame waypoints (x, y, z) in mm.
    # All within D1±(L1+L2); spread across the workspace to show range.
    # Verify each in PuTTY first so you KNOW they reach on demo day.
    sequence = [
        ("M,DEMO,245,0,315",   "home / start"),
        ("M,DEMO,250,80,200",  "reach right + down"),
        ("M,DEMO,250,-80,200", "sweep left"),
        ("M,DEMO,300,0,150",   "reach out low"),
        ("M,DEMO,200,0,300",   "pull in high"),
        ("M,DEMO,245,0,315",   "return home"),
    ]

    print("\n=== IK DEMO (no vision) ===")
    for cmd, label in sequence:
        print(f"[demo] {label}: {cmd}")
        ser.reset_input_buffer()
        ser.write((cmd + "\n").encode("ascii"))
        # wait for the move to finish (DONE) before the next one
        deadline = time.time() + ACK_TIMEOUT_S
        while time.time() < deadline:
            r = ser.readline().decode("ascii", errors="ignore").strip()
            if r == "DONE": break
            if r == "ERR":
                print(f"   ERR — {cmd} unreachable, skipping"); break
        time.sleep(0.3)   # brief pause between poses so it reads as distinct moves
    print("[demo] done.\n")

# -----------------------------
# Main
# -----------------------------
def main():
    """! @brief Program entry point: start voice thread and run the vision loop.

    Opens the serial link (falling back to vision-only if absent) and the camera,
    launches the Vosk recognizer thread, then loops: reads frames, on each voice
    command detects the requested color and (if calibrated and connected) sends a
    pick. Hotkeys: 'w' workspace homography, 'c' table->robot calibration,
    'd' scripted IK demo, 'q' quit. Releases all resources on exit.
    """
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
            elif key == ord('c'):
                calibrate_robot_transform(ser, cap)
            elif key == ord('d'):
                run_ik_demo(ser)

    finally:
        stop_flag.set()
        cap.release()
        cv.destroyAllWindows()
        if ser is not None:
            ser.close()


if __name__ == "__main__":
    main()
