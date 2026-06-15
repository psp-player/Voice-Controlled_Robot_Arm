## @file block_detection.py
#  @brief Vision-only detect-and-sort (host side): finds every enabled color in
#         the camera frame and, on a key press, sends each block's robot-frame
#         coordinate to the arm. No voice; the manual counterpart to
#         voiceplusvision.py.
#  @ingroup python_host

import os
import time
import numpy as np
import cv2 as cv
import serial   # pip install pyserial

# -----------------------------
# Config
# -----------------------------
CHESSBOARD = (8, 6)         ##< Inner-corner count of the calibration board.
SQUARE_SIZE_MM = 25.0       ##< Physical chessboard square size (mm).
MIN_AREA = 800              ##< Minimum contour area (px) to count as a block.
KERNEL = np.ones((5, 5), np.uint8)  ##< Morphological kernel for mask cleanup.
PICK_Z_MM = 5.0             ##< Approach height in the robot frame (mm).

SERIAL_PORT = "COM5"        ##< STM32 virtual COM port (e.g. /dev/ttyACM0 on Linux).
BAUD = 115200               ##< Serial baud rate.
ACK_TIMEOUT_S = 15.0        ##< Seconds to wait for the MCU to report DONE.

CALIB_FILE = "camera_calib.npz"             ##< Saved camera intrinsics (mtx, dist).
HOMOGRAPHY_FILE = "workspace_homography.npz"  ##< Saved pixel->table homography (H).
ROBOT_TF_FILE = "robot_transform.npz"       ##< Saved table->robot affine (M).

## Sub-pixel corner-refinement termination criteria for the chessboard.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

## HSV color definitions (OpenCV: H 0-179, S/V 0-255). RED wraps the hue circle
#  so it uses two bands; every entry is matched with the same cv.inRange logic.
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
}

# -----------------------------
# Table -> Robot calibration
# -----------------------------
## Manual table->robot correspondences: ((table_x, table_y), (robot_x, robot_y)).
#  Fill once with 4+ well-spread points; used if no saved transform file exists.
TABLE_ROBOT_POINTS = [
    # ((0.0,   0.0),   (rx0, ry0)),
    # ((175.0, 0.0),   (rx1, ry1)),
    # ((175.0, 125.0), (rx2, ry2)),
    # ((0.0,   125.0), (rx3, ry3)),
]


def fit_table_to_robot(points):
    """! @brief Fit a full 2D affine (rotation, scale, reflection, slight shear).
    @param points  List of ((table_x, table_y), (robot_x, robot_y)) pairs.
    @return 2x3 affine matrix from cv.estimateAffine2D, or None if <3 points.
    """
    if len(points) < 3:
        return None
    src = np.array([p[0] for p in points], dtype=np.float32)
    dst = np.array([p[1] for p in points], dtype=np.float32)
    M, _ = cv.estimateAffine2D(src, dst)
    return M


# -----------------------------
# Load saved calibration
# -----------------------------
mtx = dist = None                   ##< Camera intrinsics (matrix, distortion) once loaded.
if os.path.exists(CALIB_FILE):
    data = np.load(CALIB_FILE)
    mtx, dist = data["mtx"], data["dist"]

H = None                            ##< Pixel->table homography once loaded.
if os.path.exists(HOMOGRAPHY_FILE):
    H = np.load(HOMOGRAPHY_FILE)["H"]

M_robot = None                      ##< Table->robot affine; loaded from file or fit from points.
if os.path.exists(ROBOT_TF_FILE):
    M_robot = np.load(ROBOT_TF_FILE)["M"]
elif TABLE_ROBOT_POINTS:
    M_robot = fit_table_to_robot(TABLE_ROBOT_POINTS)
    if M_robot is not None:
        np.savez(ROBOT_TF_FILE, M=M_robot)


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
    @param M  Table->robot affine (3x3).
    @return (x, y) in robot-frame millimetres.
    """
    p = M @ np.array([x, y, 1.0])
    return float(p[0]), float(p[1])


# -----------------------------
# Detection
# -----------------------------
def make_mask(hsv, color_def):
    """! @brief Build a cleaned binary mask for one color definition.

    ORs each HSV band in @p color_def together, then applies morphological
    open+close to remove speckle and fill small gaps.

    @param hsv        Frame already converted to HSV.
    @param color_def  Entry from ::COLOR_RANGES.
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
    """! @brief Detect all configured colors in a frame.

    Blurs, converts to HSV, masks each color in ::COLOR_RANGES, and returns one
    record per contour above ::MIN_AREA with its centroid, bounding box, and color.

    @param frame  BGR camera frame.
    @return List of detection dicts (name, centroid, bbox, bgr).
    """
    blurred = cv.GaussianBlur(frame, (5, 5), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    detections = []
    for name, color_def in COLOR_RANGES.items():
        mask = make_mask(hsv, color_def)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv.contourArea(cnt) < MIN_AREA:
                continue
            mm = cv.moments(cnt)
            if mm["m00"] == 0:
                continue
            cx = int(mm["m10"] / mm["m00"])
            cy = int(mm["m01"] / mm["m00"])
            x, y, w, h = cv.boundingRect(cnt)
            detections.append({
                "name": name, "centroid": (cx, cy),
                "bbox": (x, y, w, h), "bgr": color_def["bgr"],
            })
    return detections


# -----------------------------
# Serial link to STM32
# -----------------------------
def send_pick(ser, color, x, y, z):
    """! @brief Send a pick command and wait for the MCU's ACK.

    Frames 'M,<COLOR>,<X>,<Y>,<Z>\\n', transmits, and waits up to ::ACK_TIMEOUT_S
    for 'DONE' (success) or 'ERR' (rejected).

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


# -----------------------------
# Main
# -----------------------------
ser = None
try:
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.5)
    time.sleep(2)   # let the STM32 reset/boot after the port opens
    print(f"Connected to STM32 on {SERIAL_PORT}")
except serial.SerialException as e:
    print(f"No serial link ({e}). Running VISION-ONLY - 'p' (pick) is disabled.")

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit

print("\n'w' - calibrate workspace (chessboard flat in view)")
print("'p' - pick & sort every detected block")
print("'q' - quit\n")

# Main loop: draw live detections (with table/robot coords when calibrated);
# 'w' recalibrates the workspace homography, 'p' sends a pick for each block,
# 'q' quits.
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if mtx is not None:
        frame = cv.undistort(frame, mtx, dist)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detections = detect_blocks(frame)

    for d in detections:
        x, y, w, h = d["bbox"]
        cx, cy = d["centroid"]
        bgr = d["bgr"]
        cv.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
        cv.circle(frame, (cx, cy), 4, bgr, -1)

        label = d["name"]
        if H is not None:
            tx, ty = pixel_to_table(cx, cy, H)
            if M_robot is not None:
                rx, ry = table_to_robot(tx, ty, M_robot)
                label = f"{d['name']} R({rx:.0f},{ry:.0f})"
                d["robot"] = (rx, ry)
            else:
                label = f"{d['name']} T({tx:.0f},{ty:.0f})"
        cv.putText(frame, label, (x, y - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2)

    cv.imshow("Vision -> STM32", frame)
    key = cv.waitKey(1)

    if key == ord('w'):
        new_H = compute_workspace_homography(gray)
        if new_H is not None:
            H = new_H
            np.savez(HOMOGRAPHY_FILE, H=H)
            print("Workspace homography saved.")
        else:
            print("Chessboard not detected.")

    elif key == ord('p'):
        if ser is None:
            print("No serial link - can't send picks (vision-only mode).")
            continue
        if M_robot is None:
            print("No table->robot transform yet. Fill TABLE_ROBOT_POINTS first.")
            continue
        for d in detections:
            if "robot" not in d:
                continue
            rx, ry = d["robot"]
            send_pick(ser, d["name"], rx, ry, PICK_Z_MM)

    elif key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
if ser is not None:
    ser.close()
