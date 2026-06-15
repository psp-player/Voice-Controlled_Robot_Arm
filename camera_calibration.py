## @file camera_calibration.py
#  @brief Camera intrinsic calibration (host side): captures chessboard views and
#         computes the camera matrix + distortion coefficients used to undistort
#         frames in the detection scripts. Run once per camera.
#  @ingroup python_host

import numpy as np
import cv2 as cv

# -----------------------------
# Chessboard Settings
# -----------------------------
CHESSBOARD = (8, 6)   ##< Inner-corner count of the calibration board.
## Sub-pixel corner-refinement termination criteria.
criteria = (
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

## Template object points for one board view (z=0 plane, unit grid).
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

objpoints = []   ##< Accumulated 3D object points, one set per saved view.
imgpoints = []   ##< Accumulated 2D image corner points, one set per saved view.

# -----------------------------
# Open Camera
# -----------------------------
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 's' to save a calibration frame")
print("Press 'c' to calibrate camera")
print("Press 'q' to quit")

# Live capture loop: detect and draw chessboard corners each frame; 's' stores a
# view, 'c' runs calibration once enough views are collected, 'q' quits.
while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Find chessboard corners
    found, corners = cv.findChessboardCorners(gray, CHESSBOARD, None)

    # If found, draw corners
    if found:

        # Refine corner locations
        corners2 = cv.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        # Draw corners on image
        cv.drawChessboardCorners(frame, CHESSBOARD, corners2, found)

    # Show live feed
    cv.imshow("Live Feed", frame)

    key = cv.waitKey(1)

    # Save frame for calibration
    if key == ord('s'):

        if found:
            objpoints.append(objp)
            imgpoints.append(corners2)

            print(f"Saved calibration image #{len(objpoints)}")

        else:
            print("Chessboard not detected")

    # Perform calibration
    elif key == ord('c'):

        if len(objpoints) < 5:
            print("Need at least 5 images for calibration")
            continue

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            None,
            None
        )

        # print("\nCalibration Successful!")
        # print("\nCamera Matrix:")
        # print(mtx)

        # print("\nDistortion Coefficients:")
        # print(dist)
        np.savez("calib.npz", mtx=mtx, dist=dist)

    # Quit
    elif key == ord('q'):
        break

# Cleanup
cap.release()
cv.destroyAllWindows()
