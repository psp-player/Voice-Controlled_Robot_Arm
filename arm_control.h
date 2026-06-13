/**
 * @file    arm_control.h
 * @brief   Public API for the 3-DoF (RRR) robotic arm: kinematics, motion,
 *          TMC2240 stepper driver bring-up, and gripper control.
 *
 * @details
 * This module owns all joint-space motion for the arm. Callers work in
 * Cartesian millimetres in the robot base frame; the module solves inverse
 * kinematics, converts joint angles to microsteps, and generates step pulses
 * on a per-joint hardware timer with a coordinated trapezoidal speed profile.
 *
 * Typical bring-up order:
 *   1. MX_*_Init()            (CubeMX peripheral init: SPI1, TIM1/2/3/4, GPIO)
 *   2. arm_init()             set joint angles to the known home pose
 *   3. arm_drivers_init()     configure the three TMC2240 drivers over SPI
 *   4. gripper_init()         start the gripper servo PWM
 *   5. arm_move_to() / gripper_open() / ... as commanded
 *
 * @note  Joint indices are 0 = base (yaw), 1 = shoulder, 2 = elbow.
 */

#ifndef ARM_CONTROL_H
#define ARM_CONTROL_H

#include <stdint.h>

/**
 * @brief Initialise joint-space state to the home pose.
 *
 * Sets the internal current-angle vector to the mechanical home pose
 * (base = 0, shoulder = +pi/2, elbow = -pi/2). Call once after the motors are
 * physically homed and after MX_*_Init(), so that subsequent relative moves in
 * ::arm_move_to() are referenced from a correct starting configuration.
 */
void arm_init(void);

/**
 * @brief Move the end-effector to a Cartesian target via inverse kinematics.
 *
 * Solves IK for @p (x,y,z), converts the per-joint angular deltas to microsteps,
 * selects each joint's direction, and launches a coordinated multi-joint move.
 * The joint with the largest step count becomes the "master" that drives the
 * shared trapezoidal velocity profile; the other joints are rate-scaled so all
 * three start and finish together.
 *
 * @param x         Target X in the robot base frame (mm).
 * @param y         Target Y in the robot base frame (mm).
 * @param z         Target Z in the robot base frame (mm).
 * @param elbow_up  1 for the elbow-up IK solution, 0 for elbow-down.
 * @return 1 if the target is reachable and motion was launched; 0 if the target
 *         is out of reach or a move is already in progress.
 *
 * @note Non-blocking: returns immediately once stepping starts. Poll
 *       ::arm_is_moving() to detect completion.
 */
int arm_move_to(float x, float y, float z, int elbow_up);

/**
 * @brief Forward kinematics: joint angles -> Cartesian tip position.
 *
 * @param th0  Base (yaw) angle (rad).
 * @param th1  Shoulder angle (rad).
 * @param th2  Elbow angle (rad).
 * @param[out] x  Resulting tip X (mm).
 * @param[out] y  Resulting tip Y (mm).
 * @param[out] z  Resulting tip Z (mm).
 */
void arm_fk(float th0, float th1, float th2, float *x, float *y, float *z);

/**
 * @brief Report whether any joint is still stepping.
 * @return Non-zero while a move is in progress, 0 when all joints are idle.
 *
 * @details Poll after ::arm_move_to() / ::arm_test_spin() to block until a move
 *          completes (e.g. @code while (arm_is_moving()) { } @endcode).
 */
int arm_is_moving(void);

/**
 * @brief Configure all three TMC2240 stepper drivers over SPI.
 *
 * Resets driver status, sets per-joint current limits (IHOLD/IRUN), current
 * range, and chopper mode (SpreadCycle, 16 microsteps), then reads back each
 * driver's version field to confirm communication and enables the ones that
 * respond. Call after MX_SPI1_Init().
 *
 * @return Bitmask of successfully configured drivers (bit 0 = joint 0, etc.).
 *         A value of 0b111 means all three drivers acknowledged.
 */
uint8_t arm_drivers_init(void);

/**
 * @brief Spin a single joint at a constant rate, bypassing IK and the profile.
 *
 * Diagnostic helper for checking wiring, direction, and step rate on one motor.
 * Does @b not update the tracked joint angles, so re-home and call ::arm_init()
 * before resuming ::arm_move_to().
 *
 * @param joint   Joint index 0..2.
 * @param dir     Direction: >=0 forward, <0 reverse.
 * @param nsteps  Number of microsteps to issue (>0).
 * @param hz      Constant step rate (Hz, >0).
 */
void arm_test_spin(int joint, int dir, int32_t nsteps, float hz);

/**
 * @brief Read a TMC2240 DRV_STATUS register (diagnostics: stall, OT, etc.).
 * @param joint Joint index 0..2.
 * @return Raw 32-bit DRV_STATUS register value.
 */
uint32_t arm_drv_status(int joint);

/**
 * @brief Read a TMC2240 IOIN register (input pin states + version byte).
 * @param joint Joint index 0..2.
 * @return Raw 32-bit IOIN register value; the top byte is the silicon version.
 */
uint32_t arm_dbg_ioin(int joint);

/**
 * @brief Current end-effector position from the tracked joint angles (via FK).
 * @param[out] x Tip X (mm).
 * @param[out] y Tip Y (mm).
 * @param[out] z Tip Z (mm).
 */
void arm_get_xyz(float *x, float *y, float *z);

/** @brief Start the gripper servo PWM (TIM4 CH2) and open the gripper. */
void gripper_init(void);

/** @brief Command the gripper to its open position. */
void gripper_open(void);

/** @brief Command the gripper to its closed position. */
void gripper_close(void);

/**
 * @brief Set the gripper servo pulse width directly.
 * @param pulse_us Servo pulse width in microseconds, clamped to [500, 2500].
 */
void gripper_set_us(uint16_t pulse_us);

#endif /* ARM_CONTROL_H */