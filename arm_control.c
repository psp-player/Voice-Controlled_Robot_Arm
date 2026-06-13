/**
 * @file    arm_control.c
 * @brief   3-DoF (RRR) arm: inverse/forward kinematics, coordinated stepper
 *          motion with an acceleration ramp, and TMC2240 driver control.
 *
 * @details
 * Motion is generated entirely in interrupt context. Each joint owns a hardware
 * timer (TIM1/2/3) running in PWM mode on CH1; every update event emits one
 * microstep and decrements that joint's remaining-step counter in
 * ::HAL_TIM_PeriodElapsedCallback(). For a coordinated move the joint with the
 * most steps is the "master": its progress drives a single trapezoidal velocity
 * @f$ v(s) @f$ (constant-acceleration ramp up, constant-deceleration ramp down),
 * and every joint's timer reload is rescaled so all joints finish together.
 *
 * The three TMC2240 drivers are configured over a shared SPI bus with per-joint
 * chip selects; current limits and chopper mode are tuned per joint to match the
 * differing torque demands of the base, shoulder, and elbow.
 */

#include "arm_control.h"
#include "main.h"
#include <math.h>

extern TIM_HandleTypeDef htim1, htim2, htim3;
extern SPI_HandleTypeDef hspi1;

/** @name Link geometry (mm)
 *  Denavit-style link lengths for the RRR arm.
 *  @{ */
#define D1   145.0f   /**< Base height: shoulder pivot above the base frame. */
#define L1   170.0f   /**< Upper-arm length (shoulder to elbow). */
#define L2   245.0f   /**< Forearm length (elbow to tip; includes end-effector). */
/** @} */

/** @name Stepping resolution
 *  @{ */
#define STEPS_PER_REV  200.0f   /**< Full steps per motor revolution (1.8 deg). */
#define MICROSTEP      16.0f    /**< Microsteps per full step (TMC2240 MRES). */
#define GEAR           1.0f     /**< Gear ratio (1.0 = direct drive). */
/** @brief Microsteps per radian of joint rotation. */
#define STEPS_PER_RAD  (STEPS_PER_REV * MICROSTEP * GEAR / (2.0f * (float)M_PI))
/** @} */

/** @name Timing / motion profile
 *  @{ */
#define TIM_TICK_HZ    1000000.0f   /**< Timer tick rate feeding the joint timers (1 MHz). */
#define MAX_STEP_HZ    800.0f       /**< Cruise (max) step rate after ramp-up. */
#define ARR_MAX        0xFFFFu      /**< Max auto-reload value (16-bit timer). */
#define V_START_HZ     400.0f       /**< Ramp start/stop rate, kept below stall. */
#define ACCEL_HZ2      1000.0f      /**< Acceleration (microsteps/s^2) for the ramp. */
/** @} */

/** @name TMC2240 SPI register addresses
 *  @{ */
#define TMC_GCONF        0x00
#define TMC_GSTAT        0x01
#define TMC_IOIN         0x04
#define TMC_DRV_CONF     0x0A
#define TMC_GLOBALSCALER 0x0B
#define TMC_IHOLD_IRUN   0x10
#define TMC_TPOWERDOWN   0x11
#define TMC_TPWMTHRS     0x13
#define TMC_CHOPCONF     0x6C
#define TMC_DRV_STATUS   0x6F
#define TMC_WRITE        0x80   /**< OR into the register address to write. */
#define TMC_SPI_TIMEOUT  10     /**< SPI transfer timeout (ms). */
/** @} */

/** @brief Per-joint chip-select GPIO ports. */
static GPIO_TypeDef *const cs_port[3] = { M1_CS_GPIO_Port, M2_CS_GPIO_Port, M3_CS_GPIO_Port };
/** @brief Per-joint chip-select GPIO pins. */
static const uint16_t      cs_pin[3]  = { M1_CS_Pin,       M2_CS_Pin,       M3_CS_Pin };
/** @brief Per-joint driver-enable GPIO ports (active low). */
static GPIO_TypeDef *const en_port[3] = { M1_EN_GPIO_Port, M2_EN_GPIO_Port, M3_EN_GPIO_Port };
/** @brief Per-joint driver-enable GPIO pins. */
static const uint16_t      en_pin[3]  = { M1_EN_Pin,       M2_EN_Pin,       M3_EN_Pin };

/** @name Per-joint TMC2240 current and chopper configuration
 *  Tuned individually: the shoulder carries the most load and is driven hardest.
 *  @{ */
static const uint8_t  drv_range[3]    = { 2, 2, 1 };       /**< DRV_CONF current range per joint. */
static const uint8_t  drv_irun[3]     = { 16, 31, 20 };    /**< Run current (shoulder maxed). */
static const uint8_t  drv_ihold[3]    = { 8, 20, 8 };      /**< Hold current per joint. */
static const uint8_t  drv_stealth[3]  = { 0, 0, 0 };       /**< 0 = SpreadCycle on all joints. */
static const uint32_t drv_tpwmthrs[3] = { 0, 0, 0 };       /**< StealthChop switch threshold (unused). */
/** @} */

/**
 * @brief Write a 32-bit value to a TMC2240 register over SPI.
 * @param i    Joint/driver index 0..2.
 * @param reg  Register address (write bit is added internally).
 * @param val  32-bit value to write (MSB first).
 */
static void tmc_write(int i, uint8_t reg, uint32_t val)
{
    uint8_t tx[5] = { reg | TMC_WRITE, val >> 24, val >> 16, val >> 8, val };
    HAL_GPIO_WritePin(cs_port[i], cs_pin[i], GPIO_PIN_RESET);
    HAL_SPI_Transmit(&hspi1, tx, 5, TMC_SPI_TIMEOUT);
    HAL_GPIO_WritePin(cs_port[i], cs_pin[i], GPIO_PIN_SET);
}

/**
 * @brief Read a 32-bit TMC2240 register over SPI.
 *
 * The TMC2240 returns the requested register's contents on the @e following
 * transfer, so this issues the address twice and returns the second response.
 *
 * @param i    Joint/driver index 0..2.
 * @param reg  Register address to read.
 * @return 32-bit register value (MSB first in the response).
 */
static uint32_t tmc_read(int i, uint8_t reg)
{
    uint8_t tx[5] = { reg & 0x7F, 0, 0, 0, 0 };
    uint8_t rx[5];
    HAL_GPIO_WritePin(cs_port[i], cs_pin[i], GPIO_PIN_RESET);
    HAL_SPI_TransmitReceive(&hspi1, tx, rx, 5, TMC_SPI_TIMEOUT);
    HAL_GPIO_WritePin(cs_port[i], cs_pin[i], GPIO_PIN_SET);
    HAL_GPIO_WritePin(cs_port[i], cs_pin[i], GPIO_PIN_RESET);
    HAL_SPI_TransmitReceive(&hspi1, tx, rx, 5, TMC_SPI_TIMEOUT);
    HAL_GPIO_WritePin(cs_port[i], cs_pin[i], GPIO_PIN_SET);
    return ((uint32_t)rx[1] << 24) | ((uint32_t)rx[2] << 16)
         | ((uint32_t)rx[3] << 8)  |  (uint32_t)rx[4];
}

/**
 * @brief Configure and enable all three TMC2240 drivers.
 * @copydoc arm_drivers_init
 */
uint8_t arm_drivers_init(void)
{
    uint8_t ok = 0;
    const uint32_t chopconf = 0x14010043u;  /* SpreadCycle, MRES=4 (16 ustep), INTPOL */

    for (int i = 0; i < 3; i++) {
        HAL_GPIO_WritePin(en_port[i], en_pin[i], GPIO_PIN_SET);   /* disable while configuring */

        tmc_write(i, TMC_GSTAT, 0x07);
        tmc_write(i, TMC_GCONF, drv_stealth[i] ? 0x00000004u : 0x00000000u);  /* bit2 = en_pwm_mode */
        tmc_write(i, TMC_TPWMTHRS, drv_tpwmthrs[i]);
        tmc_write(i, TMC_DRV_CONF, drv_range[i]);
        tmc_write(i, TMC_GLOBALSCALER, 0x00000000);   /* 0 = full scale */
        tmc_write(i, TMC_IHOLD_IRUN,
                  (uint32_t)drv_ihold[i] | ((uint32_t)drv_irun[i] << 8) | (6u << 16));
        tmc_write(i, TMC_TPOWERDOWN, 10);
        tmc_write(i, TMC_CHOPCONF, chopconf);

        uint8_t ver = (uint8_t)(tmc_read(i, TMC_IOIN) >> 24);
        if (ver == 0x40) {
            ok |= (1u << i);
            HAL_GPIO_WritePin(en_port[i], en_pin[i], GPIO_PIN_RESET);  /* enable */
        }
    }
    return ok;
}

/** @brief Per-joint step-generation timers (one PWM channel each). */
static TIM_HandleTypeDef *const joint_tim[3] = { &htim1, &htim2, &htim3 };
/** @brief Per-joint direction GPIO ports. */
static GPIO_TypeDef *const dir_port[3] = { M1_DIR_GPIO_Port, M2_DIR_GPIO_Port, M3_DIR_GPIO_Port };
/** @brief Per-joint direction GPIO pins. */
static const uint16_t      dir_pin[3]  = { M1_DIR_Pin,       M2_DIR_Pin,       M3_DIR_Pin };
/** @brief Direction level corresponding to positive joint rotation (per joint wiring). */
static const GPIO_PinState dir_pos[3]  = { GPIO_PIN_RESET, GPIO_PIN_RESET, GPIO_PIN_SET };

/** @brief Steps still owed to each joint; decremented in the timer ISR. */
static volatile int32_t steps_remaining[3] = {0, 0, 0};
/** @brief Tracked joint angles (rad), updated as moves are committed. */
static float cur_ang[3];

/** @name Motion-profile state (shared between launch and ISR)
 *  @{ */
static volatile int32_t mv_total[3];   /**< Total steps for the current move, per joint. */
static volatile int32_t mv_nmax;       /**< Master joint's total step count. */
static volatile int     mv_master;     /**< Index of the master (longest) joint. */
static volatile uint8_t mv_ramp;       /**< 1 = apply velocity profile, 0 = constant rate. */
/** @} */

extern TIM_HandleTypeDef htim1, htim2, htim3, htim4;

/** @name Gripper servo (TIM4 CH2)
 *  @{ */
#define SERVO_OPEN_US    2000U     /**< Pulse width for "open" (tune to gripper). */
#define SERVO_CLOSE_US   1000U     /**< Pulse width for "closed" (tune to gripper). */
/** @} */

/**
 * @brief Start the gripper servo PWM and command the open position.
 */
void gripper_init(void)
{
    HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_2);
    __HAL_TIM_SET_COMPARE(&htim4, TIM_CHANNEL_2, SERVO_OPEN_US);  /* start open */
}

/**
 * @brief Set the gripper servo pulse width, clamped to a safe range.
 * @copydetails gripper_set_us
 */
void gripper_set_us(uint16_t pulse_us)
{
    if (pulse_us < 500U)  pulse_us = 500U;
    if (pulse_us > 2500U) pulse_us = 2500U;
    __HAL_TIM_SET_COMPARE(&htim4, TIM_CHANNEL_2, pulse_us);
}

/** @brief Open the gripper. */
void gripper_open(void)  { gripper_set_us(SERVO_OPEN_US);  }
/** @brief Close the gripper. */
void gripper_close(void) { gripper_set_us(SERVO_CLOSE_US); }

/**
 * @brief Set a joint timer's step rate by reprogramming its auto-reload.
 *
 * Converts a desired step frequency to a timer ARR (from the 1 MHz tick) and
 * keeps the PWM duty near 50%. Clamps to >=1 Hz and the 16-bit ARR ceiling.
 *
 * @param i   Joint index 0..2.
 * @param hz  Desired step rate (Hz).
 */
static inline void joint_set_rate(int i, float hz)
{
    if (hz < 1.0f) hz = 1.0f;
    uint32_t arr = (uint32_t)(TIM_TICK_HZ / hz) - 1u;
    if (arr > ARR_MAX) arr = ARR_MAX;
    __HAL_TIM_SET_AUTORELOAD(joint_tim[i], arr);
    __HAL_TIM_SET_COMPARE(joint_tim[i], TIM_CHANNEL_1, (arr + 1u) / 2u);
}

/**
 * @brief Compute the master joint's instantaneous step rate along the profile.
 *
 * Implements a symmetric trapezoidal profile from a constant-acceleration model:
 * the accel branch grows with distance travelled, the decel branch grows with
 * distance remaining, and the lower of the two (capped at ::MAX_STEP_HZ) gives a
 * smooth ramp up, optional cruise, and ramp down.
 *
 * @return Master-joint step rate (Hz) for the current position in the move.
 */
static float master_velocity(void)
{
    int32_t rem  = steps_remaining[mv_master];
    int32_t done = mv_nmax - rem;
    float v_acc = sqrtf(V_START_HZ * V_START_HZ + 2.0f * ACCEL_HZ2 * (float)done);
    float v_dec = sqrtf(V_START_HZ * V_START_HZ + 2.0f * ACCEL_HZ2 * (float)rem);
    float v = (v_acc < v_dec) ? v_acc : v_dec;
    if (v > MAX_STEP_HZ) v = MAX_STEP_HZ;
    return v;
}

/**
 * @brief Set tracked joint angles to the mechanical home pose.
 * @copydoc arm_init
 */
void arm_init(void)
{
    cur_ang[0] = 0.0f;
    cur_ang[1] = (float)M_PI / 2.0f;
    cur_ang[2] = -(float)M_PI / 2.0f;
}

/**
 * @brief Inverse kinematics for the RRR arm.
 *
 * Closed-form solution: base yaw from the planar projection, then the
 * shoulder/elbow angles in the arm plane via the law of cosines. The @p elbow_up
 * flag selects between the two valid elbow configurations.
 *
 * @param x         Target X (mm).
 * @param y         Target Y (mm).
 * @param z         Target Z (mm).
 * @param elbow_up  1 = elbow-up solution, 0 = elbow-down.
 * @param[out] th   Joint angles {base, shoulder, elbow} (rad).
 * @return 1 if reachable, 0 if the target lies outside the workspace.
 */
static int arm_ik(float x, float y, float z, int elbow_up, float th[3])
{
    th[0] = atan2f(y, x);
    float r  = sqrtf(x * x + y * y);
    float zp = z - D1;
    float c2 = r * r + zp * zp;
    float D = (c2 - L1 * L1 - L2 * L2) / (2.0f * L1 * L2);
    if (D < -1.0f || D > 1.0f) return 0;   /* out of reach */
    float s = sqrtf(1.0f - D * D);
    if (elbow_up) s = -s;
    th[2] = atan2f(s, D);
    th[1] = atan2f(zp, r) - atan2f(L2 * s, L1 + L2 * D);
    return 1;
}

/**
 * @brief Launch a coordinated Cartesian move (see public ::arm_move_to).
 * @copydoc arm_move_to
 */
int arm_move_to(float x, float y, float z, int elbow_up)
{
    if (arm_is_moving()) return 0;

    float th[3];
    if (!arm_ik(x, y, z, elbow_up, th)) return 0;

    int32_t steps[3];
    int     pos[3];
    int32_t nmax = 0;
    int     master = 0;

    /* Per joint: signed angular delta -> direction + microstep count.
       Track the longest joint; it becomes the profile master. */
    for (int i = 0; i < 3; i++) {
        float d  = th[i] - cur_ang[i];
        pos[i]   = (d >= 0.0f);
        steps[i] = (int32_t)lroundf(fabsf(d) * STEPS_PER_RAD);
        if (steps[i] > nmax) { nmax = steps[i]; master = i; }
    }
    if (nmax == 0) return 1;   /* already there */

    mv_nmax   = nmax;
    mv_master = master;
    mv_ramp   = 1;

    for (int i = 0; i < 3; i++) {
        /* Set DIR for this joint, honouring its wiring polarity. */
        HAL_GPIO_WritePin(dir_port[i], dir_pin[i],
                          pos[i] ? dir_pos[i]
                                 : (dir_pos[i] == GPIO_PIN_SET ? GPIO_PIN_RESET : GPIO_PIN_SET));

        mv_total[i]        = steps[i];
        steps_remaining[i] = steps[i];
        /* Commit the target angle now; motion follows in the ISR. */
        cur_ang[i] += (pos[i] ? 1.0f : -1.0f) * (float)steps[i] / STEPS_PER_RAD;

        if (steps[i] == 0) continue;

        /* Seed each joint's start rate proportional to its share of the move. */
        joint_set_rate(i, V_START_HZ * (float)steps[i] / (float)nmax);

        TIM_HandleTypeDef *h = joint_tim[i];
        __HAL_TIM_SET_COUNTER(h, 0);
        HAL_TIM_GenerateEvent(h, TIM_EVENTSOURCE_UPDATE);
        __HAL_TIM_CLEAR_FLAG(h, TIM_FLAG_UPDATE);
        HAL_TIM_PWM_Start(h, TIM_CHANNEL_1);
        __HAL_TIM_ENABLE_IT(h, TIM_IT_UPDATE);
    }
    return 1;
}

/**
 * @brief Constant-rate single-joint spin for diagnostics.
 *
 * Bypasses IK and the velocity profile and does @b not update ::cur_ang, so the
 * tracked pose is invalid afterwards; re-home and call ::arm_init() before the
 * next ::arm_move_to().
 *
 * @copydetails arm_test_spin
 */
void arm_test_spin(int joint, int dir, int32_t nsteps, float hz)
{
    if (joint < 0 || joint > 2 || nsteps <= 0 || hz <= 0.0f) return;
    if (arm_is_moving()) return;
    mv_ramp = 0;   /* constant rate, no profile */

    GPIO_PinState lvl = (dir >= 0) ? dir_pos[joint]
                      : (dir_pos[joint] == GPIO_PIN_SET ? GPIO_PIN_RESET : GPIO_PIN_SET);
    HAL_GPIO_WritePin(dir_port[joint], dir_pin[joint], lvl);

    joint_set_rate(joint, hz);

    TIM_HandleTypeDef *h = joint_tim[joint];
    steps_remaining[joint] = nsteps;
    __HAL_TIM_SET_COUNTER(h, 0);
    HAL_TIM_GenerateEvent(h, TIM_EVENTSOURCE_UPDATE);
    __HAL_TIM_CLEAR_FLAG(h, TIM_FLAG_UPDATE);
    HAL_TIM_PWM_Start(h, TIM_CHANNEL_1);
    __HAL_TIM_ENABLE_IT(h, TIM_IT_UPDATE);
}

/**
 * @brief Forward kinematics for the RRR arm.
 * @copydoc arm_fk
 */
void arm_fk(float th0, float th1, float th2, float *x, float *y, float *z)
{
    float r = L1 * cosf(th1) + L2 * cosf(th1 + th2);
    *x = r * cosf(th0);
    *y = r * sinf(th0);
    *z = D1 + L1 * sinf(th1) + L2 * sinf(th1 + th2);
}

/** @brief Read a driver's IOIN register. @copydoc arm_dbg_ioin */
uint32_t arm_dbg_ioin(int joint)   { return tmc_read(joint, TMC_IOIN); }
/** @brief Read a driver's DRV_STATUS register. @copydoc arm_drv_status */
uint32_t arm_drv_status(int joint) { return tmc_read(joint, TMC_DRV_STATUS); }

/**
 * @brief True while any joint still has steps pending.
 * @copydoc arm_is_moving
 */
int arm_is_moving(void)
{
    return (steps_remaining[0] | steps_remaining[1] | steps_remaining[2]) != 0;
}

/**
 * @brief Current tip position from tracked angles via FK.
 * @copydoc arm_get_xyz
 */
void arm_get_xyz(float *x, float *y, float *z)
{
    arm_fk(cur_ang[0], cur_ang[1], cur_ang[2], x, y, z);
}

/**
 * @brief Timer update ISR: emit one microstep and advance the velocity profile.
 *
 * Called by HAL on every joint-timer update event. Identifies which joint fired,
 * decrements its remaining-step counter, and stops that joint's PWM when it
 * reaches zero. When the @e master joint fires mid-move, recomputes the profile
 * velocity and rescales every active joint's rate so they stay synchronised.
 *
 * @param htim Handle of the timer that generated the interrupt.
 */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    int fired = -1;
    for (int i = 0; i < 3; i++)
        if (htim->Instance == joint_tim[i]->Instance) { fired = i; break; }
    if (fired < 0) return;

    if (steps_remaining[fired] > 0) {
        if (--steps_remaining[fired] == 0) {
            HAL_TIM_PWM_Stop(htim, TIM_CHANNEL_1);
            __HAL_TIM_DISABLE_IT(htim, TIM_IT_UPDATE);
        }
    }

    /* On each master step, refresh the profile and rescale all active joints. */
    if (mv_ramp && fired == mv_master && steps_remaining[mv_master] > 0) {
        float v = master_velocity();
        for (int j = 0; j < 3; j++)
            if (mv_total[j] > 0 && steps_remaining[j] > 0)
                joint_set_rate(j, v * (float)mv_total[j] / (float)mv_nmax);
    }
}