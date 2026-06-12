#ifndef ARM_CONTROL_H
#define ARM_CONTROL_H

#include <stdint.h>

/* Call once after MX_*_Init() / homing. Sets current angles to your home pose. */
void arm_init(void);

/* Drive the tip (end-effector point, since L2 includes it) to (x,y,z) in mm.
 * elbow_up = 1 for the elbow-up solution, 0 for elbow-down.
 * Returns 1 if the target is reachable and motion was launched, 0 otherwise. */
int arm_move_to(float x, float y, float z, int elbow_up);
void arm_fk(float th0, float th1, float th2, float *x, float *y, float *z);

/* True while any joint is still stepping. Poll this to know when a move ends. */
int arm_is_moving(void);

uint8_t arm_drivers_init(void);   /* call after MX_SPI1_Init; returns bitmask of OK drivers */

void arm_test_spin(int joint, int dir, int32_t nsteps, float hz);

uint32_t arm_drv_status(int joint);

uint32_t arm_dbg_ioin(int joint);

#endif /* ARM_CONTROL_H */
