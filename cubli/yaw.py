"""Yaw control for a One-Wheel Cubli whose wheel axis is tilted out of plane
(not in the paper; see scripts/yaw_friction.py and the README).

The wheel's vertical torque component is  tau_z = -sin(zeta) * I_w * wheel_acc.
Vertical angular momentum is only changed by the ground (pivot friction), so:
  * to TURN, the wheel must push harder than the pivot's stiction, then brake;
    every second of turning burns tau_s / (I_w sin zeta) rad/s of wheel speed
    (~26 rad/s per second at 10 deg tilt), so big turns RATCHET:
  * UNLOAD: when the wheel budget is used, stop pushing (friction stops the body),
    then spin the wheel down gently enough that its reaction stays below stiction.
    The body stays put and the momentum goes into the ground. Then turn again.
  * HOLD: on target, stiction holds the heading for free; the wheel is bled to
    zero, and then the gyro bias is re-estimated because the body is known to be
    still. Heading itself comes from the integrated gyro, so errors picked up
    during manoeuvres stay (a few degrees) unless an absolute heading sensor
    (magnetometer, camera) is added.
Output: a wheel-speed trajectory (w_ref, a_ff) for Controller.step(..., wheel_ref=...),
or None while settling (plain paper balancing).
"""
import numpy as np


class YawController:
    # gains from a sweep over tap / 45-deg-step scenarios at 10 and 15 deg tilt;
    # w_turn is what matters: the balance loop tracks the wheel-speed target
    # loosely (overshoots ~150 rad/s), so turning must stop well short of 450.
    def __init__(self, p, Ts=0.01, tau_s=None, kp=0.1, kd=0.5, a_max=45.0,
                 w_turn=150.0, w_resume=40.0, deadband_deg=1.5, t_start=3.0):
        self.Ts = Ts
        self.s = max(np.sin(p.wheel_tilt), 1e-6)
        self.Iw = p.I_wx
        self.Iz = p.I_hz + 2 * p.m_e * p.l_E**2 + p.I_wy     # slow (whole-body) yaw inertia
        self.tau_s = p.yaw_friction if tau_s is None else tau_s  # friction the controller assumes
        self.kp, self.kd, self.a_max = kp, kd, a_max
        self.w_turn, self.w_resume = w_turn, w_resume
        self.deadband = np.deg2rad(deadband_deg)
        self.t_start = t_start
        self.reset()

    def reset(self, heading=0.0, wheel_speed=0.0):
        self.gamma_hat = heading
        self.w_ref = wheel_speed
        self.bias = 0.0
        self.r_f = 0.0
        self.t = 0.0
        self.mode = "hold"

    def step(self, gyro_z, wheel_speed, heading_ref):
        self.t += self.Ts
        r = gyro_z - self.bias
        self.gamma_hat += r * self.Ts
        self.r_f += (self.Ts / 0.3) * (r - self.r_f)       # filtered yaw rate
        err = (self.gamma_hat - heading_ref + np.pi) % (2 * np.pi) - np.pi
        moving = abs(self.r_f) > np.deg2rad(0.5)
        a_bleed = 0.3 * self.tau_s / (self.s * self.Iw)    # reaction well below stiction

        still = abs(r) < np.deg2rad(2) and not (abs(self.r_f) > np.deg2rad(0.5))
        if self.t < self.t_start:                         # plain paper balancing first
            self.mode = "settle"
            self.w_ref = wheel_speed
            return None
        if self.mode == "unload" and abs(self.w_ref) > self.w_resume:
            pass                                          # keep unloading
        elif abs(err) < self.deadband and not moving:
            self.mode = "hold"
        elif abs(self.w_ref) > self.w_turn:
            self.mode = "unload"
        else:
            self.mode = "turn"

        if self.mode in ("hold", "unload"):
            if moving:
                a = 0.0                                   # let friction stop the body
            else:
                a = -np.sign(self.w_ref) * min(a_bleed, abs(self.w_ref) / 0.5)
                # zero-velocity update: only when holding with the wheel nearly
                # unloaded -- then stiction really holds the body still (while
                # unloading, balance jitter + spin-down make it creep ~0.1 deg/s)
                if still and self.mode == "hold" and abs(self.w_ref) < 20:
                    self.bias += 0.005 * (gyro_z - self.bias)
        else:
            tau = self.Iz * (-self.kp * np.clip(err, -0.5, 0.5) - self.kd * self.r_f)
            if moving:
                tau += self.tau_s * np.sign(self.r_f)          # cancel sliding friction
            else:
                tau += 1.3 * self.tau_s * np.sign(tau)         # break away from stiction
            a = float(np.clip(-tau / (self.s * self.Iw), -self.a_max, self.a_max))
        self.w_ref = float(self.w_ref + a * self.Ts)
        return self.w_ref, float(a)
