"""Estimation and control stack of the One-Wheel Cubli (Secs. 5 and 6).

  IMUs --(Eq. 8 weights)--> body gravity g_hat --(complementary filter)--> alpha, beta
  gyros --(Eq. 16 mean)--> body rates -> alpha_d, beta_d
  [alpha, beta, alpha_d, beta_d, phi_d](k-1) --steady-state KF with 1-step delay
      augmentation (Eqs. 18-21)--> x_hat(k) --LQR (Eq. 22)--> T_m(k)
  CoM offset: low-pass filtered alpha, beta subtracted from the estimate (Sec. 6).
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import solve_discrete_are

from .linearize import discrete_reduced, continuous_reduced
from .model import IMU_POS
from .params import CubliParams, NOMINAL

NX = 9          # alpha, alpha_d, beta, beta_d, phi_d, d1, d1_d, d2, d2_d
NA = NX + 5     # + delayed alpha, alpha_d, beta, beta_d, phi_d

# Scaling used to normalise the state/input before weighting (the paper says the
# state and input are normalised but does not give the scales -- ASSUMED).
X_SCALE = np.array([np.deg2rad(1), np.deg2rad(10), np.deg2rad(1), np.deg2rad(10),
                    10.0, 1e-4, 3e-2, 1e-4, 3e-2])
U_SCALE = 1.0


def imu_weights(r=IMU_POS):
    """tau_i of Eq. (8): g_hat = -sum_i tau_i m_i."""
    rbar = r.mean(0)
    rc = r - rbar
    S = rc.T @ rc
    return 1.0 / len(r) - rc @ np.linalg.solve(S, rbar)


def augment(A, B):
    """Eq. (20): append one-step-delayed measured states."""
    sel = [0, 1, 2, 3, 4]
    At = np.zeros((NA, NA))
    At[:NX, :NX] = A
    At[NX:, :NX][np.arange(5), sel] = 1.0
    Bt = np.zeros((NA, 1)); Bt[:NX] = B
    # Eq. (19), our measurement order: alpha, beta, alpha_d, beta_d, phi_d
    C = np.zeros((5, NA))
    for row, col in enumerate([0, 2, 1, 3, 4]):
        C[row, NX + col] = 1.0
    return At, Bt, C


def dlqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def kalman_gain(A, C, V, W):
    """Steady-state (measurement-update form) gain L of Eq. (21)."""
    P = solve_discrete_are(A.T, C.T, V, W)   # prior covariance
    return P @ C.T @ np.linalg.inv(C @ P @ C.T + W)


# ---------------------------------------------------------------------------
# Paper tuning (Eqs. 23, 24). Q/R are in normalised units (see X_SCALE).
PAPER_Q = np.array([1, 1/10, 1, 1/10, 5, 1, 1, 1, 1, 1/10, 1/10, 1/10, 1/10, 1/10])
PAPER_R = 20.0
PAPER_V = np.array([1/50, 1/1000, 1/50, 1/500, 2, 1/500, 1/500, 1/500, 1/500,
                    1/100, 1/100, 1/100, 1/100, 1/100])
PAPER_W = np.array([1/1000, 1/1000, 1/20000, 1/20000, 1/1000])


@dataclass
class Tuning:
    Q: np.ndarray = field(default_factory=lambda: PAPER_Q.copy())
    R: float = PAPER_R
    V: np.ndarray = field(default_factory=lambda: PAPER_V.copy())
    W: np.ndarray = field(default_factory=lambda: PAPER_W.copy())
    comp_w_acc: float = 0.02     # complementary-filter accelerometer weight
    com_tau: float = 20.0        # s, CoM-offset low-pass time constant
    com_enable: bool = True


def design(p: CubliParams = NOMINAL, tuning: Tuning = None, Ts=0.01):
    """Returns (K, L, At, Bt, C) for the controller's (possibly wrong) model p."""
    tuning = tuning or Tuning()
    A, B = discrete_reduced(p, Ts)
    # normalise: x = S xn, u = s un
    S = np.diag(X_SCALE)
    An, Bn = np.linalg.solve(S, A @ S), np.linalg.solve(S, B * U_SCALE)
    At, Bt, C = augment(An, Bn)
    Kn = dlqr(At, Bt, np.diag(tuning.Q), np.array([[tuning.R]]))
    Ln = kalman_gain(At, C, np.diag(tuning.V), np.diag(tuning.W))
    # back to physical units
    Sa = np.diag(np.concatenate([X_SCALE, X_SCALE[:5]]))
    Cs = np.diag(X_SCALE[[0, 2, 1, 3, 4]])
    K = U_SCALE * Kn @ np.linalg.inv(Sa)
    L = Sa @ Ln @ np.linalg.inv(Cs)
    At_p, Bt_p, C_p = augment(A, B)
    return K, L, At_p, Bt_p, C_p


class Controller:
    """Runs at Ts. Call step(acc[5,3], gyr[5,3], wheel_speed) -> torque."""

    def __init__(self, p: CubliParams = NOMINAL, tuning: Tuning = None, Ts=0.01):
        self.p, self.tuning, self.Ts = p, tuning or Tuning(), Ts
        self.K, self.L, self.At, self.Bt, self.C = design(p, self.tuning, Ts)
        # IMU positions relative to the pivot (the housing core may be raised)
        self.tau = imu_weights(IMU_POS + np.array([0.0, 0.0, p.core_raise]))
        # steady state that makes the wheel accelerate at 1 rad/s^2 while the body
        # stays upright: solve alpha_dd = beta_dd = d1_dd = d2_dd = 0, phi_dd = 1 for
        # (alpha, beta, d1, d2, u). Used as feedforward when a wheel acceleration is
        # commanded (yaw control with a tilted wheel).
        Ac, Bc = continuous_reduced(p)
        rows, cols = [1, 3, 6, 8, 4], [0, 2, 5, 7]
        M = np.hstack([Ac[np.ix_(rows, cols)], Bc[rows]])
        sol = np.linalg.solve(M, np.array([0, 0, 0, 0, 1.0]))
        self.ss_x = np.zeros(NX); self.ss_x[cols] = sol[:4]
        self.ss_u = sol[4]
        self.reset()

    def reset(self, gyro_bias=None):
        self.xhat = np.zeros(NA)
        self.u_prev = 0.0
        self.g = None
        self.com = np.zeros(2)
        self.gyro_bias = np.zeros(3) if gyro_bias is None else gyro_bias

    def tilt(self, acc, gyr):
        g_acc = -(self.tau[:, None] * acc).sum(0)          # Eq. (8)
        g_acc /= np.linalg.norm(g_acc)
        w = gyr.mean(0) - self.gyro_bias                   # Eq. (16)
        if self.g is None:
            self.g = g_acc
        else:
            # propagate the gravity direction with the gyro: d/dt g_B = -w x g_B
            g_pred = self.g - self.Ts * np.cross(w, self.g)
            a = self.tuning.comp_w_acc
            self.g = (1 - a) * g_pred + a * g_acc
            self.g /= np.linalg.norm(self.g)
        gx, gy, gz = self.g            # unit vector; g_B = (sb ca, -sa, -cb ca)
        alpha = np.arcsin(np.clip(-gy, -1, 1))
        beta = np.arctan2(gx, -gz)
        alpha_d = w[0] / np.cos(beta)
        beta_d = w[1]
        return alpha, beta, alpha_d, beta_d

    def step(self, acc, gyr, wheel_speed, wheel_ref=None):
        """wheel_ref: optional (w_ref, a_ff) wheel-speed trajectory to follow, used
        for yaw control with a tilted wheel. The LQR regulates around the moving
        reference (its own wheel-speed feedback does the tracking) plus the
        feedforward lean/torque that makes the wheel accelerate at a_ff."""
        a, b, ad, bd = self.tilt(acc, gyr)
        xs, us = np.zeros(NX), 0.0
        if wheel_ref is not None:
            w_ref, a_ff = wheel_ref
            xs, us = self.ss_x * a_ff, self.ss_u * a_ff
            xs[4] = w_ref
        # CoM-offset estimate (Sec. 6): slow low-pass of the tilt angles
        # (minus any lean commanded on purpose)
        if self.tuning.com_enable:
            k = self.Ts / self.tuning.com_tau
            self.com += k * (np.array([a - xs[0], b - xs[2]]) - self.com)
        z = np.array([a - self.com[0], b - self.com[1], ad, bd, wheel_speed])
        # Eq. (21): predict with previous input, correct with delayed measurement
        xp = self.At @ self.xhat + self.Bt[:, 0] * self.u_prev
        self.xhat = xp + self.L @ (z - self.C @ xp)
        xref = np.concatenate([xs, xs[:5]])
        u = float(us - (self.K @ (self.xhat - xref))[0])
        self.u_prev = u
        return u

    def applied(self, u):
        """Report the torque actually applied (after saturation / derating),
        which the motor controller knows; used by the next KF prediction."""
        self.u_prev = u
