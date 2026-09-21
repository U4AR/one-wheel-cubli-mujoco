"""Linearisation of the MuJoCo model about the upright equilibrium, reduced
exactly as in Sec. 4 of the paper (drop phi, gamma, gamma_dot)."""
import numpy as np
import mujoco
from scipy.linalg import expm

from .model import load
from .params import CubliParams, NOMINAL

# reduced state (13): alpha, alpha_d, beta, beta_d, phi_d, d1, d1_d, d2, d2_d
STATE_NAMES = ["alpha", "alpha_d", "beta", "beta_d", "phi_d",
               "delta1", "delta1_d", "delta2", "delta2_d"]
# indices into full [q(6), qd(6)] with q order alpha,beta,gamma,phi,delta1,delta2
_RED = [0, 6, 1, 7, 9, 4, 10, 5, 11]


def continuous_full(p: CubliParams = NOMINAL, eps=1e-6):
    """Continuous-time A (12x12), B (12x1) of the full model at x=0, u=0."""
    p = p.with_(com_offset_xy=(0.0, 0.0), yaw_damping=0.0, yaw_friction=0.0)
    m, d = load(p)
    nq = m.nq

    def f(x, u):
        d.qpos[:] = x[:nq]
        d.qvel[:] = x[nq:]
        d.ctrl[:] = u
        mujoco.mj_forward(m, d)
        return np.concatenate([d.qvel.copy(), d.qacc.copy()])

    x0, u0 = np.zeros(2 * nq), np.zeros(1)
    A = np.zeros((2 * nq, 2 * nq))
    for i in range(2 * nq):
        dx = np.zeros(2 * nq); dx[i] = eps
        A[:, i] = (f(x0 + dx, u0) - f(x0 - dx, u0)) / (2 * eps)
    B = ((f(x0, u0 + eps) - f(x0, u0 - eps)) / (2 * eps))[:, None]
    return A, B


def continuous_reduced(p: CubliParams = NOMINAL):
    A, B = continuous_full(p)
    if A.shape[0] == 12:
        return A[np.ix_(_RED, _RED)], B[_RED]
    # rigid variant (ring): no bending joints. Keep the 9-state layout so the
    # estimator/controller code is unchanged; the bending states become stable,
    # decoupled placeholders that are never excited (their estimates stay 0).
    red = [0, 4, 1, 5, 7]                   # alpha, alpha_d, beta, beta_d, phi_d
    Ar = np.zeros((9, 9)); Br = np.zeros((9, 1))
    Ar[:5, :5] = A[np.ix_(red, red)]; Br[:5] = B[red]
    w, z = 2 * np.pi * 60.0, 0.05
    for i in (5, 7):
        Ar[i, i + 1] = 1.0
        Ar[i + 1, i], Ar[i + 1, i + 1] = -w * w, -2 * z * w
    return Ar, Br


def c2d(A, B, Ts):
    n, mm = B.shape
    M = np.zeros((n + mm, n + mm))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * Ts)
    return E[:n, :n], E[:n, n:]


def discrete_reduced(p: CubliParams = NOMINAL, Ts=0.01):
    return c2d(*continuous_reduced(p), Ts)
