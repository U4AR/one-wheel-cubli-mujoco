"""What if the flywheel is offset?

(1) Tilted wheel axis: the spin axis is tilted up out of the horizontal plane by
    zeta, so the motor torque gets a vertical component and yaw becomes actuated.
    Vertical angular momentum L_z (about the pivot) is still conserved, so yaw spin
    can only be moved INTO the wheel: the wheel ends at L_z / (I_w sin zeta).
(2) Eccentric wheel: the wheel's centre of mass is off its axis (unbalance).

The cascade runs the full estimator/controller with sensor noise and delay; the
naive joint LQR uses delayed full-state feedback. No pivot friction in these tests.
    python scripts/wheel_offset.py
"""
import json, sys
sys.path.insert(0, ".")
import numpy as np
import mujoco

from cubli.linearize import continuous_full, c2d
from cubli.controller import dlqr
from cubli.model import load
from cubli.params import NOMINAL
from cubli.sim import Motor, run
from cubli.tunings import get

Ts, DT = 0.01, 5e-4
# state: alpha, alpha_d, beta, beta_d, gamma, gamma_d, d1, d1_d, d2, d2_d
IDX = [0, 6, 1, 7, 2, 8, 4, 10, 5, 11]
SCALE = np.array([np.deg2rad(1), np.deg2rad(10), np.deg2rad(1), np.deg2rad(10),
                  np.deg2rad(5), np.deg2rad(10), 1e-4, 3e-2, 1e-4, 3e-2])
QW = np.array([1, 0.1, 1, 0.1, 0.3, 1.0, 1, 1, 1, 1])


def design(p):
    A, B = continuous_full(p)
    A, B = A[np.ix_(IDX, IDX)], B[IDX]
    Ad, Bd = c2d(A, B, Ts)
    S = np.diag(SCALE)
    K = dlqr(np.linalg.solve(S, Ad @ S), np.linalg.solve(S, Bd), np.diag(QW), np.array([[20.0]]))
    return Ad, Bd, K @ np.linalg.inv(S)


def pbh_yaw_controllable(p):
    """Is the yaw mode controllable? (PBH test at the double eigenvalue 0)"""
    A, B = continuous_full(p)
    A, B = A[np.ix_(IDX, IDX)], B[IDX]
    return np.linalg.matrix_rank(np.hstack([A, B]), tol=1e-6) == len(IDX)


def simulate(p, Ad, Bd, K, T=15.0, x0=(0.035, -0.026), gamma0=0.0, yaw_rate0=0.0,
             push=None):
    m, d = load(p.with_(com_offset_xy=(0.0, 0.0), yaw_damping=0.0, yaw_friction=0.0), DT)
    motor = Motor(p)
    d.qpos[0], d.qpos[1], d.qpos[2] = x0[0], x0[1], gamma0
    d.qvel[2] = yaw_rate0
    mujoco.mj_forward(m, d)
    st = lambda: np.array([d.qpos[0], d.qvel[0], d.qpos[1], d.qvel[1], d.qpos[2], d.qvel[2],
                           d.qpos[4], d.qvel[4], d.qpos[5], d.qvel[5]])
    x_meas, u = st(), 0.0
    log = []
    bid = m.body("endmass2").id
    for k in range(int(T / Ts)):
        xhat = Ad @ x_meas + Bd[:, 0] * u          # delay compensation
        u, _ = motor.limit(float(-(K @ xhat)[0]), d.qvel[3], Ts)
        x_meas = st()                              # becomes available next step
        d.ctrl[0] = u
        for _ in range(int(Ts / DT)):
            d.xfrc_applied[:] = 0
            if push and push[0] <= d.time < push[0] + 0.05:
                d.xfrc_applied[bid, :3] = push[1]
            mujoco.mj_step(m, d)
        log.append([d.time, d.qpos[0], d.qpos[1], d.qpos[2], d.qvel[2], d.qvel[3], u])
        if max(abs(d.qpos[0]), abs(d.qpos[1])) > np.deg2rad(25):
            return np.array(log), True
    return np.array(log), False


def max_drop(p, Ad, Bd, K):
    lo, hi = 0.0, 6.0
    for _ in range(8):
        mid = (lo + hi) / 2
        _, fell = simulate(p, Ad, Bd, K, T=6.0, x0=(0, 0), push=(1.0, (0, 0, -mid)))
        lo, hi = (mid, hi) if not fell else (lo, mid)
    return lo


# ---------------------------------------------------------------------------
# Cascade: paper's balance controller (inner, fast) + yaw loop (outer, slow).
# L_z = I_z*gamma_d + I_w*sin(zeta)*phi_d is conserved, so the wheel-speed setpoint
# decides the yaw rate:  phi_d* = phi_d + I_z/(I_w sin zeta) * (gamma_d + lam*(gamma - gamma_ref))
from cubli.controller import Controller
from cubli.model import IMU_POS
from cubli.sim import SensorModel


def yaw_inertia(p):
    return p.I_hz + 2 * p.m_e * p.l_E**2 + p.I_wy


def simulate_cascade(p, lam=0.3, w_lim=300.0, w_rate=40.0, T=30.0, x0=(0.035, -0.026), gamma0=0.0,
                     yaw_rate0=0.0, push=None, yaw_loop=True, yaw_friction=0.0, seed=0):
    rng = np.random.default_rng(seed)
    sm = SensorModel()
    plant = p.with_(yaw_damping=0.0, yaw_friction=yaw_friction)
    m, d = load(plant, DT)
    ctrl = Controller(p.with_(com_offset_xy=(0.0, 0.0)), get("tuned"), Ts)
    motor = Motor(p)
    d.qpos[0], d.qpos[1], d.qpos[2] = x0[0], x0[1], gamma0
    d.qvel[2] = yaw_rate0
    mujoco.mj_forward(m, d)
    acc_ids = [m.sensor(f"acc{i}").adr[0] for i in range(len(IMU_POS))]
    gyr_ids = [m.sensor(f"gyr{i}").adr[0] for i in range(len(IMU_POS))]
    ws_id = m.sensor("wheel_speed").adr[0]
    acc_b = rng.normal(0, sm.acc_bias_std, (5, 3)); gyr_b = rng.normal(0, sm.gyr_bias_std, 3)

    def meas():
        acc = np.array([d.sensordata[a:a + 3] for a in acc_ids]) + acc_b + rng.normal(0, sm.acc_std, (5, 3))
        gyr = np.array([d.sensordata[g:g + 3] for g in gyr_ids]) + gyr_b + rng.normal(0, sm.gyr_std, (5, 3))
        return acc, gyr, d.sensordata[ws_id] + rng.normal(0, sm.wheel_std)

    k_yaw = yaw_inertia(p) / (p.I_wx * max(np.sin(p.wheel_tilt), 1e-9))
    buf = [meas(), meas()]
    gamma_hat, w_ref = gamma0, 0.0   # initial heading assumed known (compass / mocap)
    bid = m.body("endmass2").id
    log = []
    for k in range(int(T / Ts)):
        buf.append(meas()); buf.pop(0)
        acc, gyr, ws = buf[0]
        yaw_rate = gyr.mean(0)[2]              # body z ~ vertical when balanced
        gamma_hat += yaw_rate * Ts             # heading from the gyro (drifts slowly)
        if yaw_loop:
            target = float(np.clip(ws + k_yaw * (yaw_rate + lam * gamma_hat), -w_lim, w_lim))
            # rate-limit the setpoint so the balance loop is never asked for a step
            w_ref += float(np.clip(target - w_ref, -w_rate * Ts, w_rate * Ts))
        u_cmd = ctrl.step(acc, gyr, ws - w_ref)
        u, _ = motor.limit(u_cmd, d.qvel[3], Ts)
        ctrl.applied(u)
        d.ctrl[0] = u
        for _ in range(int(Ts / DT)):
            d.xfrc_applied[:] = 0
            if push and push[0] <= d.time < push[0] + 0.05:
                d.xfrc_applied[bid, :3] = push[1]
            mujoco.mj_step(m, d)
        log.append([d.time, d.qpos[0], d.qpos[1], d.qpos[2], d.qvel[2], d.qvel[3], u, w_ref])
        if max(abs(d.qpos[0]), abs(d.qpos[1])) > np.deg2rad(25):
            return np.array(log), True
    return np.array(log), False


def max_drop_cascade(p, yaw_loop=True):
    lo, hi = 0.0, 6.0
    for _ in range(8):
        mid = (lo + hi) / 2
        _, fell = simulate_cascade(p, T=7.0, x0=(0, 0), push=(2.0, (0, 0, -mid)), yaw_loop=yaw_loop)
        lo, hi = (mid, hi) if not fell else (lo, mid)
    return lo


def tilted_wheel():
    """(a) naive: one LQR over roll, pitch AND yaw -> balance gains collapse, falls.
    (b) cascade: paper's balance loop + slow yaw loop through the wheel-speed setpoint."""
    rows = []
    for zdeg in [0, 5, 10, 15, 20, 25]:
        p = NOMINAL.with_(wheel_tilt=np.deg2rad(zdeg))
        row = dict(tilt_deg=zdeg, yaw_controllable=bool(pbh_yaw_controllable(p)))
        if zdeg > 0:
            Ad, Bd, K = design(p)
            _, fell = simulate(p, Ad, Bd, K, T=20.0)
            row["naive_joint_lqr_balances_20s"] = not fell
            L, fell = simulate_cascade(p, T=40.0, yaw_rate0=0.2, w_rate=10.0)
            row["cascade_stop_0.2rad_s_spin"] = "fell" if fell else dict(
                final_yaw_rate=round(float(L[-1, 4]), 3), final_wheel=round(float(L[-1, 5])),
                predicted_wheel=round(yaw_inertia(p) * 0.2 / (p.I_wx * np.sin(p.wheel_tilt))))
            L, fell = simulate_cascade(p, T=40.0, gamma0=np.deg2rad(30), w_rate=10.0)
            row["cascade_heading_30deg"] = "fell" if fell else dict(
                final_heading_deg=round(float(np.rad2deg(L[-1, 3])), 2),
                peak_wheel=round(float(np.abs(L[:, 5]).max())))
        row["max_drop_N"] = round(max_drop_cascade(p, yaw_loop=zdeg > 0), 2)
        rows.append(row)
        print(row, flush=True)
    return rows


def eccentric_wheel():
    rows = []
    for ecc_mm in [0.0, 0.05, 0.2, 0.5, 1.0]:
        p = NOMINAL.with_(wheel_ecc=ecc_mm / 1e3)
        r = run(T=20.0, plant=p, model_p=NOMINAL.with_(com_offset_xy=(0, 0)),
                tuning=get("tuned"), x0=(0.035, -0.026), seed=0)
        sel = r.t > 5
        w = r.x[:, 4]
        row = dict(ecc_mm=ecc_mm, unbalance_gmm=0.228 * ecc_mm * 1e3, fell=bool(r.fell))
        if not r.fell:
            row |= dict(std_alpha_deg=float(np.rad2deg(r.x[sel, 0].std())),
                        std_beta_deg=float(np.rad2deg(r.x[sel, 2].std())),
                        rms_torque=float(np.sqrt((r.u[sel] ** 2).mean())),
                        mean_abs_wheel=float(np.abs(w[sel]).mean()),
                        peak_unbalance_force_N=float(0.228 * ecc_mm / 1e3 * np.abs(w).max() ** 2))
        rows.append(row)
        print(row, flush=True)
    return rows


if __name__ == "__main__":
    out = dict(tilted=tilted_wheel(), eccentric=eccentric_wheel())
    json.dump(out, open("results/wheel_offset.json", "w"), indent=1)
