"""Hoop with the housing in the middle, plus a balancing pole (the paper's tube and
end masses) mounted at the hoop centre:
  none          -- hoop A alone (does not balance)
  rigid         -- pole fixed to the body (what the paper does, now on a hoop)
  gimbal        -- pole on a free 2-axis gimbal at its CoM: stays level by inertia
  gimbal+hang   -- pole CoM 5 cm below the gimbal: gravity keeps it level
  gimbal+motor  -- a second motor on the pole's pitch gimbal (two actuators)
Most generous test: full-state feedback (perfect sensing) with the paper's 10 ms
delay compensated, and the paper's motor limits.   python scripts/ring_pole.py
"""
import json, sys
sys.path.insert(0, ".")
import numpy as np
import mujoco
from multiprocessing import Pool

from cubli.model import build_xml
from cubli.params import NOMINAL
from cubli.controller import dlqr
from cubli.linearize import c2d
from cubli.sim import Motor

Ts, DT = 0.01, 5e-4
R_HOOP, M_HOOP = 0.3196, 0.7427


def config(name):
    lS = NOMINAL.housing_without_tube()[1]
    c = 1.05 * R_HOOP
    p = NOMINAL.with_(layout="ring", ring_mass=M_HOOP, ring_radius=R_HOOP, ring_center=c,
                      core_raise=c - lS, com_offset_xy=(0.0, 0.0))
    pole = {"none": "none", "rigid": "rigid"}.get(name, "gimbal")
    p = p.with_(pole=pole, pole_mount=c, pole_drop=0.05 if name == "gimbal+hang" else 0.0)
    xml = build_xml(p, DT)
    if name == "gimbal+motor":
        xml = xml.replace("</actuator>", '    <motor name="pole_motor" joint="pole_p" gear="1" ctrlrange="-3.4 3.4"/>\n  </actuator>')
    m = mujoco.MjModel.from_xml_string(xml)
    return p, m


def linearize(m, eps=1e-6):
    d = mujoco.MjData(m)
    nq, nu = m.nq, m.nu
    def f(x, u):
        d.qpos[:] = x[:nq]; d.qvel[:] = x[nq:]; d.ctrl[:] = u
        mujoco.mj_forward(m, d)
        return np.concatenate([d.qvel.copy(), d.qacc.copy()])
    x0, u0 = np.zeros(2 * nq), np.zeros(nu)
    A = np.column_stack([(f(x0 + e, u0) - f(x0 - e, u0)) / (2 * eps) for e in np.eye(2 * nq) * eps])
    B = np.column_stack([(f(x0, u0 + e) - f(x0, u0 - e)) / (2 * eps) for e in np.eye(nu) * eps])
    return A, B


def state_index(m):
    """keep alpha, beta, pole angles and all their rates + wheel rate; drop yaw and phi."""
    names = [m.joint(i).name for i in range(m.njnt)]
    q = {n: m.joint(n).qposadr[0] for n in names}
    v = {n: m.nq + m.joint(n).dofadr[0] for n in names}
    keep_q = [n for n in names if n not in ("gamma", "phi")]
    idx = [q[n] for n in keep_q] + [v[n] for n in names if n != "gamma"]
    scale = [np.deg2rad(1)] * 2 + [np.deg2rad(5)] * (len(keep_q) - 2) + \
            [np.deg2rad(10) if n != "phi" else 10.0 for n in names if n != "gamma"]
    return idx, np.array(scale)


def design(m):
    A, B = linearize(m)
    idx, S = state_index(m)
    A, B = A[np.ix_(idx, idx)], B[idx]
    unstable = sorted(np.real(np.linalg.eigvals(A))[np.real(np.linalg.eigvals(A)) > 1e-6], reverse=True)
    Ad, Bd = c2d(A, B, Ts)
    Sm = np.diag(S)
    Qw = np.ones(len(idx)); Qw[2:] = 0.1
    K = dlqr(np.linalg.solve(Sm, Ad @ Sm), np.linalg.solve(Sm, Bd), np.diag(Qw), 20.0 * np.eye(B.shape[1]))
    return Ad, Bd, K @ np.linalg.inv(Sm), idx, unstable


def simulate(name, T=15.0, tilt_deg=1.0, drop=0.0):
    p, m = config(name)
    Ad, Bd, K, idx, _ = design(m)
    d = mujoco.MjData(m)
    d.qpos[m.joint("alpha").qposadr[0]] = np.deg2rad(tilt_deg)
    d.qpos[m.joint("beta").qposadr[0]] = -np.deg2rad(tilt_deg)
    mujoco.mj_forward(m, d)
    motor = Motor(p)
    ring = m.body("ring").id
    x_meas = np.concatenate([d.qpos, d.qvel])[idx]
    u = np.zeros(m.nu)
    peak_u = 0.0
    for k in range(int(T / Ts)):
        xhat = Ad @ x_meas + Bd @ u                     # one-step delay compensation
        u = -(K @ xhat)
        u[0], _ = motor.limit(float(u[0]), d.qvel[m.joint("phi").dofadr[0]], Ts)
        if m.nu > 1:
            u[1] = float(np.clip(u[1], -3.4, 3.4))
        peak_u = max(peak_u, float(np.abs(u).max()))
        x_meas = np.concatenate([d.qpos, d.qvel])[idx]
        d.ctrl[:] = u
        for _ in range(int(Ts / DT)):
            d.xfrc_applied[:] = 0
            if drop and 2.0 <= d.time < 2.05:
                F = np.array([0, 0, -drop])
                pw = d.xpos[ring] + d.xmat[ring].reshape(3, 3) @ np.array([R_HOOP, 0, 0])
                d.xfrc_applied[ring, :3] = F
                d.xfrc_applied[ring, 3:] = np.cross(pw - d.xipos[ring], F)
            mujoco.mj_step(m, d)
        a, b = d.qpos[m.joint("alpha").qposadr[0]], d.qpos[m.joint("beta").qposadr[0]]
        if max(abs(a), abs(b)) > np.deg2rad(25) or not np.all(np.isfinite(d.qpos)):
            return False, peak_u
    return True, peak_u


def evaluate(name):
    p, m = config(name)
    try:
        _, _, K, _, unstable = design(m)
    except Exception as e:
        return name, dict(error=str(e))
    out = dict(unstable_rates=[round(float(v), 2) for v in unstable],
               gains_alpha_beta=[round(float(K[0, 0]), 1), round(float(K[0, 1]), 1)])
    ok_tilt = 0.0
    for t in [0.1, 0.25, 0.5, 1, 2, 4, 6]:
        ok, pk = simulate(name, tilt_deg=t)
        if not ok:
            break
        ok_tilt, out["peak_torque_at_that_tilt"] = t, round(pk, 2)
    out["recovers_from_tilt_deg"] = ok_tilt
    if ok_tilt >= 0.25:
        lo, hi = 0.0, 8.0
        for _ in range(8):
            mid = (lo + hi) / 2
            ok, _ = simulate(name, T=7.0, tilt_deg=0.0, drop=mid)
            lo, hi = (mid, hi) if ok else (lo, mid)
        out["max_drop_on_hoop_edge_N"] = round(lo, 2)
    return name, out


if __name__ == "__main__":
    names = ["none", "rigid", "gimbal", "gimbal+hang", "gimbal+motor"]
    with Pool(len(names)) as pool:
        res = dict(pool.map(evaluate, names))
    for n in names:
        print(f"{n:13s}", res[n])
    json.dump(res, open("results/ring_pole.json", "w"), indent=1)
