"""Yaw control on a realistic plant: tilted wheel + pivot dry friction + CoM offset
+ sensor noise/bias + delay.   python scripts/yaw_friction.py [tilt_deg ...]"""
import json, sys
sys.path.insert(0, ".")
import numpy as np
import mujoco

from cubli.controller import Controller
from cubli.model import load, IMU_POS
from cubli.params import NOMINAL
from cubli.sim import Motor, SensorModel
from cubli.tunings import get
from cubli.yaw import YawController

Ts, DT = 0.01, 5e-4


def simulate(p, T, heading_cmd=lambda t: 0.0, pushes=(), yaw_on=True, seed=0,
             yaw_kw=None, x0=(0.02, -0.015)):
    rng = np.random.default_rng(seed)
    sm = SensorModel()
    m, d = load(p, DT)
    ctrl = Controller(p.with_(com_offset_xy=(0.0, 0.0), wheel_ecc=0.0), get("tuned"), Ts)
    yaw = YawController(p, Ts, **(yaw_kw or {}))
    motor = Motor(p)
    d.qpos[0], d.qpos[1] = x0
    mujoco.mj_forward(m, d)
    acc_ids = [m.sensor(f"acc{i}").adr[0] for i in range(len(IMU_POS))]
    gyr_ids = [m.sensor(f"gyr{i}").adr[0] for i in range(len(IMU_POS))]
    ws_id = m.sensor("wheel_speed").adr[0]
    acc_b = rng.normal(0, sm.acc_bias_std, (5, 3)); gyr_b = rng.normal(0, sm.gyr_bias_std, 3)

    def meas():
        acc = np.array([d.sensordata[a:a + 3] for a in acc_ids]) + acc_b + rng.normal(0, sm.acc_std, (5, 3))
        gyr = np.array([d.sensordata[g:g + 3] for g in gyr_ids]) + gyr_b + rng.normal(0, sm.gyr_std, (5, 3))
        return acc, gyr, d.sensordata[ws_id] + rng.normal(0, sm.wheel_std)

    buf = [meas(), meas()]
    bodies = {b: m.body(b).id for _, b, _ in pushes}
    log = []
    for k in range(int(T / Ts)):
        t = d.time
        buf.append(meas()); buf.pop(0)
        acc, gyr, ws = buf[0]
        ref = yaw.step(gyr.mean(0)[2], ws, heading_cmd(t)) if yaw_on else None
        u, _ = motor.limit(ctrl.step(acc, gyr, ws, wheel_ref=ref), d.qvel[3], Ts)
        ctrl.applied(u)
        d.ctrl[0] = u
        for _ in range(int(Ts / DT)):
            d.xfrc_applied[:] = 0
            for t0, b, F in pushes:
                if t0 <= d.time < t0 + 0.05:
                    d.xfrc_applied[bodies[b], :3] = F
            mujoco.mj_step(m, d)
        log.append([d.time, d.qpos[0], d.qpos[1], d.qpos[2], d.qvel[2], d.qvel[3], u,
                    heading_cmd(t), yaw.gamma_hat])
        if max(abs(d.qpos[0]), abs(d.qpos[1])) > np.deg2rad(25):
            return np.array(log), True
    return np.array(log), False


def at(L, t):
    return L[min(int(t / Ts), len(L) - 1)]


def scenario_report(tilt_deg, seed=0):
    p = NOMINAL.with_(wheel_tilt=np.deg2rad(tilt_deg))
    out = dict(tilt_deg=tilt_deg)
    # A) heading steps: 0 -> 45 deg at t=5 s, back to 0 at t=70 s
    cmd = lambda t: np.deg2rad(45) if 5 <= t < 70 else 0.0
    L, fell = simulate(p, 130, cmd, seed=seed)
    out["steps"] = "fell" if fell else dict(
        heading_at_40s=round(float(np.rad2deg(at(L, 40)[3])), 1),
        heading_at_130s=round(float(np.rad2deg(L[-1, 3])), 1),
        peak_wheel=round(float(np.abs(L[:, 5]).max())),
        wheel_at_130s=round(float(L[-1, 5])))
    # B) sideways tap while holding heading 0
    L, fell = simulate(p, 90, pushes=[(5.0, "endmass2", (0, 1.5, 0))], seed=seed)
    out["tap"] = "fell" if fell else dict(
        peak_heading=round(float(np.rad2deg(np.abs(L[:, 3]).max())), 1),
        heading_at_90s=round(float(np.rad2deg(L[-1, 3])), 1),
        wheel_at_90s=round(float(L[-1, 5])))
    # C) long hold: heading drift from gyro bias (ZUPT during stiction)
    L, fell = simulate(p, 180, seed=seed)
    out["hold_180s"] = "fell" if fell else dict(true_heading_drift_deg=round(float(np.rad2deg(L[-1, 3])), 2))
    # D) balance margin with yaw control on
    lo, hi = 0.0, 6.0
    for _ in range(7):
        mid = (lo + hi) / 2
        _, f = simulate(p, 8, pushes=[(3.0, "endmass2", (0, 0, -mid))], seed=seed, x0=(0, 0))
        lo, hi = (mid, hi) if not f else (lo, mid)
    out["max_drop_N"] = round(lo, 2)
    return out


if __name__ == "__main__":
    tilts = [float(a) for a in sys.argv[1:]] or [10.0, 15.0]
    from multiprocessing import Pool
    with Pool(len(tilts)) as pool:
        res = pool.map(scenario_report, tilts)
    for r in res:
        print(r)
    json.dump(res, open("results/yaw_friction.json", "w"), indent=1)
