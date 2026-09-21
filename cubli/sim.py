"""Closed-loop simulation: MuJoCo plant + sensor noise/delay + motor limits +
the estimation/control stack of the paper."""
from dataclasses import dataclass, field
import numpy as np
import mujoco

from .model import load, IMU_POS
from .params import CubliParams, NOMINAL
from .controller import Controller, Tuning

N_IMU = len(IMU_POS)


@dataclass
class SensorModel:
    """MPU-6050-class noise levels at 100 Hz (ASSUMED, datasheet ballpark)."""
    acc_std: float = 0.04          # m/s^2 white noise per axis
    acc_bias_std: float = 0.03     # m/s^2 residual bias after calibration
    gyr_std: float = 0.004         # rad/s white noise per axis
    gyr_bias_std: float = 0.002    # rad/s residual bias after start-up averaging
    wheel_std: float = 0.3         # rad/s Hall-sensor speed noise
    delay_steps: int = 1           # measurement delay in control periods (~10 ms)


class Motor:
    """Torque-speed envelope + I^2t thermal derating of the EPOS4 (Sec. 3.2)."""

    def __init__(self, p: CubliParams):
        self.p = p
        self.i2t = 0.0

    def limit(self, u, w, dt):
        p = self.p
        lim = p.tau_peak
        if u * w > 0:  # motoring: back-EMF reduces the available torque
            lim = min(lim, p.tau_stall * max(0.0, 1 - abs(w) / p.w_noload))
        if self.i2t >= p.i2t_budget:
            lim = min(lim, p.tau_cont)
        u = float(np.clip(u, -lim, lim))
        self.i2t = max(0.0, self.i2t + (u * u - p.tau_cont**2) * dt)
        return u, lim


@dataclass
class Disturbance:
    """Force (N, world frame) on a body for [t0, t0+dur), applied at the body's
    centre of mass or at `point` (body frame, m)."""
    t0: float
    dur: float
    body: str
    force: tuple
    point: tuple = None


@dataclass
class SimResult:
    t: np.ndarray
    x: np.ndarray          # true reduced state (alpha, alpha_d, beta, beta_d, phi_d, d1, d1_d, d2, d2_d)
    xhat: np.ndarray
    u: np.ndarray
    ulim: np.ndarray
    i2t: np.ndarray
    gamma: np.ndarray
    fell: bool
    t_fall: float


def run(T=10.0, plant: CubliParams = NOMINAL, model_p: CubliParams = None,
        tuning: Tuning = None, sensors: SensorModel = None, x0=(0.0, 0.0),
        disturbances=(), seed=0, Ts=0.01, dt=5e-4, fall_deg=25.0, frames=None,
        frame_every=None, renderer=None, camera=None):
    """x0 = initial (alpha, beta) in rad. model_p = parameters assumed by the
    controller (defaults to the CAD model: plant params without CoM offset)."""
    rng = np.random.default_rng(seed)
    sensors = sensors or SensorModel()
    model_p = model_p or plant.with_(com_offset_xy=(0.0, 0.0))
    m, d = load(plant, dt)
    ctrl = Controller(model_p, tuning, Ts)
    motor = Motor(plant)

    names = [n for n in ["alpha", "beta", "gamma", "phi", "delta1", "delta2"]
             if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0]
    jid = {n: m.joint(n).qposadr[0] for n in names}
    vid = {n: m.joint(n).dofadr[0] for n in names}
    qp = lambda n: d.qpos[jid[n]] if n in jid else 0.0   # ring variant: no bending joints
    qv = lambda n: d.qvel[vid[n]] if n in vid else 0.0
    d.qpos[jid["alpha"]], d.qpos[jid["beta"]] = x0
    mujoco.mj_forward(m, d)

    acc_bias = rng.normal(0, sensors.acc_bias_std, (N_IMU, 3))
    gyr_bias = rng.normal(0, sensors.gyr_bias_std, 3)
    acc_ids = [m.sensor(f"acc{i}").adr[0] for i in range(N_IMU)]
    gyr_ids = [m.sensor(f"gyr{i}").adr[0] for i in range(N_IMU)]
    ws_id = m.sensor("wheel_speed").adr[0]
    body_ids = {b.body: m.body(b.body).id for b in disturbances}

    def measure():
        acc = np.array([d.sensordata[a:a+3] for a in acc_ids]) + acc_bias \
            + rng.normal(0, sensors.acc_std, (N_IMU, 3))
        gyr = np.array([d.sensordata[g:g+3] for g in gyr_ids]) + gyr_bias \
            + rng.normal(0, sensors.gyr_std, (N_IMU, 3))
        ws = d.sensordata[ws_id] + rng.normal(0, sensors.wheel_std)
        return acc, gyr, ws

    def true_state():
        return np.array([d.qpos[jid["alpha"]], d.qvel[vid["alpha"]],
                         d.qpos[jid["beta"]], d.qvel[vid["beta"]],
                         d.qvel[vid["phi"]],
                         qp("delta1"), qv("delta1"), qp("delta2"), qv("delta2")])

    n_ctrl = int(round(T / Ts))
    sub = int(round(Ts / dt))
    buf = [measure()] * (sensors.delay_steps + 1)
    log = {k: [] for k in ["t", "x", "xhat", "u", "ulim", "i2t", "gamma"]}
    fell, t_fall = False, np.nan
    u = 0.0

    for k in range(n_ctrl):
        t = d.time
        buf.append(measure()); buf.pop(0)
        u_cmd = ctrl.step(*buf[0])                      # delayed measurement
        u, lim = motor.limit(u_cmd, d.qvel[vid["phi"]], Ts)
        ctrl.applied(u)
        d.ctrl[0] = u

        log["t"].append(t); log["x"].append(true_state()); log["xhat"].append(ctrl.xhat[:9].copy())
        log["u"].append(u); log["ulim"].append(lim); log["i2t"].append(motor.i2t)
        log["gamma"].append(d.qpos[jid["gamma"]])

        for _ in range(sub):
            d.xfrc_applied[:] = 0
            for db in disturbances:
                if db.t0 <= d.time < db.t0 + db.dur:
                    b = body_ids[db.body]
                    d.xfrc_applied[b, :3] += db.force
                    if db.point is not None:
                        pw = d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.asarray(db.point)
                        d.xfrc_applied[b, 3:] += np.cross(pw - d.xipos[b], db.force)
            mujoco.mj_step(m, d)
            if frames is not None and renderer is not None and \
                    int(round(d.time / dt)) % frame_every == 0:
                renderer.update_scene(d, camera=camera)
                frames.append(renderer.render())

        if max(abs(d.qpos[jid["alpha"]]), abs(d.qpos[jid["beta"]])) > np.deg2rad(fall_deg) \
                or not np.all(np.isfinite(d.qpos)):
            fell, t_fall = True, d.time
            break

    arr = {k: np.array(v) for k, v in log.items()}
    return SimResult(arr["t"], arr["x"], arr["xhat"], arr["u"], arr["ulim"],
                     arr["i2t"], arr["gamma"], fell, t_fall)
