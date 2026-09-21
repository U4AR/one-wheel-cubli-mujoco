"""Step-by-step, interactive version of sim.run() for the live viewer.

Same plant, sensors, delay, motor envelope and controller as the batch simulation,
plus: timed force pulses, a mouse "grab" spring force, and offscreen rendering.
"""
import io
from dataclasses import replace
import numpy as np
import mujoco
from PIL import Image

from .model import load, IMU_POS
from .params import NOMINAL
from .controller import Controller
from .sim import SensorModel, Motor
from .tunings import get

N_IMU = len(IMU_POS)
FALL_DEG = 16.0   # just before an end mass / housing corner touches the ground


class LiveSim:
    def __init__(self, width=960, height=540, Ts=0.01, dt=5e-4):
        self.W, self.H, self.Ts, self.dt = width, height, Ts, dt
        self.plant = NOMINAL
        self.tuning_name = "tuned"
        self.com_enable = True
        self.yaw_on = False           # yaw loop through the wheel-speed setpoint
        self.heading_ref = 0.0        # rad
        self.noise_scale = 1.0
        self.delay_steps = 1
        self.controller_on = True
        self.renderer = None
        self.cam = mujoco.MjvCamera()
        self.reset_camera()
        self.vopt = mujoco.MjvOption()
        self.rng = np.random.default_rng()
        self.build()

    # ------------------------------------------------------------------ setup
    def reset_camera(self):
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0.0, 0.0, 0.17]
        self.cam.distance = 1.55
        self.cam.azimuth = 40.0
        self.cam.elevation = -16.0

    def build(self):
        """(Re)create model, controller and renderer from the current settings."""
        self.m, self.d = load(self.plant, self.dt, ground_limits=True)
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(self.m, self.H, self.W)
        self.jid = {n: self.m.joint(n).qposadr[0] for n in
                    ["alpha", "beta", "gamma", "phi", "delta1", "delta2"]}
        self.vid = {n: self.m.joint(n).dofadr[0] for n in self.jid}
        self.acc_ids = [self.m.sensor(f"acc{i}").adr[0] for i in range(N_IMU)]
        self.gyr_ids = [self.m.sensor(f"gyr{i}").adr[0] for i in range(N_IMU)]
        self.ws_id = self.m.sensor("wheel_speed").adr[0]
        self.body_names = {self.m.body(i).name: i for i in range(self.m.nbody)}
        self.reset()

    def make_controller(self):
        t = replace(get(self.tuning_name), com_enable=self.com_enable)
        self.ctrl = Controller(self.plant.with_(com_offset_xy=(0.0, 0.0), wheel_ecc=0.0), t, self.Ts)

    def reset(self, tilt_deg=(2.0, -1.5)):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.jid["alpha"]] = np.deg2rad(tilt_deg[0])
        self.d.qpos[self.jid["beta"]] = np.deg2rad(tilt_deg[1])
        mujoco.mj_forward(self.m, self.d)
        self.make_controller()
        self.motor = Motor(self.plant)
        s = self.sensors()
        self.acc_bias = self.rng.normal(0, s.acc_bias_std, (N_IMU, 3))
        self.gyr_bias = self.rng.normal(0, s.gyr_bias_std, 3)
        self.buf = [self.measure()] * (self.delay_steps + 1)
        self.pulses = []          # [t_end, body_id, force(3)]
        self.grab = None          # dict(body, local point, force)
        self.fell = False
        self.gamma_hat = 0.0          # heading from integrated gyro (starts at truth)
        self.w_ref = 0.0
        self.u = self.ulim = 0.0
        self.samples = []

    def sensors(self):
        k = self.noise_scale
        return replace(SensorModel(), acc_std=0.04 * k, gyr_std=0.004 * k,
                       wheel_std=0.3 * k, delay_steps=self.delay_steps)

    # --------------------------------------------------------------- dynamics
    def measure(self):
        s, d = self.sensors(), self.d
        acc = np.array([d.sensordata[a:a + 3] for a in self.acc_ids]) + self.acc_bias \
            + self.rng.normal(0, s.acc_std, (N_IMU, 3))
        gyr = np.array([d.sensordata[g:g + 3] for g in self.gyr_ids]) + self.gyr_bias \
            + self.rng.normal(0, s.gyr_std, (N_IMU, 3))
        ws = d.sensordata[self.ws_id] + self.rng.normal(0, s.wheel_std)
        return acc, gyr, ws

    def pulse(self, body, force, duration):
        self.pulses.append([self.d.time + duration, self.body_names[body], np.asarray(force, float)])

    def control_step(self):
        d = self.d
        self._update_grab_force()
        self.buf.append(self.measure()); self.buf.pop(0)
        if self.controller_on and not self.fell:
            acc, gyr, ws = self.buf[0]
            yaw_rate = gyr.mean(0)[2]
            self.gamma_hat += yaw_rate * self.Ts
            zeta = self.plant.wheel_tilt
            if self.yaw_on and abs(np.sin(zeta)) > 1e-3:
                # L_z = I_z*gamma_d + I_w*sin(zeta)*phi_d is conserved -> the
                # wheel-speed setpoint sets the yaw rate (see scripts/wheel_offset.py)
                p = self.plant
                k_yaw = (p.I_hz + 2 * p.m_e * p.l_E**2 + p.I_wy) / (p.I_wx * np.sin(zeta))
                err = (self.gamma_hat - self.heading_ref + np.pi) % (2 * np.pi) - np.pi
                target = float(np.clip(ws + k_yaw * (yaw_rate + 0.3 * err), -380, 380))
                self.w_ref += float(np.clip(target - self.w_ref, -20 * self.Ts, 20 * self.Ts))
            else:
                self.w_ref += float(np.clip(-self.w_ref, -10 * self.Ts, 10 * self.Ts))
            u_cmd = self.ctrl.step(acc, gyr, ws - self.w_ref)
            self.u, self.ulim = self.motor.limit(u_cmd, d.qvel[self.vid["phi"]], self.Ts)
            self.ctrl.applied(self.u)
        else:
            self.u, self.ulim = 0.0, 0.0
        d.ctrl[0] = self.u
        for _ in range(int(round(self.Ts / self.dt))):
            d.xfrc_applied[:] = 0
            self.pulses = [p for p in self.pulses if d.time < p[0]]
            for _, b, f in self.pulses:
                d.xfrc_applied[b, :3] += f
            if self.grab is not None:
                b, F = self.grab["body"], self.grab["force"]
                p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ self.grab["local"]
                d.xfrc_applied[b, :3] += F
                d.xfrc_applied[b, 3:] += np.cross(p - d.xipos[b], F)
            mujoco.mj_step(self.m, d)
        a, b = d.qpos[self.jid["alpha"]], d.qpos[self.jid["beta"]]
        if not self.fell and max(abs(a), abs(b)) > np.deg2rad(FALL_DEG):
            self.fell = True
        self.samples.append([round(d.time, 3), float(np.rad2deg(a)), float(np.rad2deg(b)),
                             float(self.u), float(self.ulim), float(d.qvel[self.vid["phi"]]),
                             float(100 * self.motor.i2t / self.plant.i2t_budget)])

    def pop_samples(self):
        s, self.samples = self.samples, []
        return s

    def state(self):
        d = self.d
        return dict(t=d.time, fell=self.fell, tuning=self.tuning_name,
                    com_enable=self.com_enable, noise=self.noise_scale,
                    yaw_on=self.yaw_on, heading_ref_deg=float(np.rad2deg(self.heading_ref)),
                    heading_est_deg=float(np.rad2deg(self.gamma_hat)), wheel_ref=self.w_ref,
                    delay=self.delay_steps, controller_on=self.controller_on,
                    com_est_deg=np.rad2deg(self.ctrl.com).tolist(),
                    gamma_deg=float(np.rad2deg(d.qpos[self.jid["gamma"]])),
                    beam_mrad=[1e3 * d.qpos[self.jid["delta1"]], 1e3 * d.qpos[self.jid["delta2"]]],
                    plant=dict(tilt_deg=float(np.rad2deg(self.plant.wheel_tilt)),
                               ecc_mm=1e3 * self.plant.wheel_ecc, m_e=self.plant.m_e, f_beam=self.plant.beam_freq_hz(),
                               com_offset_mm=1e3 * self.plant.com_offset_xy[0]))

    # ------------------------------------------------------------ interaction
    def _cam_axes(self):
        c = self.renderer.scene.camera[0]      # from the last update_scene
        fwd, up = np.array(c.forward, float), np.array(c.up, float)
        right = np.cross(fwd, up)
        return fwd, right / np.linalg.norm(right), up

    def grab_start(self, relx, rely):
        """relx, rely in [0,1] from the top-left of the image."""
        self.renderer.update_scene(self.d, camera=self.cam)
        selpnt = np.zeros(3); geomid = np.zeros(1, np.int32)
        flexid = np.zeros(1, np.int32); skinid = np.zeros(1, np.int32)
        body = mujoco.mjv_select(self.m, self.d, self.vopt, self.W / self.H, relx, 1 - rely,
                                 self.renderer.scene, selpnt, geomid, flexid, skinid)
        if body <= 0 or self.m.body(body).name == "target":
            self.grab = None
            return None
        R = self.d.xmat[body].reshape(3, 3)
        self.grab = dict(body=body, local=R.T @ (selpnt - self.d.xpos[body]),
                         anchor=selpnt.copy(), target=selpnt.copy(), force=np.zeros(3),
                         start=(relx, rely))
        return self.m.body(body).name

    def grab_move(self, relx, rely, stiffness=15.0, fmax=25.0):
        g = self.grab
        if g is None:
            return
        _, right, up = self._cam_axes()
        # screen displacement -> world displacement on the plane through the anchor
        scale = 2 * self.cam.distance * np.tan(np.deg2rad(self.m.vis.global_.fovy) / 2)
        dx, dy = (relx - g["start"][0]) * scale * self.W / self.H, (rely - g["start"][1]) * scale
        g["target"] = g["anchor"] + dx * right - dy * up
        g["k"], g["fmax"] = stiffness, fmax
        self._update_grab_force()

    def _update_grab_force(self):
        """Spring from the grabbed point to the mouse target (re-evaluated every
        control step, so holding the mouse still keeps pulling)."""
        g = self.grab
        if g is None or "k" not in g:
            return
        b = g["body"]
        p = self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ g["local"]
        F = g["k"] * (g["target"] - p)
        n = np.linalg.norm(F)
        g["force"] = F if n < g["fmax"] else F * g["fmax"] / n

    def grab_end(self):
        self.grab = None

    # -------------------------------------------------------------- rendering
    def _arrow(self, p_from, p_to, rgba, width=0.012):
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom or np.linalg.norm(p_to - p_from) < 1e-4:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                            np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width,
                             np.asarray(p_from, float), np.asarray(p_to, float))
        scn.ngeom += 1

    def render_jpeg(self, quality=80):
        d = self.d
        self.renderer.update_scene(d, camera=self.cam)
        for _, b, f in self.pulses:     # pulse forces: arrow pointing at the body
            p = d.xipos[b]
            self._arrow(p - 0.03 * f, p, (1.0, 0.85, 0.1, 1))
        if self.grab is not None:
            b = self.grab["body"]
            p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ self.grab["local"]
            self._arrow(p, p + 0.03 * self.grab["force"], (0.2, 0.9, 1.0, 1), 0.008)
        img = self.renderer.render()
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
