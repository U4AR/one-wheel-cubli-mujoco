"""Interactive rolling hoop for the live viewer (same interface as LiveSim)."""
import io
import types
import numpy as np
import mujoco
from PIL import Image

from .rolling import RollingParams, load, P
from .rolling_kane import DriveController, measure6
from .sim import Motor

_CTRL_CACHE = {}


def controller_for(rp):
    key = (rp.R, rp.M_hoop, rp.d, rp.eta, rp.wheel_tilt, rp.bearing_damping)
    if key not in _CTRL_CACHE:
        _CTRL_CACHE[key] = DriveController(rp)      # wheel (balance) + hub motor (speed)
    c = _CTRL_CACHE[key]
    c.reset(0.0)
    return c


class LiveRolling:
    FALL_DEG = 30.0

    def __init__(self, width=960, height=540, Ts=0.01, dt=5e-4, rp: RollingParams = RollingParams()):
        self.W, self.H, self.Ts, self.dt, self.rp = width, height, Ts, dt, rp
        self.r_eff = rp.R + 0.01
        # attributes the server / UI expect from LiveSim
        self.plant = types.SimpleNamespace(layout="rolling", ring_radius=rp.R, m_e=0.0,
                                           beam_freq_hz=lambda: 0.0, com_offset_xy=(0.0, 0.0),
                                           wheel_tilt=rp.wheel_tilt, wheel_ecc=0.0)
        self.tuning_name = "speed-scheduled LQR, 2 motors"
        self.com_enable, self.noise_scale, self.delay_steps = False, 1.0, 1
        self.controller_on, self.yaw_on, self.yaw, self.heading_ref = True, False, None, 0.0
        self.renderer = None
        self.cam = mujoco.MjvCamera()
        self.reset_camera()
        self.vopt = mujoco.MjvOption()
        self.rng = np.random.default_rng()
        self.build()

    # ------------------------------------------------------------------ setup
    def reset_camera(self):
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.4, 60.0, -18.0
        self.cam.lookat[:] = [0.0, 0.0, 0.3]

    def build(self):
        self.m, self.d = load(self.rp, self.dt)
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(self.m, self.H, self.W)
        self.ctrl = controller_for(self.rp)
        self.body_names = {self.m.body(i).name: i for i in range(self.m.nbody)}
        self.wheel_dof = self.m.joint("phi").dofadr[0]
        self.axle_dof = self.m.joint("axle").dofadr[0]
        self.reset()

    def make_controller(self):          # settings changes: nothing to rebuild here
        pass

    def set_yaw(self, on):
        self.yaw_on = False

    def reset(self, tilt_deg=(2.0, 0.0), v0=0.0):
        d, rp = self.d, self.rp
        mujoco.mj_resetData(self.m, d)
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([1.0, 0, 0]), np.deg2rad(tilt_deg[0]))
        d.qpos[3:7] = q
        d.qpos[2] = self.r_eff * np.cos(np.deg2rad(tilt_deg[0])) + 1e-4
        self.launch(v0, reset_pose=False)
        self.ctrl.reset(v0)
        self.motor = Motor(P)
        self.bias = np.array([self.rng.normal(0, np.deg2rad(0.1)), 0, 0, self.rng.normal(0, np.deg2rad(0.1)), 0, 0])
        self.buf = [self.measure()] * (self.delay_steps + 1)
        self.pulses, self.grab, self.fell = [], None, False
        self.u = self.ulim = self.u_hub = 0.0
        self.samples = []
        self.dist = 0.0
        self.last_xy = d.qpos[:2].copy()

    def set_speed(self, v):
        """Rolling-speed command (m/s); the hub motor tracks it, rate-limited."""
        self.ctrl.v_target = float(v)

    def launch(self, v, reset_pose=True):
        """Set the hoop rolling at v (m/s) along its current heading."""
        d = self.d
        hb = self.m.body("hoop").id
        mujoco.mj_forward(self.m, d)
        a = d.xmat[hb].reshape(3, 3)[:, 1]
        f = np.cross(a, [0, 0, 1.0]); f /= np.linalg.norm(f)
        d.qvel[0:3] = v * f
        w_world = a * (v / self.r_eff)
        d.qvel[3:6] = d.xmat[hb].reshape(3, 3).T @ w_world
        d.qvel[self.axle_dof] = -v / self.r_eff            # housing keeps hanging still
        mujoco.mj_forward(self.m, d)

    # --------------------------------------------------------------- dynamics
    def measure(self):
        x6, v = measure6(self.m, self.d, self.r_eff)
        x = np.append(x6, v / self.r_eff)
        k = self.noise_scale
        noise = np.array([np.deg2rad(0.03), 0.005, 0.005, np.deg2rad(0.03), 0.005, 0.3, 0.01]) * k
        return x + np.append(self.bias, 0.0) * k + self.rng.normal(0, 1, 7) * noise, v + self.rng.normal(0, 0.01 * k)

    def pulse(self, body, force, duration, point=None):
        self.pulses.append([self.d.time + duration, self.body_names[body], np.asarray(force, float),
                            None if point is None else np.asarray(point, float)])

    def heading_frame(self):
        hb = self.m.body("hoop").id
        a = self.d.xmat[hb].reshape(3, 3)[:, 1]
        f = np.cross(a, [0, 0, 1.0]); f /= np.linalg.norm(f)
        return f, np.cross([0, 0, 1.0], f)                 # forward, left (horizontal)

    def kick(self, preset, F, duration):
        """Viewer presets mapped onto the rolling hoop's heading frame."""
        f, left = self.heading_frame()
        if preset in ("push_px", "push_nx"):              # roll it forward / back
            s = 1.0 if preset == "push_px" else -1.0
            self.pulse("hoop", s * F * f * 3.0, max(duration, 0.1))
        elif preset in ("push_py", "drop_right"):
            self.pulse("housing", -F * left, duration)
        elif preset in ("push_ny", "drop_left"):
            self.pulse("housing", F * left, duration)
        elif preset == "twist":                            # sideways tap high on the rim
            self.pulse("hoop", F * left, duration, (0.0, 0.0, self.rp.R))

    def control_step(self):
        d = self.d
        self._update_grab_force()
        self.buf.append(self.measure()); self.buf.pop(0)
        y, vm = self.buf[0]
        if self.controller_on and not self.fell:
            u_cmd = self.ctrl.step(y, vm)
            self.u, self.ulim = self.motor.limit(float(u_cmd[0]), d.qvel[self.wheel_dof], self.Ts)
            self.u_hub = float(np.clip(u_cmd[1], -self.rp.hub_tau, self.rp.hub_tau))
            self.ctrl.applied([self.u, self.u_hub])
        else:
            self.u, self.ulim, self.u_hub = 0.0, 0.0, 0.0
        d.ctrl[0], d.ctrl[1] = self.u, self.u_hub
        for _ in range(int(round(self.Ts / self.dt))):
            d.xfrc_applied[:] = 0
            self.pulses = [p for p in self.pulses if d.time < p[0]]
            for _, b, f, pt in self.pulses:
                d.xfrc_applied[b, :3] += f
                if pt is not None:
                    pw = d.xpos[b] + d.xmat[b].reshape(3, 3) @ pt
                    d.xfrc_applied[b, 3:] += np.cross(pw - d.xipos[b], f)
            if self.grab is not None:
                b, F = self.grab["body"], self.grab["force"]
                p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ self.grab["local"]
                d.xfrc_applied[b, :3] += F
                d.xfrc_applied[b, 3:] += np.cross(p - d.xipos[b], F)
            mujoco.mj_step(self.m, d)
        x, v = measure6(self.m, d, self.r_eff)
        self.dist += float(np.linalg.norm(d.qpos[:2] - self.last_xy)); self.last_xy = d.qpos[:2].copy()
        if not self.fell and abs(x[0]) > np.deg2rad(self.FALL_DEG):
            self.fell = True
        self.v, self.x = v, x
        self.samples.append([round(d.time, 3), float(np.rad2deg(x[0])), float(np.rad2deg(x[3])),
                             float(self.u), float(self.ulim), float(x[5]),
                             float(100 * self.motor.i2t / P.i2t_budget)])

    def pop_samples(self):
        s, self.samples = self.samples, []
        return s

    def state(self):
        d = self.d
        x, v = measure6(self.m, d, self.r_eff)
        f, _ = self.heading_frame()
        return dict(t=d.time, fell=self.fell, tuning=self.tuning_name, com_enable=False,
                    noise=self.noise_scale, delay=self.delay_steps, controller_on=self.controller_on,
                    yaw_on=False, yaw_available=False, yaw_mode="off", heading_ref_deg=0.0,
                    heading_est_deg=0.0, wheel_ref=0.0, com_est_deg=[0.0, 0.0],
                    gamma_deg=float(np.rad2deg(np.arctan2(f[1], f[0]))), beam_mrad=[0.0, 0.0],
                    plant=dict(tilt_deg=float(np.rad2deg(self.rp.wheel_tilt)), ecc_mm=0.0, m_e=0.0,
                               f_beam=0.0, com_offset_mm=0.0),
                    rolling=dict(speed=float(v), distance=self.dist, yaw_rate=float(x[2]),
                                 pitch_deg=float(np.rad2deg(x[3])), v_target=float(self.ctrl.v_target),
                                 v_cmd=float(self.ctrl.v_cmd), hub_torque=float(self.u_hub)))

    # ------------------------------------------------------------ interaction
    def _cam_axes(self):
        c = self.renderer.scene.camera[0]
        fwd, up = np.array(c.forward, float), np.array(c.up, float)
        right = np.cross(fwd, up)
        return fwd, right / np.linalg.norm(right), up

    def grab_start(self, relx, rely):
        self.renderer.update_scene(self.d, camera=self.cam)
        selpnt = np.zeros(3); geomid = np.zeros(1, np.int32)
        flexid = np.zeros(1, np.int32); skinid = np.zeros(1, np.int32)
        body = mujoco.mjv_select(self.m, self.d, self.vopt, self.W / self.H, relx, 1 - rely,
                                 self.renderer.scene, selpnt, geomid, flexid, skinid)
        if body <= 0:
            self.grab = None
            return None
        R = self.d.xmat[body].reshape(3, 3)
        self.grab = dict(body=body, local=R.T @ (selpnt - self.d.xpos[body]), anchor=selpnt.copy(),
                         target=selpnt.copy(), force=np.zeros(3), start=(relx, rely))
        return self.m.body(body).name

    def grab_move(self, relx, rely, stiffness=15.0, fmax=25.0):
        g = self.grab
        if g is None:
            return
        _, right, up = self._cam_axes()
        scale = 2 * self.cam.distance * np.tan(np.deg2rad(self.m.vis.global_.fovy) / 2)
        dx, dy = (relx - g["start"][0]) * scale * self.W / self.H, (rely - g["start"][1]) * scale
        g["target"] = g["anchor"] + dx * right - dy * up
        g["k"], g["fmax"] = stiffness, fmax
        self._update_grab_force()

    def _update_grab_force(self):
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
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(p_from, float), np.asarray(p_to, float))
        scn.ngeom += 1

    def render_jpeg(self, quality=80):
        d = self.d
        hub = d.xpos[self.m.body("hoop").id]
        target = np.array([hub[0], hub[1], 0.3])
        if np.linalg.norm(target - self.cam.lookat) > 1.5:       # snap if far behind
            self.cam.lookat[:] = target
        self.cam.lookat[:] = 0.7 * np.asarray(self.cam.lookat) + 0.3 * target
        self.renderer.update_scene(d, camera=self.cam)
        for _, b, f, pt in self.pulses:
            p = d.xipos[b] if pt is None else d.xpos[b] + d.xmat[b].reshape(3, 3) @ pt
            self._arrow(p - 0.03 * f, p, (1.0, 0.85, 0.1, 1))
        if self.grab is not None:
            b = self.grab["body"]
            p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ self.grab["local"]
            self._arrow(p, p + 0.03 * self.grab["force"], (0.2, 0.9, 1.0, 1), 0.008)
        img = self.renderer.render()
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
