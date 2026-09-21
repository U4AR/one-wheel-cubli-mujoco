"""Interactive free-body One-Wheel Cubli (floor contacts) for the live viewer:
settle / somersault / jump-up attempt / stand-and-balance, same interface as LiveSim."""
import types
import numpy as np
import mujoco

from .live_rolling import LiveRolling
from .somersault import load, place, tilt_angles, balance_gain
from .sim import Motor
from .params import NOMINAL as P

PHASES = ["idle", "spin", "throw", "catch", "balance"]


class LiveSomersault(LiveRolling):
    def __init__(self, width=960, height=540, Ts=0.01, dt=5e-4):
        self.W, self.H, self.Ts, self.dt = width, height, Ts, dt
        self.plant = types.SimpleNamespace(layout="free", ring_radius=0.0, m_e=P.m_e, beam_freq_hz=lambda: 0.0,
                                           com_offset_xy=(0.0, 0.0), wheel_tilt=0.0, wheel_ecc=0.0)
        self.tuning_name = "free body: spin-up / throw / LQR catch"
        self.com_enable, self.noise_scale, self.delay_steps = False, 1.0, 1
        self.controller_on, self.yaw_on, self.yaw, self.heading_ref = True, False, None, 0.0
        self.renderer = None
        self.cam = mujoco.MjvCamera()
        self.reset_camera()
        self.vopt = mujoco.MjvOption()
        self.rng = np.random.default_rng()
        self.K = balance_gain()
        self.build()

    def reset_camera(self):
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.2, 70.0, -14.0
        self.cam.lookat[:] = [0.0, 0.0, 0.15]

    def build(self):
        self.m, self.d = load(P, self.dt)
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(self.m, self.H, self.W)
        self.body_names = {self.m.body(i).name: i for i in range(self.m.nbody)}
        self.wheel_dof = self.m.joint("phi").dofadr[0]
        self.settle()

    # ------------------------------------------------------------ manoeuvres
    def _clear(self, phase="idle"):
        self.motor = Motor(P)
        self.pulses, self.grab, self.fell = [], None, False
        self.u = self.ulim = 0.0
        self.samples = []
        self.phase = phase
        self.roll = 0.0
        self.message = ""
        self.move = getattr(self, "move", None) if phase in ("spin", "throw") else None

    def settle(self):
        """Drop it and let it come to rest (it rolls over and ends upside down)."""
        place(self.m, self.d, 0.05, 0.0)
        for _ in range(8000):
            mujoco.mj_step(self.m, self.d)
        self._clear("idle")
        self.message = "resting upside down (where a fall ends)"

    def stand(self, tilt_deg=2.0):
        """Put it on its tip, slightly tilted, and balance with the paper's LQR."""
        mujoco.mj_resetData(self.m, self.d)
        place(self.m, self.d, np.deg2rad(tilt_deg), 0.0)
        self._clear("balance")
        self.message = "balancing on the tip (paper LQR)"

    def reset(self, tilt_deg=(2.0, 0.0), v0=0.0):
        self.stand(tilt_deg[0])

    def start(self, name, w_spin=420.0, w_rev=None):
        """name: 'somersault' (full roll) or 'jump' (flip up and try to catch)."""
        if self.phase in ("spin", "throw"):
            return
        self.settle()                               # always start from the same rest pose
        self._clear("spin")
        self.move = name
        self.w_spin = w_spin
        self.w_rev = (420.0 if name == "somersault" else 110.0) if w_rev is None else w_rev
        self.message = ("somersault: spinning the wheel up (pressed into the floor)" if name == "somersault"
                        else "jump-up attempt: spinning the wheel up")

    # --------------------------------------------------------------- dynamics
    def state9(self):
        a, b, w, R = tilt_angles(self.m, self.d)
        return np.array([a, w[0] / max(np.cos(b), 0.2), b, w[1], self.d.qvel[self.wheel_dof], 0, 0, 0, 0])

    def kick(self, preset, F, duration):
        dirs = {"push_px": (1, 0, 0), "push_nx": (-1, 0, 0), "push_py": (0, 1, 0), "push_ny": (0, -1, 0),
                "drop_left": (0, 0, -1), "drop_right": (0, 0, -1), "twist": (0, 1, 0)}
        if preset in ("drop_left", "drop_right"):
            sx = 1.0 if preset == "drop_right" else -1.0
            self.pulse("cubli", np.array([0, 0, -F]), duration, (sx * P.l_E, 0.0, P.l_Q))
        else:
            self.pulse("cubli", np.array(dirs.get(preset, (0, 1, 0)), float) * F, duration)

    def set_speed(self, v):
        pass

    def control_step(self):
        d = self.d
        self._update_grab_force()
        a, b, w, R = tilt_angles(self.m, d)
        wd = d.qvel[self.wheel_dof]
        u = 0.0
        if self.controller_on:
            if self.phase == "spin":
                u = 0.6
                if wd >= self.w_spin:
                    self.phase = "throw"
                    self.message = f"{self.move}: throw! wheel driven hard to -{self.w_rev:.0f} rad/s"
            elif self.phase == "throw":
                u = -3.4 if wd > -self.w_rev else 0.0
                if (self.move == "jump" and R[2, 2] > np.cos(np.deg2rad(20))
                        and abs(np.rad2deg(self.roll)) > 140):
                    self.phase = "catch"
                    self.message = "jump-up: near upright -> LQR catch"
                elif u == 0.0 and np.linalg.norm(d.qvel[3:6]) < 0.05 and abs(np.rad2deg(self.roll)) > 20:
                    turns = abs(np.rad2deg(self.roll)) / 360
                    self.phase = "idle"
                    self.message = (f"done: rolled {abs(np.rad2deg(self.roll)):.0f} deg "
                                    f"({'full somersault' if turns > 0.9 else 'fell back'}), resting "
                                    f"{'upside down' if R[2, 2] < -0.5 else 'on its side'}")
            elif self.phase in ("catch", "balance"):
                u = float(-(self.K @ self.state9()))
                if R[2, 2] < 0.8:
                    self.phase = "idle"
                    self.fell = True
                    self.message = ("jump-up: catch failed (outside the ~3-5 deg catch region)"
                                    if self.move == "jump" else "fell over")
            u, self.ulim = self.motor.limit(u, wd, self.Ts)
        self.u = u
        d.ctrl[0] = u
        for _ in range(int(round(self.Ts / self.dt))):
            d.xfrc_applied[:] = 0
            self.pulses = [p for p in self.pulses if d.time < p[0]]
            for _, bb, f, pt in self.pulses:
                d.xfrc_applied[bb, :3] += f
                if pt is not None:
                    pw = d.xpos[bb] + d.xmat[bb].reshape(3, 3) @ pt
                    d.xfrc_applied[bb, 3:] += np.cross(pw - d.xipos[bb], f)
            if self.grab is not None:
                bb, F = self.grab["body"], self.grab["force"]
                p = d.xpos[bb] + d.xmat[bb].reshape(3, 3) @ self.grab["local"]
                d.xfrc_applied[bb, :3] += F
                d.xfrc_applied[bb, 3:] += np.cross(p - d.xipos[bb], F)
            mujoco.mj_step(self.m, d)
            self.roll += d.qvel[3] * self.dt
        a, b, w, R = tilt_angles(self.m, d)
        self.samples.append([round(d.time, 3), float(np.rad2deg(a)), float(np.rad2deg(b)), float(self.u),
                             float(self.ulim), float(d.qvel[self.wheel_dof]),
                             float(100 * self.motor.i2t / P.i2t_budget)])

    def state(self):
        d = self.d
        a, b, w, R = tilt_angles(self.m, d)
        return dict(t=d.time, fell=False, tuning=self.tuning_name, com_enable=False, noise=1.0, delay=1,
                    controller_on=self.controller_on, yaw_on=False, yaw_available=False, yaw_mode="off",
                    heading_ref_deg=0.0, heading_est_deg=0.0, wheel_ref=0.0, com_est_deg=[0.0, 0.0],
                    gamma_deg=0.0, beam_mrad=[0.0, 0.0],
                    plant=dict(tilt_deg=0.0, ecc_mm=0.0, m_e=P.m_e, f_beam=0.0, com_offset_mm=0.0),
                    free=dict(phase=self.phase, roll_deg=float(np.rad2deg(self.roll)),
                              tilt_from_vertical_deg=float(np.rad2deg(np.arccos(np.clip(R[2, 2], -1, 1)))),
                              body_rate=float(np.linalg.norm(w)), message=self.message))

    def render_jpeg(self, quality=80):
        com = self.d.xipos[self.m.body("cubli").id]
        target = np.array([com[0], com[1], 0.15])
        if np.linalg.norm(target - self.cam.lookat) > 1.0:
            self.cam.lookat[:] = target
        self.cam.lookat[:] = 0.8 * np.asarray(self.cam.lookat) + 0.2 * target
        # LiveRolling.render_jpeg re-targets on the hoop; do the rendering here
        import io
        from PIL import Image
        self.renderer.update_scene(self.d, camera=self.cam)
        for _, bb, f, pt in self.pulses:
            p = self.d.xipos[bb] if pt is None else self.d.xpos[bb] + self.d.xmat[bb].reshape(3, 3) @ pt
            self._arrow(p - 0.03 * f, p, (1.0, 0.85, 0.1, 1))
        if self.grab is not None:
            bb = self.grab["body"]
            p = self.d.xpos[bb] + self.d.xmat[bb].reshape(3, 3) @ self.grab["local"]
            self._arrow(p, p + 0.03 * self.grab["force"], (0.2, 0.9, 1.0, 1), 0.008)
        img = self.renderer.render()
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
