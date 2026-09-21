"""Interactive Pogo Cubli (one motor balances and winds/fires the pogo spring) for
the live viewer: stick / hop / stop / somersault, same interface as LiveSim."""
import io
import types

import numpy as np
import mujoco

from .live_rolling import LiveRolling
from .pogo import PogoSim, PogoParams
from .params import NOMINAL as P


class LivePogo(LiveRolling):
    def __init__(self, width=960, height=540, Ts=0.01, dt=5e-4):
        self.W, self.H, self.Ts, self.dt = width, height, Ts, dt
        self.plant = types.SimpleNamespace(layout="pogo", ring_radius=0.0, m_e=P.m_e, beam_freq_hz=lambda: 0.0,
                                           com_offset_xy=(0.0, 0.0), wheel_tilt=0.0, wheel_ecc=0.0)
        self.tuning_name = "pogo: stance LQR + flight law + flip guidance (external sensors)"
        self.com_enable, self.noise_scale, self.delay_steps = False, 1.0, 1
        self.controller_on, self.yaw_on, self.yaw, self.heading_ref = True, False, None, 0.0
        self.renderer = None
        self.cam = mujoco.MjvCamera()
        self.reset_camera()
        self.vopt = mujoco.MjvOption()
        self.rng = np.random.default_rng()
        self.build()

    def reset_camera(self):
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.4, 15.0, -10.0
        self.cam.lookat[:] = [0.0, 0.0, 0.25]

    def build(self):
        self.pogo = PogoSim(PogoParams(), dt=self.dt, Ts=self.Ts, seed=int(self.rng.integers(1 << 30)))
        self.pogo.substep_hook = self._apply_external
        self.m, self.d = self.pogo.m, self.pogo.d
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(self.m, self.H, self.W)
        self.body_names = {self.m.body(i).name: i for i in range(self.m.nbody)}
        self.reset()

    def reset(self, tilt_deg=(1.0, 0.0), v0=0.0):
        self.pogo.reset((float(tilt_deg[0]), float(tilt_deg[1])))
        self.pulses, self.grab, self.samples = [], None, []
        self.fell, self.u, self.ulim = False, 0.0, 0.0
        self.message = "standing as a stick (clutch parked below its speed)"

    # ------------------------------------------------------------ commands
    def start(self, name, **_):
        if self.fell:
            self.reset()
        self.pogo.set_mode(name)
        self.message = dict(hop="hopping: wheel above the clutch speed winds the spring, the cam fires it",
                            stop="stopping: finishing the current jump, then back to a stick",
                            stick="stick: wheel parked below the clutch speed",
                            flip="somersault: winding with a slow wheel, then a 360 deg flip in the air"
                            ).get(name, name)

    def set_speed(self, v):            # "hop speed" slider: wheel speed while winding
        self.pogo.w_hop = float(np.clip(v, 30.0, 120.0))

    def kick(self, preset, F, duration):
        dirs = {"push_px": (1, 0, 0), "push_nx": (-1, 0, 0), "push_py": (0, 1, 0), "push_ny": (0, -1, 0),
                "twist": (0, 1, 0)}
        pp = self.pogo.pp
        if preset in ("drop_left", "drop_right"):
            sx = 1.0 if preset == "drop_right" else -1.0
            self.pulse("cubli", np.array([0, 0, -F]), duration, (sx * pp.l_E, 0.0, P.l_Q))
        else:
            self.pulse("cubli", np.array(dirs.get(preset, (0, 1, 0)), float) * F, duration)

    # ------------------------------------------------------------ dynamics
    def _apply_external(self):
        d = self.d
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

    def control_step(self):
        self._update_grab_force()
        pg = self.pogo
        pg.controller_on = self.controller_on and not self.fell
        pg.step()
        s = pg.state()
        self.u, self.ulim = pg.u, pg.ulim
        if s["fallen"] and not self.fell:
            self.fell = True
            self.message = "fell over - press Reset"
        self.samples.append([round(s["t"], 3), float(s["alpha"]), float(s["beta"]), float(self.u),
                             float(self.ulim), float(s["wheel"]), float(100 * pg.motor.i2t / P.i2t_budget)])

    def state(self):
        pg = self.pogo
        s = pg.state()
        if not self.fell:
            if pg.flipping:
                self.message = "somersault: flipping about the bar in the air"
            elif pg.flip_request:
                self.message = "somersault: winding with a slow wheel (small gyroscopic pitch error)"
            elif s["mode"] == "hop":
                self.message = "hopping: wheel above the clutch speed winds the spring, the cam fires it"
            elif s["mode"] == "stop":
                self.message = "stopping: finishing the current jump, then back to a stick"
            elif s["phase"] == "stick":
                self.message = (f"stick: balancing on the foot, wheel parked below the clutch speed "
                                f"({int(s['jumps'])} jumps, {pg.flips} somersaults so far)")
            else:
                self.message = "settling after a landing (one controlled hop may follow)"
        return dict(t=s["t"], fell=self.fell, tuning=self.tuning_name, com_enable=False, noise=1.0, delay=1,
                    controller_on=self.controller_on, yaw_on=False, yaw_available=False, yaw_mode="off",
                    heading_ref_deg=0.0, heading_est_deg=0.0, wheel_ref=float(pg.ctrl.w_ref),
                    com_est_deg=[0.0, 0.0], gamma_deg=0.0, beam_mrad=[0.0, 0.0],
                    plant=dict(tilt_deg=0.0, ecc_mm=0.0, m_e=pg.pp.m_e, f_beam=0.0, com_offset_mm=0.0),
                    pogo=dict(mode=s["mode"], phase=s["phase"], jumps=int(s["jumps"]), flips=int(pg.flips),
                              leg_mm=1e3 * float(s["leg"]), cam=float(s["cam"]),
                              height_cm=100 * float(s["z"] - 0.2),
                              apex_cm=100 * float(pg.apex[-1]) if pg.apex else 0.0,
                              airborne=bool(s["airborne"]), wheel=float(s["wheel"]),
                              clutch=bool(s["wheel"] > pg.pp.w_clutch), w_hop=float(pg.w_hop),
                              flip_deg=float(np.rad2deg(pg.flip_angle)) if pg.flipping else 0.0,
                              escapement=bool(pg.latched), message=self.message))

    # ----------------------------------------------------------- rendering
    def render_jpeg(self, quality=80):
        from PIL import Image
        d = self.d
        com = d.xipos[self.pogo.body]
        target = np.array([com[0], com[1], 0.25])
        if np.linalg.norm(target - self.cam.lookat) > 1.0:
            self.cam.lookat[:] = target
        self.cam.lookat[:] = 0.8 * np.asarray(self.cam.lookat) + 0.2 * target
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
