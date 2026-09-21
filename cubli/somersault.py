"""The paper's One-Wheel Cubli as a FREE body on the floor (contacts at the pivot
tip, housing corners, cantilever and end masses) -- for fall / jump-up / flip-over
manoeuvres, where the pivot model with joint limits is not enough.

Same masses, inertias, wheel and motor as the paper (rigid cantilever: the
bending mode is irrelevant for these large, slow manoeuvres).
"""
import numpy as np
import mujoco

from .params import NOMINAL
from .sim import Motor

P = NOMINAL


def build_xml(p=NOMINAL, timestep=5e-4, mu=0.8):
    m_body = p.m_h + 2 * p.m_e                             # housing + end masses (rigid)
    # composite inertia about the pivot, then about the composite CoM
    zc = (p.m_h * p.l_S + 2 * p.m_e * p.l_Q) / m_body
    Ix = p.I_hx + 2 * p.m_e * p.l_Q**2 - m_body * zc**2
    Iy = p.I_hy + 2 * p.m_e * (p.l_E**2 + p.l_Q**2) - m_body * zc**2
    Iz = p.I_hz + 2 * p.m_e * p.l_E**2
    Iwt = max(p.I_wy, 0.5 * p.I_wx * 1.0001)
    wq = f"{np.cos(p.eta / 2)} 0 0 {np.sin(p.eta / 2)}"
    h, z0 = 0.075, p.l_P
    corners = [(sx * h, sy * h, z0 + sz * h) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    edges = [(a, b) for i, a in enumerate(corners) for b in corners[i + 1:]
             if sum(abs(u - v) > 1e-9 for u, v in zip(a, b)) == 1]
    edges += [((0, 0, 0.0), c) for c in corners if c[2] < z0]
    col = f'contype="1" conaffinity="1" friction="{mu} 0.005 0.0001"'
    frame = "\n".join(
        f'      <geom type="capsule" fromto="{a[0]} {a[1]} {a[2]} {b[0]} {b[1]} {b[2]}" size="0.005" material="carbon" {col}/>'
        for a, b in edges)
    return f"""
<mujoco model="one_wheel_cubli_free">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep}" integrator="implicitfast" gravity="0 0 {-p.g0}" cone="elliptic"/>
  <default><geom density="0" contype="0" conaffinity="0"/></default>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.65 0.8" rgb2="0.1 0.12 0.18" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".78 .78 .78" rgb2=".66 .66 .68" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="20 20" reflectance="0.05"/>
    <material name="carbon" rgba="0.12 0.12 0.13 1"/>
    <material name="alu" rgba="0.72 0.74 0.78 1"/>
    <material name="wheel" rgba="0.85 0.35 0.1 1"/>
    <material name="mass" rgba="0.2 0.35 0.8 1"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 5" dir="-0.3 0.3 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>
    <light directional="true" pos="0 0 5" dir="0.4 -0.3 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="10 10 0.1" material="grid" contype="1" conaffinity="1" friction="{mu} 0.005 0.0001"/>
    <body name="cubli" pos="0 0 0.006">
      <freejoint name="free"/>
      <inertial pos="0 0 {zc}" mass="{m_body}" diaginertia="{Ix} {Iy} {Iz}"/>
      <geom name="tip" type="sphere" size="0.004" material="alu" {col}/>
{frame}
      <geom type="capsule" fromto="{-p.l_E} 0 {p.l_Q} {p.l_E} 0 {p.l_Q}" size="0.012" material="carbon" {col}/>
      <geom type="cylinder" pos="{-p.l_E} 0 {p.l_Q}" size="0.03 0.025" euler="0 1.5708 0" material="mass" {col}/>
      <geom type="cylinder" pos="{p.l_E} 0 {p.l_Q}" size="0.03 0.025" euler="0 1.5708 0" material="mass" {col}/>
      <body name="wheel" pos="0 0 {p.l_P}" quat="{wq}">
        <joint name="phi" type="hinge" axis="1 0 0"/>
        <inertial pos="0 0 0" mass="{p.m_w}" diaginertia="{p.I_wx} {Iwt} {Iwt}"/>
        <geom type="cylinder" fromto="0.012 0 0 0.024 0 0" size="0.068" material="wheel"/>
        <geom type="box" pos="0.0185 0 0.04" size="0.007 0.01 0.02" rgba="1 1 1 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="phi" gear="1" ctrlrange="{-p.tau_peak} {p.tau_peak}"/>
  </actuator>
</mujoco>
"""


def load(p=NOMINAL, timestep=5e-4, mu=0.8):
    m = mujoco.MjModel.from_xml_string(build_xml(p, timestep, mu))
    return m, mujoco.MjData(m)


def tilt_angles(m, d):
    """alpha (roll about x), beta (pitch about y) of the body z-axis, in the paper's
    convention, plus body angular velocity."""
    b = m.body("cubli").id
    R = d.xmat[b].reshape(3, 3)
    g = R.T @ np.array([0, 0, -1.0])                      # gravity in body frame
    alpha = np.arcsin(np.clip(-g[1], -1, 1))
    beta = np.arctan2(g[0], -g[2])
    return alpha, beta, d.qvel[3:6].copy(), R


def place(m, d, alpha=0.0, beta=0.0):
    """Put the body at a given tilt with the tip on the floor, then let contacts settle."""
    qa = np.zeros(4); mujoco.mju_axisAngle2Quat(qa, np.array([1.0, 0, 0]), alpha)
    qb = np.zeros(4); mujoco.mju_axisAngle2Quat(qb, np.array([0, 1.0, 0]), beta)
    q = np.zeros(4); mujoco.mju_mulQuat(q, qa, qb)
    d.qpos[3:7] = q
    d.qpos[:3] = [0, 0, 0.0045]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


# ---------------------------------------------------------------------------
from .controller import design
from .tunings import get

_K = None


def balance_gain():
    """The paper-structure LQR (tuned weights) on the pivot model, used as a
    full-state catch/balance controller once the tip is down and the body upright."""
    global _K
    if _K is None:
        K, *_ = design(NOMINAL.with_(com_offset_xy=(0.0, 0.0)), get("tuned"))
        _K = K[0, :9]
    return _K


class Maneuver:
    """Spin-up -> brake (throw) -> optional catch, on the free-body model.
    Records cumulative rotation about the body x-axis (the bar axis)."""

    def __init__(self, p=NOMINAL, dt=5e-4, Ts=0.01, mu=0.8):
        self.m, self.d = load(p, dt, mu)
        self.dt, self.Ts = dt, Ts
        self.wd = self.m.joint("phi").dofadr[0]
        self.K = balance_gain()

    def settle_upside_down(self):
        place(self.m, self.d, 0.05, 0.0)
        for _ in range(8000):
            mujoco.mj_step(self.m, self.d)

    def state9(self):
        a, b, w, R = tilt_angles(self.m, self.d)
        return np.array([a, w[0] / max(np.cos(b), 0.2), b, w[1], self.d.qvel[self.wd], 0, 0, 0, 0])

    def run(self, spin_torque, w_spin, brake_torque=-3.4, catch=True, catch_deg=12.0, w_rev=420.0,
            T=8.0, frames=None, frame_every=None, renderer=None, camera=None, pre=None):
        m, d = self.m, self.d
        motor = Motor(P)
        phase, t0 = "spin", d.time
        roll = 0.0
        log = []
        u_hold = 0.0
        steps_ctrl = int(round(self.Ts / self.dt))
        k = 0
        while d.time - t0 < T:
            if k % steps_ctrl == 0:
                a, b, w, R = tilt_angles(m, d)
                if pre is not None and phase == "spin":
                    u_hold = pre(d.time - t0, self)
                    if u_hold is None:
                        phase, u_hold = "brake", 0.0
                elif phase == "spin":
                    u_hold = spin_torque
                    if abs(d.qvel[self.wd]) >= w_spin:
                        phase = "brake"
                if phase == "brake":
                    u_hold = brake_torque
                    if abs(a) < np.deg2rad(catch_deg) and abs(b) < np.deg2rad(catch_deg) and catch \
                            and R[2, 2] > 0.9:
                        phase = "catch"
                    elif abs(d.qvel[self.wd]) > w_rev and np.sign(d.qvel[self.wd]) == np.sign(brake_torque):
                        u_hold = 0.0                     # wheel driven into reverse up to w_rev
                if phase == "catch":
                    u_hold = float(-(self.K @ self.state9()))
                u_hold, _ = motor.limit(u_hold, d.qvel[self.wd], self.Ts)
            d.ctrl[0] = u_hold
            mujoco.mj_step(m, d)
            roll += d.qvel[3] * self.dt                  # body-x angular velocity (bar axis)
            if frames is not None and renderer is not None and k % frame_every == 0:
                renderer.update_scene(d, camera=camera)
                frames.append(renderer.render())
            if k % steps_ctrl == 0:
                a, b, w, R = tilt_angles(m, d)
                log.append([d.time - t0, np.rad2deg(a), np.rad2deg(b), np.rad2deg(roll), d.qvel[self.wd],
                            u_hold, R[2, 2], {"spin": 0, "brake": 1, "catch": 2}[phase]])
            k += 1
        return np.array(log)


# ---------------------------------------------------------------------------
def body_energy(m, d):
    """Mechanical energy of the body (housing + end masses + wheel as a rigid part,
    excluding the wheel's own spin), with the potential referenced to the floor."""
    b = m.body("cubli").id
    wb = m.body("wheel").id
    R = d.xmat[b].reshape(3, 3)
    w_world = R @ d.qvel[3:6]
    E = 0.0
    for bid in (b, wb):
        mass = m.body_mass[bid]
        com = d.xipos[bid]
        v = d.qvel[0:3] + np.cross(w_world, com - d.xpos[b])
        Ib = m.body_inertia[bid]
        Rb = d.ximat[bid].reshape(3, 3)
        wl = Rb.T @ w_world                                 # body rotation only (no spin)
        E += 0.5 * mass * v @ v + 0.5 * wl @ (Ib * wl) + mass * P.g0 * com[2]
    return E


def upright_energy(m, d):
    """Energy of the body standing exactly upright at rest on its tip -- measured in
    the simulator (tip radius, contact penetration and the composite CoM included)."""
    d2 = mujoco.MjData(m)
    place(m, d2, 0.0, 0.0)
    for _ in range(200):                       # let the tip contact settle (upright is
        mujoco.mj_step(m, d2)                  # unstable, but 0.1 s is harmless)
    d2.qvel[:] = 0
    mujoco.mj_forward(m, d2)
    return body_energy(m, d2)


class SwingUp(Maneuver):
    """spin-up (pressed into the floor) -> energy-shaping swing-up -> LQR catch.
    Mode 'jump': arrive at the top with ~zero energy error and catch.
    Mode 'somersault': pump extra energy so the body rolls over the top once,
    then regulate to the top energy on the way round and catch it upright
    (a full 360 deg roll ending in balance)."""

    def run_energy(self, w_spin=300.0, k_e=6.0, extra=0.0, catch_deg=15.0, catch_rate=1.5,
                   T=12.0, roll_target=180.0, throw_rev=None, trim_after=100.0, u_trim=3.4,
                   remove_only=False,
                   frames=None, frame_every=None, renderer=None, camera=None):
        """throw_rev: if set, first a hard open-loop throw (full reverse torque until the
        wheel reaches -throw_rev), then energy trimming once the body has rolled
        past trim_after degrees, then the LQR catch."""
        m, d = self.m, self.d
        motor = Motor(P)
        Estar = upright_energy(m, d)
        phase, t0, roll = "spin", d.time, 0.0
        u = 0.0
        log = []
        steps = int(round(self.Ts / self.dt))
        k = 0
        ax_b = np.array([np.cos(P.eta), np.sin(P.eta), 0.0])
        while d.time - t0 < T:
            if k % steps == 0:
                a, b, w, R = tilt_angles(m, d)
                E = body_energy(m, d)
                wd = d.qvel[self.wd]
                w_world = R @ w
                if phase == "spin":
                    u = 0.6
                    if wd >= w_spin:
                        phase = "throw" if throw_rev is not None else "swing"
                if phase == "throw":
                    u = -3.4 if wd > -throw_rev else 0.0
                    if abs(np.rad2deg(roll)) > trim_after:
                        phase = "swing"
                if phase == "swing":
                    # reaction torque on the body is -u * axis: power into the body
                    # = -u * (axis . omega). Push energy toward the target.
                    roll_deg = abs(np.rad2deg(roll))
                    target = Estar + (extra if roll_deg < roll_target - 20 else 0.0)
                    s = float(np.dot(R @ ax_b, w_world))
                    err = target - E
                    if remove_only:
                        err = min(err, 0.0)                  # only bleed off excess energy
                    u = float(np.clip(-k_e * err * np.sign(s if abs(s) > 0.05 else 1.0), -u_trim, u_trim))
                    if abs(wd) > 400 and np.sign(u) == np.sign(wd):
                        u = 0.0                              # never push the wheel past its limit
                    if (R[2, 2] > np.cos(np.deg2rad(catch_deg)) and np.linalg.norm(w) < catch_rate
                            and roll_deg > roll_target - 40):
                        phase = "catch"
                if phase == "catch":
                    u = float(-(self.K @ self.state9()))
                u, _ = motor.limit(u, wd, self.Ts)
            d.ctrl[0] = u
            mujoco.mj_step(m, d)
            roll += d.qvel[3] * self.dt
            if frames is not None and renderer is not None and k % frame_every == 0:
                renderer.update_scene(d, camera=camera)
                frames.append(renderer.render())
            if k % steps == 0:
                a, b, w, R = tilt_angles(m, d)
                log.append([d.time - t0, np.rad2deg(a), np.rad2deg(b), np.rad2deg(roll), d.qvel[self.wd],
                            u, R[2, 2], {"spin": 0, "throw": 1, "swing": 3, "catch": 2}[phase], body_energy(m, d) - Estar])
            k += 1
        return np.array(log)
