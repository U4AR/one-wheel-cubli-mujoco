"""Pogo Cubli: the One-Wheel Cubli as a "cross" (horizontal cantilever with end
masses + a short vertical pogo leg under the housing), where the ONE motor both
balances (reaction wheel) and powers the jumps.

Wind-and-release mechanism (all driven by the reaction-wheel shaft):
  wheel --centrifugal clutch--ratchet--(reduction G)--> snail cam --> pulls the
  foot in (compresses the leg spring) over a fraction f_wind of each cam turn; at
  the cam step the follower drops off and the spring snaps the leg out -> jump.
  * the clutch engages only above w_clutch and the ratchet only drives forward,
    so the wheel's speed decides when the spring winds and fires: spin above the
    clutch speed to hop (the cam releases after G*f_wind wheel turns), park the
    wheel below it to stand still as a "stick";
  * the spring load comes back to the wheel as a torque F * ds/dcam / G (the same
    motor pays for the winding and the reaction goes into the body);
  * a rebound escapement, tripped by the touchdown impact, lets the leg compress
    freely but re-extend only slowly, so landings do not bounce.
Control (one torque): paper-style LQR on the foot while it is loaded; a
foot-contact switch hands the motor to a flight law the instant the foot unloads
(in the air the wheel can only turn the body about one direction, ~ the bar axis);
a bang/braking-guidance law flips the body 360 deg about the bar axis in flight.

The cam and escapement act as a moving upper limit on the leg joint (a MuJoCo
joint-limit constraint updated every 0.5 ms step).
"""
from dataclasses import dataclass, replace
import numpy as np
import mujoco
from scipy.linalg import expm, solve_discrete_are

from .params import NOMINAL as P
from .sim import Motor


@dataclass(frozen=True)
class PogoParams:
    l_E: float = 0.9            # cantilever half-length (paper 0.5975; lengthened to restore the inertia ratio)
    m_e: float = P.m_e
    L0: float = 0.03            # fixed leg below the housing tip (m)
    stroke: float = 0.05        # leg extension range (m)
    s_min: float = 0.01         # cam winds the leg in to this extension
    preload: float = 0.02       # spring preload length at full extension (m)
    k_leg: float = 4000.0       # leg spring (N/m)
    b_leg: float = 2.0          # leg damping (N s/m)
    b_comp: float = 150.0       # one-way (compression-only) shock-absorber valve (N s/m)
    m_foot: float = 0.05
    G: float = 20.0             # wheel -> cam reduction
    f_wind: float = 0.4         # fraction of a cam turn spent winding (rest: dwell, leg free)
    w_clutch: float = 20.0      # centrifugal clutch: cam drive engages above this wheel speed (rad/s)
    mu: float = 0.9

    def with_(self, **kw):
        return replace(self, **kw)


def _body_inertial(pp):
    m_body = P.m_h + 2 * pp.m_e
    zc = (P.m_h * P.l_S + 2 * pp.m_e * P.l_Q) / m_body
    Ix = P.I_hx + 2 * pp.m_e * P.l_Q**2 - m_body * zc**2
    Iy = P.I_hy + 2 * pp.m_e * (pp.l_E**2 + P.l_Q**2) - m_body * zc**2
    Iz = P.I_hz + 2 * pp.m_e * pp.l_E**2
    return m_body, zc, Ix, Iy, Iz


def _body_geoms(pp, col):
    h, z0 = 0.075, P.l_P
    corners = [(sx * h, sy * h, z0 + sz * h) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    edges = [(a, b) for i, a in enumerate(corners) for b in corners[i + 1:]
             if sum(abs(u - v) > 1e-9 for u, v in zip(a, b)) == 1]
    edges += [((0, 0, 0.0), c) for c in corners if c[2] < z0]
    g = "\n".join(f'      <geom type="capsule" fromto="{a[0]} {a[1]} {a[2]} {b[0]} {b[1]} {b[2]}" size="0.005" material="carbon" {col}/>'
                  for a, b in edges)
    g += f'''
      <geom type="capsule" fromto="{-pp.l_E} 0 {P.l_Q} {pp.l_E} 0 {P.l_Q}" size="0.012" material="carbon" {col}/>
      <geom type="cylinder" pos="{-pp.l_E} 0 {P.l_Q}" size="0.03 0.025" euler="0 1.5708 0" material="mass" {col}/>
      <geom type="cylinder" pos="{pp.l_E} 0 {P.l_Q}" size="0.03 0.025" euler="0 1.5708 0" material="mass" {col}/>
      <geom type="cylinder" fromto="0 0 0 0 0 {-pp.L0}" size="0.009" material="alu"/>'''
    return g


def _wheel_body(pp):
    Iwt = max(P.I_wy, 0.5 * P.I_wx * 1.0001)
    wq = f"{np.cos(P.eta / 2)} 0 0 {np.sin(P.eta / 2)}"
    return f'''      <body name="wheel" pos="0 0 {P.l_P}" quat="{wq}">
        <joint name="phi" type="hinge" axis="1 0 0"/>
        <inertial pos="0 0 0" mass="{P.m_w}" diaginertia="{P.I_wx} {Iwt} {Iwt}"/>
        <geom type="cylinder" fromto="0.012 0 0 0.024 0 0" size="0.068" material="wheel"/>
        <geom type="box" pos="0.0185 0 0.04" size="0.007 0.01 0.02" rgba="1 1 1 1"/>
      </body>'''


_HEAD = """
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{dt}" integrator="implicitfast" gravity="0 0 {g}" cone="elliptic"/>
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
    <material name="spring" rgba="0.95 0.75 0.2 1"/>
  </asset>"""


def build_free_xml(pp=PogoParams(), dt=5e-4):
    """Free-flying cross with a sprung foot (for stance, flight and landings)."""
    m_body, zc, Ix, Iy, Iz = _body_inertial(pp)
    col = f'contype="1" conaffinity="1" friction="{pp.mu} 0.005 0.0001"'
    h0 = pp.L0 + pp.stroke + 0.006
    return f"""<mujoco model="pogo_cubli">{_HEAD.format(dt=dt, g=-P.g0)}
  <worldbody>
    <light directional="true" pos="0 0 5" dir="-0.3 0.3 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>
    <light directional="true" pos="0 0 5" dir="0.4 -0.3 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="20 20 0.1" material="grid" contype="1" conaffinity="1" friction="{pp.mu} 0.005 0.0001"/>
    <body name="cubli" pos="0 0 {h0}">
      <freejoint name="free"/>
      <inertial pos="0 0 {zc}" mass="{m_body}" diaginertia="{Ix} {Iy} {Iz}"/>
{_body_geoms(pp, col)}
{_wheel_body(pp)}
      <body name="foot" pos="0 0 {-pp.L0}">
        <joint name="leg" type="slide" axis="0 0 -1" range="0 {pp.stroke}" stiffness="{pp.k_leg}"
               springref="{pp.stroke + pp.preload}" damping="{pp.b_leg}" solreflimit="0.003 1"/>
        <inertial pos="0 0 0" mass="{pp.m_foot}" diaginertia="1e-6 1e-6 1e-6"/>
        <geom name="foot" type="sphere" solref="0.005 1" size="0.006" material="alu" contype="1" conaffinity="1" friction="{pp.mu} 0.005 0.0001"/>
        <geom type="capsule" fromto="0 0 0 0 0 {pp.stroke}" size="0.006" material="spring"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="phi" gear="1" ctrlrange="{-P.tau_peak} {P.tau_peak}"/>
  </actuator>
</mujoco>"""


def build_pivot_xml(pp=PogoParams(), dt=5e-4, s=None):
    """Design model: foot pinned at the origin (hinges alpha, beta), leg locked at s."""
    s = pp.stroke if s is None else s
    m_body, zc, Ix, Iy, Iz = _body_inertial(pp)
    hb = pp.L0 + s
    return f"""<mujoco model="pogo_pivot">{_HEAD.format(dt=dt, g=-P.g0)}
  <worldbody>
    <body name="pivot" pos="0 0 0">
      <joint name="alpha" type="hinge" axis="1 0 0"/>
      <joint name="beta" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 0" mass="{pp.m_foot}" diaginertia="1e-6 1e-6 1e-6"/>
      <body name="cubli" pos="0 0 {hb}">
        <inertial pos="0 0 {zc}" mass="{m_body}" diaginertia="{Ix} {Iy} {Iz}"/>
{_wheel_body(pp)}
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="phi" gear="1"/>
  </actuator>
</mujoco>"""


# ---------------------------------------------------------------------------
def pivot_linear(pp=PogoParams(), s=None, w0=0.0, eps=1e-6):
    """Continuous A, B for x = (alpha, alpha_d, beta, beta_d, phi_d) about upright,
    wheel spinning at w0 (gyroscopic terms included)."""
    m = mujoco.MjModel.from_xml_string(build_pivot_xml(pp, s=s))
    d = mujoco.MjData(m)

    def f(q, qd, u):
        d.qpos[:] = q; d.qvel[:] = qd; d.ctrl[:] = u
        mujoco.mj_forward(m, d)
        return d.qacc.copy()

    q0, qd0 = np.zeros(3), np.array([0.0, 0.0, w0])
    idx = [(0, "q"), (0, "v"), (1, "q"), (1, "v"), (2, "v")]
    A = np.zeros((5, 5)); B = np.zeros((5, 1))
    rows = {1: 0, 3: 1, 4: 2}                     # state rows holding accelerations of joint j
    for c, (j, kind) in enumerate(idx):
        dq, dv = np.zeros(3), np.zeros(3)
        (dq if kind == "q" else dv)[j] = eps
        acc = (f(q0 + dq, qd0 + dv, 0) - f(q0 - dq, qd0 - dv, 0)) / (2 * eps)
        for r, jj in rows.items():
            A[r, c] = acc[jj]
    A[0, 1] = A[2, 3] = 1.0
    acc = (f(q0, qd0, eps) - f(q0, qd0, -eps)) / (2 * eps)
    for r, jj in rows.items():
        B[r, 0] = acc[jj]
    return A, B


def inertia_ratio(pp=PogoParams(), s=None):
    A, _ = pivot_linear(pp, s)
    return float(np.sqrt(A[3, 2] / A[1, 0])), float(A[1, 0]), float(A[3, 2])


class PogoBalance:
    """Full-state discrete LQR (paper structure: tilt, rates, wheel speed) with
    one-step delay compensation; wheel-speed reference sets the jump rate."""
    SCALE = np.array([np.deg2rad(1), np.deg2rad(10), np.deg2rad(1), np.deg2rad(10), 50.0])

    def __init__(self, pp=PogoParams(), Ts=0.01, Q=(1.0, 0.1, 1.0, 0.1, 3.0), Rw=20.0, w0=0.0):
        A, B = pivot_linear(pp, w0=w0)
        n = 5
        Mx = np.zeros((n + 1, n + 1)); Mx[:n, :n] = A; Mx[:n, n:] = B
        E = expm(Mx * Ts)
        self.Ad, self.Bd = E[:n, :n], E[:n, n:]
        S = np.diag(self.SCALE); Si = np.linalg.inv(S)
        An, Bn = Si @ self.Ad @ S, Si @ self.Bd
        Pr = solve_discrete_are(An, Bn, np.diag(Q), np.array([[Rw]]))
        self.K = (np.linalg.solve(Rw + Bn.T @ Pr @ Bn, Bn.T @ Pr @ An) @ Si)[0]
        self.u = 0.0
        self.w_ref = 0.0

    def step(self, x_delayed):
        xhat = self.Ad @ np.asarray(x_delayed, float) + self.Bd[:, 0] * self.u
        e = xhat.copy(); e[4] -= self.w_ref
        self.u = float(-(self.K @ e))
        return self.u


# ---------------------------------------------------------------------------
class PogoSim:
    """Free-body pogo cross with the cam/ratchet wind-and-release driven by the
    reaction-wheel motor, and the balance controller."""

    def __init__(self, pp=PogoParams(), dt=5e-4, Ts=0.01, ctrl=None, seed=0):
        self.pp, self.dt, self.Ts = pp, dt, Ts
        self.m = mujoco.MjModel.from_xml_string(build_free_xml(pp, dt))
        self.d = mujoco.MjData(self.m)
        self.wd = self.m.joint("phi").dofadr[0]
        self.wq = self.m.joint("phi").qposadr[0]
        self.lj = self.m.joint("leg").id
        self.ld = self.m.joint("leg").dofadr[0]
        self.lq = self.m.joint("leg").qposadr[0]
        self.foot_geom = self.m.geom("foot").id
        self.weight = float(sum(self.m.body_mass) * P.g0)
        self.load_on = 0.25
        self.floor_geom = self.m.geom("floor").id
        self.body = self.m.body("cubli").id
        self.ctrl = ctrl or PogoBalance(pp, Ts)
        self.motor = Motor(P)
        self.rng = np.random.default_rng(seed)
        self.substep_hook = None
        self.controller_on = True
        self.reset()

    def reset(self, tilt_deg=(1.0, 0.0)):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        qa = np.zeros(4); mujoco.mju_axisAngle2Quat(qa, np.array([1.0, 0, 0]), np.deg2rad(tilt_deg[0]))
        qb = np.zeros(4); mujoco.mju_axisAngle2Quat(qb, np.array([0, 1.0, 0]), np.deg2rad(tilt_deg[1]))
        q = np.zeros(4); mujoco.mju_mulQuat(q, qa, qb)
        d.qpos[3:7] = q
        d.qpos[2] = self.pp.L0 + self.pp.stroke + 0.0062
        d.qpos[self.lq] = self.pp.stroke
        mujoco.mj_forward(m, d)
        self._flight_geometry()
        self.cam = 2 * np.pi * self.pp.f_wind + 0.05       # start in the release window (leg free)
        self.psi_prev = d.qpos[self.wq]
        self.buf = [self.measure(), self.measure()]
        self.ctrl.u = 0.0
        self.jumps = 0
        self.airborne = False
        self.apex = []
        self.z_takeoff = 0.0
        self.max_z = 0.0
        self.flight_hold = True
        self.mode, self.phase, self.t_phase = "stick", "stick", 0.0
        self.w_hop, self.w_idle, self.w_park = 60.0, -20.0, -80.0
        self.settle_deg, self.settle_time = 1.0, 0.15
        self.ctrl.w_ref = 0.0
        self.w_ref_s, self.w_slew = 0.0, 400.0
        self.latch_on, self.latched, self.latch_pos, self.v_pay = True, False, self.pp.stroke, 0.1
        self.latch_armed, self.t_unloaded, self.t_latch = False, 0.0, 0.0
        self.was_air = False
        self.wind_for_flip = False
        self.flipping, self.flip_angle, self.flip_request, self.flips = False, 0.0, False, 0
        self.w_flip, self.flip_decel, self.flip_handover_deg = 30.0, 0.85, 0.3
        self.flip_braking, self.flip_kw, self.flip_klin = False, 40.0, 60.0

    # ----------------------------------------------------------- mechanism
    def cam_limit(self):
        pp = self.pp
        th = self.cam % (2 * np.pi)
        tw = 2 * np.pi * pp.f_wind
        if th < tw:
            return pp.stroke - (pp.stroke - pp.s_min) * th / tw, (pp.stroke - pp.s_min) / tw
        return pp.stroke, 0.0

    def flight_control(self):
        """Ballistic phase: no gravity torque. A wheel torque u turns the body along
        v = -I^-1 a u (a: wheel axis in the body frame), so regulate the attitude and
        rate components along c = v/|v| (the one direction the wheel can act on)."""
        x = self.buf[0]
        e = np.array([x[0], x[2]])
        w = np.array([x[1], x[3]])
        return float(np.clip(self.kf_p * (self.c_fl @ e) + self.kf_d * (self.c_fl @ w), -P.tau_peak, P.tau_peak))

    def _flight_geometry(self):
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        Rb = d.xmat[self.body].reshape(3, 3)
        Rw = d.xmat[m.body("wheel").id].reshape(3, 3)
        a = Rb.T @ Rw[:, 0]
        M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, d, M)
        # response of body angular acceleration to a unit wheel-joint torque
        dq = np.linalg.solve(M, np.eye(m.nv)[:, self.wd])
        v = dq[3:5]
        self.b_fl = float(np.linalg.norm(v))
        self.c_fl = -v / self.b_fl            # u > 0 drives the body rate along -c
        wn, zeta = 25.0, 0.9
        self.kf_p = wn**2 / self.b_fl
        self.kf_d = 2 * zeta * wn / self.b_fl

    def flip_law(self, u_attitude):
        """Flip about the bar axis (+x): full torque to spin the body up, then a
        braking guidance law that commands exactly the deceleration w^2/(2 rem)
        needed to arrive at 360 deg with zero rate (re-evaluated every substep);
        the attitude law takes the last few degrees."""
        rem = 2 * np.pi - self.flip_angle
        if self.flip_angle > np.deg2rad(300):
            rem = -self.tilt()[0]               # final approach on the measured tilt
        w = self.d.qvel[3]
        a_max = self.b_fl * P.tau_peak
        if not self.flip_braking and rem < np.deg2rad(self.flip_handover_deg):
            return u_attitude
        a_req = w * w / (2 * max(rem, 1e-3)) if w > 0 else 0.0
        if not self.flip_braking and a_req >= self.flip_decel * a_max:
            self.flip_braking = True
        if self.flip_braking:
            # constant-deceleration profile blended into a linear final approach
            # (w_des = k rem) so the rate reaches zero exactly at 360 deg
            a_nom = self.flip_decel * a_max
            k = self.flip_klin
            if rem > 2 * a_nom / k**2:
                w_des, a_ff = np.sqrt(2 * a_nom * rem), a_nom
            else:
                w_des, a_ff = k * rem, k * w
            return float(np.clip((a_ff + self.flip_kw * (w - w_des)) / self.b_fl, -P.tau_peak, P.tau_peak))
        return -P.tau_peak

    def cam_phase(self):
        return (self.cam % (2 * np.pi)) / (2 * np.pi)

    def foot_contact(self):
        d = self.d
        for i in range(d.ncon):
            c = d.contact[i]
            if self.foot_geom in (c.geom1, c.geom2):
                return True
        return False

    def unloaded(self):
        """flight-mode switch: foot unloaded and the landing escapement not engaged"""
        return self.foot_load() < self.load_on * self.weight and not self.latched

    def foot_load(self):
        """normal force on the foot (a load cell in the foot)"""
        d, f, F = self.d, np.zeros(6), 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if self.foot_geom in (c.geom1, c.geom2):
                mujoco.mj_contactForce(self.m, d, i, f)
                F += f[0]
        return F

    def body_contact(self):
        d = self.d
        for i in range(d.ncon):
            c = d.contact[i]
            g = c.geom2 if c.geom1 == self.floor_geom else (c.geom1 if c.geom2 == self.floor_geom else -1)
            if g >= 0 and g != self.foot_geom:
                return True
        return False

    def tilt(self):
        R = self.d.xmat[self.body].reshape(3, 3)
        g = R.T @ np.array([0, 0, -1.0])
        return np.arcsin(np.clip(-g[1], -1, 1)), np.arctan2(g[0], -g[2]), R

    def measure(self):
        a, b, R = self.tilt()
        w = self.d.qvel[3:6]
        x = np.array([a, w[0] / max(np.cos(b), 0.2), b, w[1], self.d.qvel[self.wd]])
        noise = np.array([np.deg2rad(0.03), 0.005, np.deg2rad(0.03), 0.005, 0.3])
        return x + self.rng.normal(0, 1, 5) * noise

    def step(self):
        """one control period (Ts)"""
        m, d = self.m, self.d
        self.hop_logic()
        self.buf.append(self.measure()); self.buf.pop(0)
        in_air = self.unloaded() and self.flight_hold
        if in_air:
            u_cmd = self.flight_control()
            self.ctrl.u = u_cmd
        else:
            if self.was_air:
                # touchdown: the stance predictor must not propagate flight data
                self.ctrl.u = float(self.d.ctrl[0])
                self.buf[0] = self.buf[-1]
            u_cmd = self.ctrl.step(self.buf[0])
        self.was_air = in_air
        if not self.controller_on:
            u_cmd = 0.0
        u, lim = self.motor.limit(u_cmd, d.qvel[self.wd], self.Ts)
        self.ctrl.u = u
        d.ctrl[0] = u
        self.u, self.ulim = u, lim
        for _ in range(int(round(self.Ts / self.dt))):
            # ratchet: the cam advances only with forward wheel rotation (relative to body)
            psi = d.qpos[self.wq]
            dpsi = psi - self.psi_prev
            self.psi_prev = psi
            engaged = dpsi > 0 and d.qvel[self.wd] > self.pp.w_clutch
            if engaged:
                self.cam += dpsi / self.pp.G
            e_cam, slope = self.cam_limit()
            # rebound escapement: engaged by the touchdown impact, the leg compresses
            # freely but may re-extend only at v_pay; drops out at full extension
            if self.latch_on:
                loaded = self.foot_load() > 0
                if loaded and self.t_unloaded > 0.05 and not self.latched:
                    self.latched, self.latch_armed = True, False
                    self.latch_pos, self.t_latch = d.qpos[self.lq] + 5e-4, d.time
                self.t_unloaded = 0.0 if loaded else self.t_unloaded + self.dt
                if self.latched:
                    self.latch_pos = min(self.latch_pos + self.v_pay * self.dt, d.qpos[self.lq] + 5e-4)
                    self.latch_armed |= d.qpos[self.lq] < self.pp.stroke - 0.004
                    if self.latch_armed and self.latch_pos >= self.pp.stroke:
                        self.latched = False
                    elif not self.latch_armed and d.time - self.t_latch > 0.3:
                        self.latched = False          # soft landing that never compressed
            lim = min(e_cam, self.latch_pos) if self.latched else e_cam
            m.jnt_range[self.lj, 1] = max(lim, 1e-4)
            if self.substep_hook is not None:          # external pushes (viewer)
                self.substep_hook()
            # spring load reflected to the wheel through the cam and gear (only while
            # the ratchet drives the cam; otherwise the pawl holds it on the body)
            F = 0.0
            for i in range(d.nefc):
                if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT and d.efc_id[i] == self.lj:
                    F += abs(d.efc_force[i])
            d.qfrc_applied[self.wd] = -(F * slope / self.pp.G) if engaged else 0.0
            vl = d.qvel[self.ld]
            d.qfrc_applied[self.ld] = -self.pp.b_comp * vl if vl < 0 else 0.0
            # foot-contact switch wired to the motor driver: the instant the foot
            # unloads, the stance command is replaced by the flight law
            if self.flight_hold and self.controller_on:
                if self.unloaded():
                    a_, b_, _ = self.tilt()
                    wb = d.qvel[3:5]
                    uf = self.kf_p * (self.c_fl @ np.array([a_, b_])) + self.kf_d * (self.c_fl @ wb)
                    if self.flipping:
                        uf = self.flip_law(uf)
                    d.ctrl[0], _ = self.motor.limit(float(np.clip(uf, -P.tau_peak, P.tau_peak)), d.qvel[self.wd], 0.0)
                else:
                    d.ctrl[0] = u
            mujoco.mj_step(m, d)
            if self.flipping:
                self.flip_angle += d.qvel[3] * self.dt
        # hop bookkeeping
        on = self.foot_contact()
        z = d.xipos[self.body][2]
        if self.airborne:
            self.max_z = max(self.max_z, z)
            if on:
                self.airborne = False
                h = self.max_z - self.z_takeoff
                if h > 0.01:
                    self.apex.append(h)
                else:
                    self.jumps -= 1          # landing rebound, not a hop
        elif not on and d.qvel[2] > 0.2:
            self.airborne = True
            self.jumps += 1
            self.z_takeoff = z
            self.max_z = z

    # ----------------------------------------------------------- hop logic
    def set_mode(self, mode):
        """'stick' (balance only), 'hop' (continuous jumping), 'stop' (finish the
        current jump, then back to stick)."""
        if mode == "flip":
            self.flip_request = True
            if self.phase == "stick":
                self.phase, self.t_phase = "settle", self.d.time
            if self.mode == "stick":
                self.mode = "stop"             # one jump (the flip), then stick again
            return
        if mode == "stop" and self.mode == "hop":
            self.mode = "stop"
        elif mode in ("stick", "hop"):
            self.mode = mode
            if mode == "hop" and self.phase == "stick":
                self.phase, self.t_phase = "settle", self.d.time

    def hop_logic(self):
        d, pp = self.d, self.pp
        s = self.state()
        quiet = (abs(s["alpha"]) < self.settle_deg and abs(s["beta"]) < self.settle_deg
                 and np.linalg.norm(d.qvel[3:5]) < np.deg2rad(8) and abs(d.qvel[2]) < 0.05)
        in_wind = self.cam_phase() < pp.f_wind
        if self.phase == "flight":
            if not self.airborne:
                if self.flipping:
                    self.flipping = False
                    self.flips += int(self.flip_angle > np.deg2rad(300))
                self.phase, self.t_phase = "settle", d.time
        elif self.airborne:
            self.phase, self.t_phase = "flight", d.time
            if self.flip_request and self.wind_for_flip:
                self.flipping, self.flip_angle, self.flip_request, self.flip_braking = True, 0.0, False, False
            self.wind_for_flip = False
        elif self.phase == "settle":
            if self.mode == "stop" and not in_wind and not self.flip_request:
                self.mode, self.phase = "stick", "stick"
            elif self.mode == "stick":
                self.phase = "stick"
            elif quiet and d.time - self.t_phase > self.settle_time:
                self.phase, self.t_phase = "wind", d.time
                self.wind_for_flip = self.flip_request
        elif self.phase == "wind":
            if self.mode == "stick":
                self.phase = "stick"
        elif self.phase == "stick" and self.mode == "hop":
            self.phase, self.t_phase = "settle", d.time
        elif self.phase == "stick" and in_wind and d.qvel[self.wd] > pp.w_clutch:
            # the clutch caught during a big balance transient: the cam is winding,
            # so fly this jump as a controlled hop and come back to stick
            self.mode, self.phase, self.t_phase = "stop", "wind", d.time
            self.w_ref_s = d.qvel[self.wd]
        # wheel-speed reference: above the clutch speed only while winding
        if self.phase == "wind":
            # flips wind (and so take off) with a slow wheel: the gyroscopic pitch
            # error of the flip grows with the wheel speed
            target = self.w_flip if self.wind_for_flip else self.w_hop
        elif self.mode == "hop" or self.flip_request:
            target = self.w_idle
        else:
            target = self.w_park                     # stick: keep the clutch well disengaged
        if self.phase == "flight":
            self.w_ref_s = d.qvel[self.wd]          # touch down with the wheel speed it has
        step = self.w_slew * self.Ts
        self.w_ref_s += np.clip(target - self.w_ref_s, -step, step)
        self.ctrl.w_ref = self.w_ref_s

    def state(self):
        a, b, R = self.tilt()
        return dict(t=self.d.time, alpha=np.rad2deg(a), beta=np.rad2deg(b), upright=R[2, 2],
                    wheel=self.d.qvel[self.wd], leg=self.d.qpos[self.lq], cam=self.cam_phase(),
                    airborne=self.airborne, jumps=self.jumps, mode=self.mode, phase=self.phase, z=self.d.xipos[self.body][2],
                    x=self.d.qpos[0], y=self.d.qpos[1], fallen=bool((R[2, 2] < 0.7 and not (self.flipping and self.airborne)) or self.body_contact()))
