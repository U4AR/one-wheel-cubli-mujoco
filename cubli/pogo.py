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
    v_pay: float = 0.1          # rebound escapement pay-out speed (m/s)
    Ts: float = 0.01            # control period (s)
    q_wheel: float = 3.0        # LQR weight on the wheel-speed error
    w_park: float = -80.0       # wheel speed parked in stick mode (below the clutch)
    catch_window: float = 1.0   # stick mode: if the clutch catches within this fraction of a cam
                                # turn before the release, fly that jump as a controlled hop
    hw: object = None           # cubli.pogo_hw.Hardware: build from a bill of materials instead of the paper

    def with_(self, **kw):
        return replace(self, **kw)


def _body_inertial(pp):
    """(mass, CoM (3,), inertia about the CoM (3x3)) of the body without flywheel and foot."""
    if pp.hw is not None:
        return pp.hw.body_props()
    m_body = P.m_h + 2 * pp.m_e
    zc = (P.m_h * P.l_S + 2 * pp.m_e * P.l_Q) / m_body
    Ix = P.I_hx + 2 * pp.m_e * P.l_Q**2 - m_body * zc**2
    Iy = P.I_hy + 2 * pp.m_e * (pp.l_E**2 + P.l_Q**2) - m_body * zc**2
    Iz = P.I_hz + 2 * pp.m_e * pp.l_E**2
    return m_body, np.array([0.0, 0.0, zc]), np.diag([Ix, Iy, Iz])


def _inertial_xml(pp):
    m, c, I = _body_inertial(pp)
    return (f'<inertial pos="{c[0]} {c[1]} {c[2]}" mass="{m}" '
            f'fullinertia="{I[0, 0]} {I[1, 1]} {I[2, 2]} {I[0, 1]} {I[0, 2]} {I[1, 2]}"/>')


def motor_params(pp):
    return pp.hw.motor.params() if pp.hw is not None else P


def _body_geoms(pp, col):
    if pp.hw is not None:
        return "      " + pp.hw.geoms(col) + f'''
      <geom type="cylinder" fromto="0 0 0 0 0 {-pp.L0}" size="{0.6 * pp.hw.foot_r * 2}" material="alu"/>'''
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
    if pp.hw is not None:
        return pp.hw.wheel_xml()
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
    col = f'contype="1" conaffinity="1" friction="{pp.mu} 0.005 0.0001"'
    foot_r = pp.hw.foot_r if pp.hw is not None else 0.006
    h0 = pp.L0 + pp.stroke + foot_r
    cam = pp.hw.cam_xml() if pp.hw is not None else ""
    # small robot: a rubber foot with real (small) torsional friction, so turning on it costs torque
    foot_fr = (f'condim="4" friction="{pp.mu} {pp.hw.foot_torsion} 0.0001"' if pp.hw is not None
               else f'friction="{pp.mu} 0.005 0.0001"')
    floor_tors = 0.0001 if pp.hw is not None else 0.005
    tau_peak = motor_params(pp).tau_peak
    return f"""<mujoco model="pogo_cubli">{_HEAD.format(dt=dt, g=-P.g0)}
  <worldbody>
    <light directional="true" pos="0 0 5" dir="-0.3 0.3 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>
    <light directional="true" pos="0 0 5" dir="0.4 -0.3 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="20 20 0.1" material="grid" contype="1" conaffinity="1" friction="{pp.mu} {floor_tors} 0.0001"/>
    <body name="cubli" pos="0 0 {h0}">
      <freejoint name="free"/>
      {_inertial_xml(pp)}
{_body_geoms(pp, col)}
{_wheel_body(pp)}
{cam}
      <body name="foot" pos="0 0 {-pp.L0}">
        <joint name="leg" type="slide" axis="0 0 -1" range="0 {pp.stroke}" stiffness="{pp.k_leg}"
               springref="{pp.stroke + pp.preload}" damping="{pp.b_leg}" solreflimit="0.003 1"/>
        <inertial pos="0 0 0" mass="{pp.m_foot}" diaginertia="1e-6 1e-6 1e-6"/>
        <geom name="foot" type="sphere" solref="0.005 1" size="{foot_r}" material="alu" contype="1" conaffinity="1" {foot_fr}/>
        <geom type="capsule" fromto="0 0 0 0 0 {pp.stroke}" size="{foot_r}" material="spring"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="phi" gear="1" ctrlrange="{-tau_peak} {tau_peak}"/>
  </actuator>
</mujoco>"""


def build_pivot_xml(pp=PogoParams(), dt=5e-4, s=None):
    """Design model: foot pinned at the origin (hinges alpha, beta), leg locked at s."""
    s = pp.stroke if s is None else s
    hb = pp.L0 + s
    return f"""<mujoco model="pogo_pivot">{_HEAD.format(dt=dt, g=-P.g0)}
  <worldbody>
    <body name="pivot" pos="0 0 0">
      <joint name="alpha" type="hinge" axis="1 0 0"/>
      <joint name="beta" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 0" mass="{pp.m_foot}" diaginertia="1e-6 1e-6 1e-6"/>
      <body name="cubli" pos="0 0 {hb}">
        {_inertial_xml(pp)}
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

    def __init__(self, pp=PogoParams(), Ts=None, Q=None, Rw=20.0, w0=0.0):
        Ts = pp.Ts if Ts is None else Ts
        Q = (1.0, 0.1, 1.0, 0.1, pp.q_wheel) if Q is None else Q
        # the torque weight is normalised to the paper motor's peak torque
        Rw = Rw * (P.tau_peak / motor_params(pp).tau_peak) ** 2
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
        self.x_ref = np.zeros(5)            # tilt reference (e.g. a lean before a forward hop)

    def step(self, x_delayed):
        xhat = self.Ad @ np.asarray(x_delayed, float) + self.Bd[:, 0] * self.u
        e = xhat - self.x_ref; e[4] -= self.w_ref
        self.u = float(-(self.K @ e))
        return self.u


# ---------------------------------------------------------------------------
class PogoSim:
    """Free-body pogo cross with the cam/ratchet wind-and-release driven by the
    reaction-wheel motor, and the balance controller."""

    def __init__(self, pp=PogoParams(), dt=5e-4, Ts=None, ctrl=None, seed=0):
        Ts = pp.Ts if Ts is None else Ts
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
        A_piv, B_piv = pivot_linear(pp)
        self.pi_a = float(np.sqrt(max(A_piv[1, 0], 1e-6)))     # roll fall rate constant on the foot
        self.b_ground = float(abs(B_piv[1, 0]))                 # roll acceleration per N m, pivoting on the foot
        self.b_ground_signed = float(B_piv[1, 0])
        self.I_w = float(pp.hw.wheel()["I_ax"]) if pp.hw is not None else float(P.I_wx)
        mujoco.mj_forward(self.m, self.d)
        Rb = self.d.xmat[self.body].reshape(3, 3)
        self.wheel_axis_body = Rb.T @ self.d.xmat[self.m.body("wheel").id].reshape(3, 3)[:, 0]
        Mf = np.zeros((self.m.nv, self.m.nv)); mujoco.mj_fullM(self.m, self.d, Mf)
        self.I_z = float(Mf[5, 5])
        # frictionless estimate of body yaw per unit wheel-speed change (rad per rad/s)
        self.yaw_per_dw = max(abs(self.I_w * self.wheel_axis_body[2] / self.I_z), 1e-6) * 0.6
        self.tau_s = (pp.hw.foot_torsion if pp.hw is not None else 0.0) * self.weight
        self.mp = motor_params(pp)
        self.motor = Motor(self.mp)
        self.has_stops = pp.hw is not None
        self.camq = self.m.joint('camj').qposadr[0] if self.has_stops else None
        self.camd = self.m.joint('camj').dofadr[0] if self.has_stops else None
        self.E_used = 0.0
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
        d.qpos[2] = self.pp.L0 + self.pp.stroke + (self.pp.hw.foot_r if self.has_stops else 0.006) + 2e-4
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
        self.w_hop, self.w_idle, self.w_park = 60.0, -20.0, self.pp.w_park
        self.settle_deg, self.settle_time = 1.0, 0.15
        self.ctrl.w_ref = 0.0
        self.w_ref_s, self.w_slew = 0.0, 400.0
        self.latch_on, self.latched, self.latch_pos, self.v_pay = True, False, self.pp.stroke, self.pp.v_pay
        self.latch_armed, self.t_unloaded, self.t_latch = False, 0.0, 0.0
        self.was_air = False
        self.wind_for_flip = False
        self.flipping, self.flip_angle, self.flip_request, self.flips = False, 0.0, False, 0
        self.w_flip, self.flip_decel, self.flip_handover_deg = 30.0, 0.85, 0.3
        self.flip_braking, self.flip_kw, self.flip_klin = False, 40.0, 60.0
        self.auto_getup, self.t_down, self.getups = True, 0.0, 0
        self.fwd, self.alpha_target, self.lean_deg, self.t_lean, self.cp_gain = 0, 0.0, 0.0, 0.25, 0.5
        self.w_fwd = 60.0
        self.step_deg = 0.0
        self.kick_deg, self.kick_T, self.kick_U, self.cp_max = 1.0, 0.12, 0.0, 12.0
        self.lean_frac, self.lean_ramp = 0.5, 0.3
        self.hop_start_y = None
        self.vault_a0 = 0.0
        self.vault_lqr = False
        self.use_vault = False
        # heading (yaw) control: wheel-speed ratchet in stance (needs the tilted wheel)
        self.heading_target, self.yaw_hold, self.yaw_tol = None, False, np.deg2rad(4.0)
        self.turn_stage, self.turn_w0, self.turn_dw = None, 0.0, 0.0
        self.turn_fast, self.turn_slow, self.turn_dw_max = 600.0, 90.0, 350.0
        self.yaw_mode, self.yaw_ofs, self.yaw_rf, self.yaw_uff = "off", 0.0, 0.0, 0.0
        self.yaw_kp, self.yaw_kd, self.yaw_a_max, self.yaw_w_turn = 1.0, 2.0, 400.0, 150.0
        self.yaw_K, self.yaw_w_max, self.yaw_short_neg = 700.0, 350.0, np.deg2rad(20.0)
        self.turns = 0
        self.resume_mode = "stick"
        mujoco.mj_forward(m, d)
        self.h_com = float(d.subtree_com[self.body][2] - (d.xpos[self.m.body('foot').id][2] - self.m.geom_size[self.foot_geom][0]))
        self.getup_gain, self.getup_k, self.getup_dw, self.getup_prespin = 1.05, 40.0, 250.0, False

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
        e = np.array([x[0] - self.alpha_target, x[2]])
        w = np.array([x[1], x[3]])
        return float(np.clip(self.kf_p * (self.c_fl @ e) + self.kf_d * (self.c_fl @ w), -self.mp.tau_peak, self.mp.tau_peak))

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
        a_max = self.b_fl * self.mp.tau_peak
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
            return float(np.clip((a_ff + self.flip_kw * (w - w_des)) / self.b_fl, -self.mp.tau_peak, self.mp.tau_peak))
        return -self.mp.tau_peak

    def getup_law(self):
        """Lift off a skid / pod: drive the roll rate onto the inverted-pendulum
        separatrix w = -pi_a * alpha (the rate that coasts to upright and stops
        there), re-tracked every step; the balance LQR takes over near upright."""
        a, b, _ = self.tilt()
        wx = self.d.qvel[3]
        w = self.d.qvel[self.wd]
        u_lift = -np.sign(a) / np.sign(self.b_ground_signed)   # sign of the lifting torque
        if self.getup_prespin:
            # still on the skid: spin the wheel the other way first (the reaction only
            # presses the body into the skid) so the lift cannot leave it above the clutch
            target = self.pp.w_clutch - 30.0 - self.getup_dw
            if u_lift > 0 and w > target and self.body_contact():
                return float(-0.6 * self.mp.tau_peak)
            self.getup_prespin = False
        w_des = -self.getup_gain * self.pi_a * a
        # roll acceleration = b_ground_signed * u  ->  track w_des with gain getup_k
        return float(np.clip(self.getup_k * (w_des - wx) / self.b_ground_signed, -self.mp.tau_peak, self.mp.tau_peak))

    def start_turn(self):
        """one ratchet cycle sized to the heading error"""
        d = self.d
        e = self.heading_error()
        a_z = float(self.wheel_axis_body[2])
        # body yaw accel = -I_w * dw/dt * a_z / I_z : a FAST decrease of w turns the body +yaw
        dw = float(np.clip(abs(e) / self.yaw_per_dw, 60.0, self.turn_dw_max))
        self.turn_w0 = float(self.w_ref_s)
        self.turn_goal = self.turn_w0 - dw
        fast_down = (e * a_z) > 0            # +yaw needed: fast down, slow back up
        self.turn_order = ("fast", "slow") if fast_down else ("slow", "fast")
        self.turn_stage = self.turn_order[0]
        self.phase, self.t_phase = "turn", d.time

    def yaw_rate(self):
        R = self.d.xmat[self.body].reshape(3, 3)
        return float((R @ self.d.qvel[3:6])[2])

    def yaw_wheel_ref(self):
        """Heading hold / turning. On its point foot the balanced robot turns at a rate
        proportional to the wheel speed it balances at (measured: ~ -0.04 deg/s per
        rad/s; the balance torque's vertical component through the tilted wheel axis
        works against the foot's torsional friction). So steering = choosing the
        wheel-speed set point. Turning -yaw needs +w, which would engage the winding
        clutch: bigger -yaw turns go the long way round."""
        e = self.heading_error()
        if e < -self.yaw_short_neg:
            e += 2 * np.pi
        self.yaw_mode = "turn" if abs(e) > self.yaw_tol else "hold"
        return float(np.clip(-self.yaw_K * e, -self.yaw_w_max, self.pp.w_clutch - 12.0))

    def yaw_step(self, w_base):
        """Heading control with the tilted wheel (after cubli/yaw.py): the wheel's
        vertical torque component turns the body on its foot; only foot friction can
        change the vertical momentum, so big turns ratchet: TURN (motor accelerates the
        wheel, beating the foot's stiction), UNLOAD (body held by friction, wheel speed
        bled back gently), turn again. Returns the wheel-speed offset for the balance
        loop and sets self.yaw_uff, the motor feed-forward torque."""
        e = self.heading_error()                    # target - heading
        r = self.yaw_rate()
        self.yaw_rf += (self.Ts / 0.3) * (r - self.yaw_rf)
        moving = abs(self.yaw_rf) > np.deg2rad(0.8)
        s, Iw, Iz, tau_s = self.wheel_axis_body[2], self.I_w, self.I_z, self.tau_s
        centre = -self.yaw_w_turn
        a_bleed = 0.3 * tau_s / (s * Iw)
        dev = self.yaw_ofs - centre
        if self.yaw_mode == "unload" and abs(dev) > 20.0:
            pass
        elif abs(e) < self.yaw_tol and not moving:
            self.yaw_mode = "hold"
        elif abs(dev) > self.yaw_w_turn:
            self.yaw_mode = "unload"
        else:
            self.yaw_mode = "turn"
        if self.yaw_mode in ("hold", "unload"):
            a = 0.0 if moving else -np.sign(dev) * min(a_bleed, abs(dev) / 0.5)
        else:
            tau = Iz * (self.yaw_kp * np.clip(e, -0.5, 0.5) - self.yaw_kd * self.yaw_rf)
            tau += (tau_s * np.sign(self.yaw_rf)) if moving else (1.3 * tau_s * np.sign(tau))
            a = float(np.clip(-tau / (s * Iw), -self.yaw_a_max, self.yaw_a_max))
        # stay inside [centre - w_turn, centre + w_turn] (the top is below the clutch)
        new = float(np.clip(self.yaw_ofs + a * self.Ts, centre - self.yaw_w_turn, centre + self.yaw_w_turn))
        a = (new - self.yaw_ofs) / self.Ts
        self.yaw_ofs = new
        self.yaw_uff = float(Iw * a)
        return self.yaw_ofs

    def heading(self):
        """yaw of the body x axis (the bar) in the world, from the external tracking"""
        R = self.d.xmat[self.body].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def heading_error(self):
        if self.heading_target is None:
            return 0.0
        return float((self.heading_target - self.heading() + np.pi) % (2 * np.pi) - np.pi)

    def set_heading(self, psi=None, delta=None):
        """hold / turn to a heading (rad); delta turns relative to the current one"""
        if delta is not None:
            base = self.heading_target if self.heading_target is not None else self.heading()
            psi = base + delta
        self.heading_target = None if psi is None else float((psi + np.pi) % (2 * np.pi) - np.pi)
        self.yaw_hold = self.heading_target is not None

    def winding_torque(self):
        """spring load reflected to the motor through the cam and gear (0 in the dwell)"""
        pp = self.pp
        e_cam, slope = self.cam_limit()
        if slope == 0.0:
            return 0.0
        F = pp.k_leg * (pp.preload + pp.stroke - e_cam)
        return F * slope / pp.G

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
        if self.phase == "vault" and self.vault_lqr:
            tau = d.time - self.t_phase
            self.ctrl.x_ref[0] = self.vault_a0 * np.exp(-self.pi_a * tau)
            self.ctrl.x_ref[1] = -self.pi_a * self.ctrl.x_ref[0]
        if self.phase in ("kick", "coast"):
            # take-off lean: an open-loop torque doublet (+U then -U). Zero net impulse,
            # so the pitch rate, yaw rate and wheel speed all come back to where they
            # were; only the roll angle (low inertia) moves by the planned lean
            tau = d.time - self.t_phase
            u_cmd = (self.kick_U if tau < self.kick_T / 2 else -self.kick_U) if self.phase == "kick" else 0.0
            if d.qvel[self.wd] > self.pp.w_clutch:
                u_cmd += self.winding_torque()
            self.ctrl.u = u_cmd
        elif self.phase == "getup" or (self.phase == "vault" and not self.vault_lqr):
            u_cmd = self.getup_law()
            self.ctrl.u = u_cmd
            self.w_ref_s = d.qvel[self.wd]
        elif in_air:
            u_cmd = self.flight_control()
            self.ctrl.u = u_cmd
        else:
            if self.was_air:
                # touchdown: the stance predictor must not propagate flight data
                self.ctrl.u = float(self.d.ctrl[0])
                self.buf[0] = self.buf[-1]
            u_cmd = self.ctrl.step(self.buf[0])
        self.was_air = in_air
        if self.has_stops and not in_air and self.phase == "wind" and d.qvel[self.wd] > self.pp.w_clutch:
            u_cmd += self.winding_torque()          # known cam load: feed it forward
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
                    uf = self.kf_p * (self.c_fl @ np.array([a_ - self.alpha_target, b_])) + self.kf_d * (self.c_fl @ wb)
                    if self.flipping:
                        uf = self.flip_law(uf)
                    d.ctrl[0], _ = self.motor.limit(float(np.clip(uf, -self.mp.tau_peak, self.mp.tau_peak)), d.qvel[self.wd], 0.0)
                else:
                    d.ctrl[0] = u
            mujoco.mj_step(m, d)
            if self.camq is not None:           # show the cam turning (kinematic, visual only)
                d.qpos[self.camq] = -self.cam
                d.qvel[self.camd] = 0.0
            if self.has_stops:
                self.E_used += (self.pp.hw.motor.electrical_power(d.ctrl[0], d.qvel[self.wd])
                                + self.pp.hw.electronics_W) * self.dt
            if self.flipping:
                self.flip_angle += d.qvel[3] * self.dt
        # hop bookkeeping
        on = self.foot_contact()
        z = d.xipos[self.body][2]
        if self.airborne:
            self.max_z = max(self.max_z, z)
            if on or self.body_contact():
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
        if self.phase == "getup" and mode in ("hop", "hop_fwd", "hop_back", "stick", "stop"):
            # remember it: the get-up hands over to this mode when it is back up
            self.fwd = {"hop_fwd": 1, "hop_back": -1}.get(mode, 0)
            self.resume_mode = "hop" if mode.startswith("hop") else "stick"
            return
        if mode in ("turn_left", "turn_right", "turn_left45", "turn_right45"):
            # +yaw (counter-clockwise seen from above) is a LEFT turn for a robot whose
            # forward is its body +y axis
            deg = {"turn_left": 90, "turn_right": -90, "turn_left45": 45, "turn_right45": -45}[mode]
            self.set_heading(delta=np.deg2rad(deg))
            return
        if mode == "hold_heading":
            self.set_heading(self.heading())
            return
        if mode == "flip":
            self.flip_request = True
            if self.phase == "stick":
                self.phase, self.t_phase = "settle", self.d.time
            if self.mode == "stick":
                self.mode = "stop"             # one jump (the flip), then stick again
            return
        if mode == "stop" and self.mode == "hop":
            # stop now: park the wheel below the clutch speed (the cam stays where it
            # is, spring partly wound); if airborne, after the landing
            self.mode = "stop" if self.phase == "flight" else "stick"
            if self.mode == "stick":
                self.phase = "stick"
        elif mode in ("hop_fwd", "hop_back"):
            self.fwd = 1 if mode == "hop_fwd" else -1
            if self.has_stops and not self.yaw_hold:
                self.set_heading(self.heading())      # go straight: hold this heading
            self.mode = "hop"
            if self.phase == "stick":
                self.phase, self.t_phase = "settle", self.d.time
        elif mode in ("stick", "hop"):
            self.fwd = 0
            self.mode = mode
            if mode == "hop" and self.phase == "stick":
                self.phase, self.t_phase = "settle", self.d.time

    def hop_logic(self):
        d, pp = self.d, self.pp
        s = self.state()
        quiet = (abs(s["alpha"]) < self.settle_deg and abs(s["beta"]) < self.settle_deg
                 and np.linalg.norm(d.qvel[3:5]) < np.deg2rad(8) and abs(d.qvel[2]) < 0.05)
        in_wind = self.cam_phase() < pp.f_wind
        a_, b_, _ = self.tilt()
        if self.has_stops and self.auto_getup and self.phase != "getup" and not self.airborne:
            resting = self.body_contact()
            self.t_down = self.t_down + self.Ts if resting else 0.0
            if self.t_down > 0.05:
                self.resume_mode = self.mode if self.mode in ("hop", "stick") else "stick"
                self.mode, self.phase, self.t_phase = "stick", "getup", d.time
                self.flip_request = False
                self.getup_prespin = True
                self.getups += 1
        if self.phase == "getup":
            if abs(np.rad2deg(a_)) < 2.5 and abs(np.rad2deg(b_)) < 4.0 and not self.body_contact():
                self.phase, self.t_phase = "stick", d.time
                self.mode = self.resume_mode
                self.ctrl.u = float(d.ctrl[0])
                self.w_ref_s = d.qvel[self.wd]
            elif d.time - self.t_phase > 3.0:      # give the balance LQR a go, then retry
                self.phase, self.t_phase, self.t_down = "stick", d.time, 0.0
                self.mode = self.resume_mode
        elif self.phase == "vault":
            done = (d.time - self.t_phase > 4.0 / self.pi_a) if self.vault_lqr else abs(np.rad2deg(a_)) < 2.0
            if done or d.time - self.t_phase > 1.5:
                self.ctrl.x_ref[:2] = 0.0
                self.phase, self.t_phase = "settle", d.time
                self.ctrl.u = float(d.ctrl[0])
                self.w_ref_s = d.qvel[self.wd]
        elif self.phase == "flight":
            if not self.airborne:
                if self.fwd and self.use_vault and abs(self.alpha_target) > np.deg2rad(1.0) and not self.flipping:
                    # landed at the capture point: let it vault up along the separatrix
                    self.alpha_target = 0.0
                    self.phase, self.t_phase = "vault", d.time
                    self.vault_a0 = float(a_)
                    self.ctrl.u = float(d.ctrl[0])
                    return
                self.alpha_target = 0.0
                if self.flipping:
                    self.flipping = False
                    self.flips += int(self.flip_angle > np.deg2rad(300))
                self.phase, self.t_phase = "settle", d.time
        elif self.airborne:
            self.phase, self.t_phase = "flight", d.time
            self.ctrl.x_ref[:] = 0.0
            if self.fwd and not (self.flip_request and self.wind_for_flip):
                # the lean at take-off gave the CoM a horizontal velocity; in the air,
                # turn the body about the bar axis (the axis the wheel controls well) so
                # the foot lands at the capture point (plus a small step): the stance
                # vault then brings it upright over the foot, carrying it forward.
                # Travel is along +y for fwd=+1: alpha > 0 puts the foot toward +y, and
                # lifting back over it spins the wheel UP (keeps the clutch winding).
                R = d.xmat[self.body].reshape(3, 3)
                yb = np.array([R[0, 1], R[1, 1], 0.0])
                yb /= max(np.linalg.norm(yb), 1e-6)
                vy = float(d.qvel[0:3] @ yb)                # velocity along the body y axis
                cap = vy / (self.h_com * self.pi_a) * self.cp_gain
                self.alpha_target = float(np.clip(cap + self.fwd * np.deg2rad(self.step_deg),
                                                  -np.deg2rad(self.cp_max), np.deg2rad(self.cp_max)))
            if self.flip_request and self.wind_for_flip:
                self.flipping, self.flip_angle, self.flip_request, self.flip_braking = True, 0.0, False, False
            self.wind_for_flip = False
        elif self.phase == "kick":
            if d.time - self.t_phase >= self.kick_T:
                self.phase, self.t_phase = "coast", d.time
        elif self.phase == "coast":
            if d.time - self.t_phase > 0.25:          # the release did not come: back to winding
                self.phase, self.t_phase = "wind", d.time
                self.ctrl.u = float(d.ctrl[0])
        elif self.phase == "turn":
            # ratchet cycle: the FAST wheel-speed change turns the body (its yaw torque
            # I_w a sin(zeta) beats the foot's torsional friction), the SLOW one does not.
            # Both stages stay at or below the start speed, so the clutch never engages.
            if self.turn_stage in ("fast", "slow") and abs(self.w_ref_s - self.turn_goal) < 1.0 \
                    and abs(d.qvel[self.wd] - self.turn_goal) < 40.0:
                if self.turn_stage == self.turn_order[0]:
                    self.turn_stage = self.turn_order[1]
                    self.turn_goal = self.turn_w0
                else:
                    self.turn_stage = None
                    self.turns += 1
                    self.phase, self.t_phase = ("settle" if self.mode == "hop" else "stick"), d.time
            elif d.time - self.t_phase > 12.0:
                self.turn_stage, self.phase, self.t_phase = None, ("settle" if self.mode == "hop" else "stick"), d.time
        elif self.phase == "settle":
            if self.mode == "stop" and not self.flip_request:
                self.mode, self.phase = "stick", "stick"
            elif self.mode == "stick":
                self.phase = "stick"
            elif quiet and d.time - self.t_phase > self.settle_time and not (
                    self.yaw_hold and (abs(self.heading_error()) > self.yaw_tol or self.yaw_mode == "turn")):
                self.phase, self.t_phase = "wind", d.time
                self.wind_for_flip = self.flip_request
        elif self.phase == "wind":
            if self.mode == "stick":
                self.phase = "stick"
            elif self.fwd and not self.wind_for_flip and self.kick_deg > 0:
                # start the lean doublet so that it ends exactly at the cam release
                w = d.qvel[self.wd]
                T = self.kick_T
                U = -self.fwd * np.deg2rad(self.kick_deg) / (self.b_ground_signed * (T / 2) ** 2)
                rem = ((pp.f_wind - self.cam_phase()) % 1.0) * 2 * np.pi * pp.G     # wheel rad to release
                need = w * T + abs(U) * T * T / (4 * self.I_w)
                if w > pp.w_clutch + 10.0 and rem <= need:
                    self.phase, self.t_phase, self.kick_U = "kick", d.time, float(U)
        elif self.phase == "stick" and self.mode == "hop":
            self.phase, self.t_phase = "settle", d.time
        elif (self.phase == "stick" and in_wind and d.qvel[self.wd] > pp.w_clutch
              and (pp.f_wind - self.cam_phase()) % 1.0 < pp.catch_window):
            # the clutch caught during a big balance transient: the cam is winding,
            # so fly this jump as a controlled hop and come back to stick
            self.mode, self.phase, self.t_phase = "stop", "wind", d.time
            self.w_ref_s = d.qvel[self.wd]
        # wheel-speed reference: above the clutch speed only while winding
        if self.phase == "wind" and self.fwd and not self.wind_for_flip and self.lean_deg > 0:
            # lean toward the travel direction for the last part of the wind. Travel
            # is along -y for fwd=+1: holding that lean spins the wheel UP, which
            # keeps the clutch in and brings the release sooner (the other lean
            # would slow the wheel below the clutch speed and stall the cam)
            to_rel = (pp.f_wind - self.cam_phase()) % 1.0
            cam_turning = d.qvel[self.wd] > pp.w_clutch + 30.0
            goal = self.fwd * np.deg2rad(self.lean_deg) if (to_rel < self.lean_frac and cam_turning) else 0.0
            step = np.deg2rad(self.lean_deg) / self.lean_ramp * self.Ts
            self.ctrl.x_ref[0] += np.clip(goal - self.ctrl.x_ref[0], -step, step)
        elif self.phase != "vault":
            self.ctrl.x_ref[:2] = 0.0
        if self.phase == "turn" and self.turn_stage is not None:
            target = self.turn_goal
            slew = self.turn_fast if self.turn_stage == "fast" else self.turn_slow
            self.w_ref_s += np.clip(target - self.w_ref_s, -slew * self.Ts, slew * self.Ts)
            self.ctrl.w_ref = self.w_ref_s
            return
        if self.phase == "wind":
            # flips wind (and so take off) with a slow wheel: the gyroscopic pitch
            # error of the flip grows with the wheel speed
            target = self.w_flip if self.wind_for_flip else (self.w_fwd if self.fwd else self.w_hop)
        elif self.mode == "hop" or self.flip_request:
            target = self.w_idle
        else:
            target = self.w_park                     # stick: keep the clutch well disengaged
        if self.phase == "flight":
            self.w_ref_s = d.qvel[self.wd]          # touch down with the wheel speed it has
        if self.yaw_hold and self.phase in ("stick", "settle") and not self.airborne:
            target = self.yaw_wheel_ref()           # steer by the balance wheel speed
        else:
            self.yaw_mode = "off"
        step = self.w_slew * self.Ts
        self.w_ref_s += np.clip(target - self.w_ref_s, -step, step)
        self.ctrl.w_ref = self.w_ref_s

    def state(self):
        a, b, R = self.tilt()
        return dict(t=self.d.time, alpha=np.rad2deg(a), beta=np.rad2deg(b), upright=R[2, 2],
                    wheel=self.d.qvel[self.wd], leg=self.d.qpos[self.lq], cam=self.cam_phase(),
                    airborne=self.airborne, jumps=self.jumps, mode=self.mode, phase=self.phase, z=self.d.xipos[self.body][2],
                    x=self.d.qpos[0], y=self.d.qpos[1], fallen=self.is_fallen(R), down=bool(self.has_stops and self.body_contact()),
                    energy_Wh=self.E_used / 3600.0)

    def is_fallen(self, R):
        if self.flipping and self.airborne:
            return False
        if self.has_stops:          # resting on the skids / battery pods is recoverable
            return bool(R[2, 2] < np.cos(np.deg2rad(40)))
        return bool(R[2, 2] < 0.7 or self.body_contact())
