"""Rolling hoop: a circular hoop that can roll on the floor, balanced sideways by the
One-Wheel Cubli's single reaction wheel (not in the paper).

Mechanism ("hub" design): the hoop turns freely on a bearing around a central
axle; the housing (with the reaction wheel) hangs from the axle like a pendulum,
so it stays upright however far the hoop rolls.
  * lean (sideways fall about the contact line) -- the only unstable motion;
    countered by the wheel torque's lean component (cos eta);
  * housing pitch (pendulum about the axle) -- stable; the wheel torque's pitch
    component (sin eta) swings it, which shifts the centre of mass and rolls the
    hoop (pendulum drive);
  * rolling -- neutral.
Coordinates of the linear model (upright, at rest):
  phi   lean of the axle/hoop about the rolling direction (world x)
  theta hoop rolling angle (hub moves x = R*theta)
  zeta  housing pitch about the axle (absolute)
  psi   reaction-wheel spin relative to the housing
"""
from dataclasses import dataclass, replace
import numpy as np
import mujoco
from scipy.linalg import expm, solve_discrete_are

from .params import NOMINAL
from .model import IMU_POS


@dataclass(frozen=True)
class RollingParams:
    R: float = 0.32            # hoop radius (m)
    M_hoop: float = 0.74       # hoop mass (kg), uniform thin hoop in the x-z plane
    d: float = 0.05            # housing CoM below the axle (m)
    # final design (scripts/rolling_hoop.py): wheel axis along the lean axis
    # (eta = 0, the paper's 45 deg only adds a harmful pitch component here, since
    # rolling cannot be driven anyway), tilted up 7 deg so its vertical component
    # can park the hoop's turning momentum -- needed to slow through the critical
    # (self-stabilising) speed without spiralling over
    eta: float = 0.0           # reaction-wheel axis angle from the lean axis
    wheel_tilt: float = np.deg2rad(7)   # wheel axis tilted up out of the horizontal plane
    bearing_damping: float = 2e-3   # N m s/rad, hub bearing
    mu: float = 1.0            # floor sliding friction
    mu_roll: float = 2e-4      # rolling friction (m)
    mu_spin: float = 1e-4      # torsional friction (m): thin rim, ~0.1 mm contact patch

    def with_(self, **kw):
        return replace(self, **kw)

    # housing = paper housing without the cantilever tube, plus the wheel as a rigid
    # part (its spin inertia is handled separately)
    def housing(self):
        m_h, lS, Ix, Iy, Iz = NOMINAL.housing_without_tube()
        Ixc, Iyc, Izc = Ix - m_h * lS**2, Iy - m_h * lS**2, Iz
        return m_h, (Ixc, Iyc, Izc)


P = NOMINAL          # motor / wheel parameters from the paper


def _quat_axis_angle(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])


def build_xml(rp: RollingParams = RollingParams(), timestep=5e-4):
    R, M, d = rp.R, rp.M_hoop, rp.d
    m_h, (Ixc, Iyc, Izc) = rp.housing()
    # the housing's geometric centre sits on the axle; its CoM hangs d below it
    n_seg = 120
    pts = [(R * np.sin(2 * np.pi * i / n_seg), -R * np.cos(2 * np.pi * i / n_seg)) for i in range(n_seg + 1)]
    ring = "\n".join(
        f'      <geom type="capsule" fromto="{x0} 0 {z0} {x1} 0 {z1}" size="0.01" material="mass" '
        f'contype="1" conaffinity="1" condim="6" friction="{rp.mu} {rp.mu_spin} {rp.mu_roll}"/>'
        for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]))
    spokes = "\n".join(
        f'      <geom type="capsule" fromto="0 0 0 {R * np.cos(t)} 0 {R * np.sin(t)}" size="0.003" material="carbon"/>'
        for t in np.linspace(0, 2 * np.pi, 8, endpoint=False))
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, _quat_axis_angle([0, 0, 1], rp.eta),
                       _quat_axis_angle([0, 1, 0], -rp.wheel_tilt))   # tilt the spin axis up
    wq = " ".join(map(str, q))
    Ihx, Ihy = M * R**2 / 2, M * R**2           # thin hoop about its centre (x-z plane)
    h, zc = 0.075, -d + 0.0                     # housing cube half size; cube centre
    corners = [(sx * h, sy * h, zc + sz * h) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    edges = [(a, b) for i, a in enumerate(corners) for b in corners[i + 1:]
             if sum(abs(u - v) > 1e-9 for u, v in zip(a, b)) == 1]
    frame = "\n".join(
        f'        <geom type="capsule" fromto="{a[0]} {a[1]} {a[2]} {b[0]} {b[1]} {b[2]}" size="0.005" material="carbon"/>'
        for a, b in edges)
    imu = "\n".join(f'        <site name="imu{i}" pos="{x} {y} {z - 0.1}" size="0.006" rgba="0.1 0.8 0.2 1"/>'
                    for i, (x, y, z) in enumerate(IMU_POS))
    return f"""
<mujoco model="rolling_hoop">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep}" integrator="implicitfast" gravity="0 0 {-P.g0}" cone="elliptic"/>
  <default><geom density="0" contype="0" conaffinity="0"/></default>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.65 0.8" rgb2="0.1 0.12 0.18" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".78 .78 .78" rgb2=".66 .66 .68" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="40 40" reflectance="0.05"/>
    <material name="carbon" rgba="0.12 0.12 0.13 1"/>
    <material name="alu" rgba="0.72 0.74 0.78 1"/>
    <material name="wheel" rgba="0.85 0.35 0.1 1"/>
    <material name="mass" rgba="0.2 0.35 0.8 1"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 5" dir="-0.3 0.3 -1" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1" castshadow="false"/>
    <light directional="true" pos="0 0 5" dir="0.4 -0.3 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="20 20 0.1" material="grid" contype="1" conaffinity="1"
          condim="6" friction="{rp.mu} {rp.mu_spin} {rp.mu_roll}"/>
    <body name="hoop" pos="0 0 {R + 0.0101}">
      <freejoint name="free"/>
      <inertial pos="0 0 0" mass="{M}" diaginertia="{Ihx} {Ihy} {Ihx}"/>
{ring}
{spokes}
      <geom type="cylinder" fromto="0 -0.03 0 0 0.03 0" size="0.012" material="alu"/>
      <body name="housing" pos="0 0 0">
        <joint name="axle" type="hinge" axis="0 1 0" damping="{rp.bearing_damping}"/>
        <inertial pos="0 0 {-d}" mass="{m_h}" diaginertia="{Ixc} {Iyc} {Izc}"/>
{frame}
{imu}
        <body name="wheel" pos="0 0 {zc}" quat="{wq}">
          <joint name="phi" type="hinge" axis="1 0 0"/>
          <inertial pos="0 0 0" mass="{P.m_w}" diaginertia="{P.I_wx} {max(P.I_wy, 0.5 * P.I_wx * 1.0001)} {max(P.I_wy, 0.5 * P.I_wx * 1.0001)}"/>
          <geom type="cylinder" fromto="0.012 0 0 0.024 0 0" size="0.068" material="wheel"/>
          <geom type="box" pos="0.0185 0 0.04" size="0.007 0.01 0.02" rgba="1 1 1 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="phi" gear="1" ctrlrange="{-P.tau_peak} {P.tau_peak}"/>
  </actuator>
</mujoco>
"""


def load(rp: RollingParams = RollingParams(), timestep=5e-4):
    m = mujoco.MjModel.from_xml_string(build_xml(rp, timestep))
    return m, mujoco.MjData(m)


# ---------------------------------------------------------------------------
# linear model about upright rest (Lagrangian, rolling without slip, small angles)
STATES = ["phi", "phi_d", "zeta", "zeta_d", "theta", "theta_d", "psi_d"]


def linear_model(rp: RollingParams = RollingParams()):
    R, M, d, eta = rp.R, rp.M_hoop, rp.d, rp.eta
    m_h, (Icx, Icy, _) = rp.housing()
    mw, Iw = P.m_w, P.I_wx
    Iwt = max(P.I_wy, 0.5 * P.I_wx * 1.0001)
    # housing + wheel (rigid part) about the housing CoM; wheel centre at -d too
    mc = m_h + mw
    c, s = np.cos(eta), np.sin(eta)
    Icx_t = Icx + Iwt * s**2                  # wheel transverse inertia about x / y
    Icy_t = Icy + Iwt * c**2
    Ihx, Ihy = M * R**2 / 2, M * R**2
    # mass matrix for q = (phi, theta, zeta, psi); wheel spin inertia on psi + projections
    Mm = np.zeros((4, 4))
    Mm[0, 0] = M * R**2 + Ihx + mc * (R - d)**2 + Icx_t + Iw * c**2
    Mm[1, 1] = M * R**2 + Ihy + mc * R**2
    Mm[2, 2] = mc * d**2 + Icy_t + Iw * s**2
    Mm[3, 3] = Iw
    Mm[1, 2] = Mm[2, 1] = -mc * R * d
    Mm[0, 3] = Mm[3, 0] = Iw * c
    Mm[2, 3] = Mm[3, 2] = Iw * s
    Mm[0, 2] = Mm[2, 0] = Iw * c * s
    K = np.diag([-(M * R + mc * (R - d)) * P.g0, 0.0, mc * P.g0 * d, 0.0])
    Dm = np.zeros((4, 4))                      # bearing damping between zeta and theta
    b = rp.bearing_damping
    Dm[1, 1] = Dm[2, 2] = b; Dm[1, 2] = Dm[2, 1] = -b
    Bq = np.array([0, 0, 0, 1.0])              # motor torque on the relative wheel coordinate
    Mi = np.linalg.inv(Mm)
    # state x = (phi, phi_d, zeta, zeta_d, theta, theta_d, psi_d)
    qi = {"phi": 0, "theta": 1, "zeta": 2, "psi": 3}
    A = np.zeros((7, 7)); B = np.zeros((7, 1))
    pos = {"phi": 0, "zeta": 2, "theta": 4}
    vel = {"phi": 1, "zeta": 3, "theta": 5, "psi": 6}
    for n, i in pos.items():
        A[i, vel[n]] = 1.0
    acc_K, acc_D, acc_B = -Mi @ K, -Mi @ Dm, Mi @ Bq
    for n, row in vel.items():
        r = qi[n]
        for m_, col in pos.items():
            A[row, col] += acc_K[r, qi[m_]]
        for m_, col in vel.items():
            A[row, col] += acc_D[r, qi[m_]]
        B[row, 0] = acc_B[r]
    return A, B


def c2d(A, B, Ts):
    n = A.shape[0]
    Mx = np.zeros((n + 1, n + 1)); Mx[:n, :n] = A; Mx[:n, n:] = B
    E = expm(Mx * Ts)
    return E[:n, :n], E[:n, n:]


# ---------------------------------------------------------------------------
def measure(m, d):
    """True state (phi, phi_d, zeta, zeta_d, theta_rate) from the simulator, in the
    linear model's convention. theta itself is integrated by the caller."""
    hb, hs = m.body("hoop").id, m.body("housing").id
    Rh = d.xmat[hb].reshape(3, 3); Rs = d.xmat[hs].reshape(3, 3)
    a = Rh[:, 1]                                     # axle direction
    zw = np.array([0.0, 0.0, 1.0])
    f = np.cross(a, zw); f /= np.linalg.norm(f)      # rolling direction (heading)
    phi = np.arcsin(np.clip(a @ zw, -1, 1))
    h = Rs[:, 2]                                     # housing "up"
    zeta = np.arctan2(h @ f, h @ zw)
    w_hoop = d.cvel[hb][:3] if False else None
    # world angular velocities from qvel: free joint (world frame for rotation? no:
    # MuJoCo free-joint angular velocity is in the body frame)
    wb = Rh @ d.qvel[3:6]
    w_house = wb + a * d.qvel[m.joint("axle").dofadr[0]]
    return np.array([phi, wb @ f, zeta, w_house @ a, wb @ a,
                     d.qvel[m.joint("phi").dofadr[0]]])


class RollingController:
    """Discrete LQR on the 7-state model with one-step delay compensation.
    Modes: hold position (theta_ref fixed) or track a rolling speed v_ref."""

    SCALE = np.array([np.deg2rad(1), np.deg2rad(10), np.deg2rad(5), np.deg2rad(20),
                      1.0, 1.0, 50.0])

    def __init__(self, rp: RollingParams = RollingParams(), Ts=0.01,
                 Q=(1, 0.1, 0.3, 0.1, 1e-6, 1e-4, 0.05), Rw=20.0):
        # Rolling (theta) is essentially uncontrollable: the wheel pushes only on the
        # hanging housing and the hoop turns on a free bearing, so the hoop's rolling
        # momentum is conserved apart from friction (it can only coast). Its weights
        # are ~0 so the controller lets it roll instead of fighting it.
        self.rp, self.Ts = rp, Ts
        A, B = linear_model(rp)
        self.Ad, self.Bd = c2d(A, B, Ts)
        S = np.diag(self.SCALE)
        An, Bn = np.linalg.solve(S, self.Ad @ S), np.linalg.solve(S, self.Bd)
        P_ = solve_discrete_are(An, Bn, np.diag(Q), np.array([[Rw]]))
        Kn = np.linalg.solve(Rw + Bn.T @ P_ @ Bn, Bn.T @ P_ @ An)
        self.K = Kn @ np.linalg.inv(S)
        self.reset()

    def reset(self):
        self.u = 0.0
        self.x_prev = None
        self.theta_ref = None
        self.v_ref = 0.0
        self.hold = True

    def step(self, x_meas_delayed):
        """x_meas_delayed: measured state from the previous control step."""
        x = np.asarray(x_meas_delayed, float)
        xhat = self.Ad @ x + self.Bd[:, 0] * self.u           # predict to now
        if self.theta_ref is None:
            self.theta_ref = xhat[4]
        if self.hold:
            ref_theta, ref_rate = self.theta_ref, 0.0
        else:
            self.theta_ref = xhat[4]                           # velocity mode
            ref_theta, ref_rate = xhat[4], self.v_ref / self.rp.R
        e = xhat.copy()
        e[4] -= ref_theta
        e[5] -= ref_rate
        self.u = float(-(self.K @ e)[0])
        return self.u

    def applied(self, u):
        self.u = u


# ---------------------------------------------------------------------------
@dataclass
class RollingSensors:
    """Measurement errors on the controller's inputs (ASSUMED; a rolling-contact
    tilt estimator is not implemented -- these stand in for its output)."""
    angle_std: float = np.deg2rad(0.03)
    angle_bias_std: float = np.deg2rad(0.1)
    rate_std: float = 0.005
    wheel_std: float = 0.3
    theta_std: float = 0.002      # hub encoder + housing pitch


def simulate(rp=RollingParams(), T=20.0, v_ref=lambda t: None, pushes=(), lean0_deg=1.0, v0=0.0,
             sensors=RollingSensors(), seed=0, Ts=0.01, dt=5e-4, Q=None, frames=None,
             frame_every=None, renderer=None, camera=None):
    """v_ref(t): None = hold position, else rolling speed (m/s).
    pushes: [(t0, dur, body, force_world(3), point_body(3) or None)]."""
    from .sim import Motor
    rng = np.random.default_rng(seed)
    m, d = load(rp, dt)
    q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([1.0, 0, 0]), np.deg2rad(lean0_deg))
    d.qpos[3:7] = q
    d.qpos[2] = rp.R * np.cos(np.deg2rad(lean0_deg)) + 0.0101
    d.qvel[0] = v0                        # rolling start: hub speed (world x) ...
    d.qvel[4] = v0 / (rp.R + 0.01)        # ... and spin about the axle (body y);
    d.qvel[m.joint("axle").dofadr[0]] = -v0 / (rp.R + 0.01)   # housing hangs still
    mujoco.mj_forward(m, d)
    ctrl = RollingController(rp, Ts) if Q is None else RollingController(rp, Ts, Q=Q)
    motor = Motor(P)
    bias = np.array([rng.normal(0, sensors.angle_bias_std), 0, rng.normal(0, sensors.angle_bias_std), 0])
    theta = 0.0
    ids = {b: m.body(b).id for _, _, b, _, _ in pushes}

    def meas():
        x = measure(m, d)
        x_ = np.array([x[0], x[1], x[2], x[3], theta, x[4], x[5]])
        noise = np.array([sensors.angle_std, sensors.rate_std, sensors.angle_std, sensors.rate_std,
                          sensors.theta_std, sensors.rate_std, sensors.wheel_std])
        x_[:4] += bias
        return x_ + rng.normal(0, 1, 7) * noise, x_

    y_prev, _ = meas()
    log = []
    fell = False
    for k in range(int(T / Ts)):
        t = d.time
        vr = v_ref(t)
        ctrl.hold = vr is None
        ctrl.v_ref = 0.0 if vr is None else vr
        u, lim = motor.limit(ctrl.step(y_prev), d.qvel[m.joint("phi").dofadr[0]], Ts)
        ctrl.applied(u)
        y_prev, x_true = meas()
        d.ctrl[0] = u
        for _ in range(int(round(Ts / dt))):
            d.xfrc_applied[:] = 0
            for t0, dur, b, F, pt in pushes:
                if t0 <= d.time < t0 + dur:
                    bid = ids[b]
                    d.xfrc_applied[bid, :3] += F
                    if pt is not None:
                        pw = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ np.asarray(pt)
                        d.xfrc_applied[bid, 3:] += np.cross(pw - d.xipos[bid], F)
            mujoco.mj_step(m, d)
            theta += measure(m, d)[4] * dt
            if frames is not None and renderer is not None and int(round(d.time / dt)) % frame_every == 0:
                renderer.update_scene(d, camera=camera)
                frames.append(renderer.render())
        log.append([d.time, *x_true, u, lim, d.qpos[0], d.qpos[1]])
        if abs(x_true[0]) > np.deg2rad(30):
            fell = True
            break
    L = np.array(log)
    # columns: t, phi, phi_d, zeta, zeta_d, theta, theta_d, psi_d, u, ulim, x, y
    return L, fell
