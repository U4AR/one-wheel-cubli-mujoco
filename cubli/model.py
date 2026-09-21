"""MuJoCo model of the One-Wheel Cubli, generated from CubliParams.

Generalised coordinates match the paper (Appendix A):
    alpha (roll, about x), beta (pitch, about y), gamma (yaw, about z)  -> housing
    phi   (reaction wheel spin, about wheel axis e_x^D)
    delta1, delta2 (cantilever bending, about body z at Q)
The three housing hinges are stacked in one body in the order x -> y -> z, which
reproduces the paper's Euler chain I -x-> C -y-> B -z-> A exactly.
"""
import numpy as np
import mujoco

from .params import CubliParams, NOMINAL

# IMU positions in the body frame (m): four in a square next to the pivot and one
# far away, the variance-minimising layout of Sec. 3.3 (exact coordinates ASSUMED).
IMU_POS = np.array([
    [0.03, 0.03, 0.03],
    [-0.03, 0.03, 0.03],
    [-0.03, -0.03, 0.03],
    [0.03, -0.03, 0.03],
    [0.0, 0.0, 0.21],
])

JOINTS = ["alpha", "beta", "gamma", "phi", "delta1", "delta2"]


def build_xml(p: CubliParams = NOMINAL, timestep=5e-4, ground_limits=False) -> str:
    """ground_limits: add joint limits where the real system would hit the floor
    (an end mass at |beta| ~ 17 deg, a housing corner at |alpha| ~ 38 deg), so a
    fall ends lying on the ground instead of rotating through it (viewer only)."""
    if ground_limits:
        lim_a = f'limited="true" range="{-np.deg2rad(38)} {np.deg2rad(38)}"'
        lim_b = f'limited="true" range="{-np.deg2rad(17)} {np.deg2rad(17)}"'
    else:
        lim_a = lim_b = 'limited="false"'

    cx, cy = p.com_offset_xy
    ring = p.layout == "ring"
    if ring:
        m_h, l_S, I_hx, I_hy, I_hz = p.housing_without_tube()
    else:
        m_h, l_S, I_hx, I_hy, I_hz = p.m_h, p.l_S, p.I_hx, p.I_hy, p.I_hz
    # housing inertia about its CoM (parallel-axis theorem from the pivot values)
    Ihx = I_hx - m_h * (l_S**2 + cy**2)
    Ihy = I_hy - m_h * (l_S**2 + cx**2)
    Ihz = I_hz - m_h * (cx**2 + cy**2)
    zr = p.core_raise
    wq = f"{np.cos(p.eta/2)} 0 0 {np.sin(p.eta/2)}"  # rotate D about z by eta
    # wheel frame: rotate by eta about z, then tilt the spin axis up by wheel_tilt
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.array([np.cos(p.eta/2), 0, 0, np.sin(p.eta/2)]),
                       np.array([np.cos(-p.wheel_tilt/2), 0, np.sin(-p.wheel_tilt/2), 0]))
    wheel_q = " ".join(f"{v}" for v in q)
    tiny = 1e-8
    # The paper's wheel inertia includes the motor rotor, so I_wx > I_wy + I_wz,
    # which a single rigid body cannot have. Inflate the transverse inertia just
    # enough to be physical (+10% of a term that is ~1% of the housing inertia).
    Iwt = max(p.I_wy, 0.5 * p.I_wx * 1.0001)

    # open cube frame (edge length 0.15) standing on a pointed tip at the pivot
    h, z0 = 0.075, p.l_P
    corners = [(sx*h, sy*h, z0+sz*h) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    edges = [(a, b) for i, a in enumerate(corners) for b in corners[i+1:]
             if sum(abs(u - v) > 1e-9 for u, v in zip(a, b)) == 1]
    edges += [((0, 0, 0.004), c) for c in corners if c[2] < z0]
    frame_geoms = "\n".join(
        f'      <geom type="capsule" fromto="{a[0]} {a[1]} {a[2]} {b[0]} {b[1]} {b[2]}" size="0.005" material="carbon" contype="0" conaffinity="0"/>'
        for a, b in edges)
    if ring:
        R, c, M = p.ring_radius, p.ring_center, p.ring_mass
        n_seg = 64
        pts = [(R * np.sin(2 * np.pi * i / n_seg), c - R * np.cos(2 * np.pi * i / n_seg))
               for i in range(n_seg + 1)]
        ring_geoms = "\n".join(
            f'        <geom type="capsule" fromto="{x0} 0 {z0 - c} {x1} 0 {z1 - c}" size="0.008" material="mass" contype="0" conaffinity="0"/>'
            for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]))
        hub = zr + p.l_P                      # spokes from the housing to the hoop
        spokes = "\n".join(
            f'        <geom type="capsule" fromto="0 0 {hub - c} {R * np.sin(t)} 0 {-R * np.cos(t)}" size="0.003" material="carbon" contype="0" conaffinity="0"/>'
            for t in np.deg2rad([60, 120, 180, 240, 300]))
        mw_ = p.ring_weights            # point masses at (+-R, 0, 0) from the hoop centre
        Mt = M + 2 * mw_
        wgeoms = "\n".join(
            f'        <geom type="sphere" pos="{sx * R} 0 0" size="0.03" material="alu" contype="0" conaffinity="0"/>'
            for sx in (-1, 1)) if mw_ > 0 else ""
        mass_bodies = f"""      <body name="ring" pos="0 0 {c}">
        <inertial pos="0 0 0" mass="{Mt}" diaginertia="{M * R**2 / 2 + 1e-9} {M * R**2 + 2 * mw_ * R**2} {M * R**2 / 2 + 2 * mw_ * R**2}"/>
{wgeoms}
{ring_geoms}
{spokes}
      </body>"""
    else:
        mass_bodies = f"""      <body name="endmass1" pos="0 0 {p.l_Q}">
        <joint name="delta1" type="hinge" axis="0 0 1" stiffness="{p.k}" damping="{p.d}" limited="false"/>
        <inertial pos="{-p.l_E} 0 0" mass="{p.m_e}" diaginertia="{tiny} {tiny} {tiny}"/>
        <geom type="capsule" fromto="0 0 0 {-p.l_E} 0 0" size="0.012" material="carbon" contype="0" conaffinity="0"/>
        <geom type="cylinder" pos="{-p.l_E} 0 0" size="0.03 0.025" euler="0 1.5708 0" material="mass" contype="0" conaffinity="0"/>
      </body>
      <body name="endmass2" pos="0 0 {p.l_Q}">
        <joint name="delta2" type="hinge" axis="0 0 1" stiffness="{p.k}" damping="{p.d}" limited="false"/>
        <inertial pos="{p.l_E} 0 0" mass="{p.m_e}" diaginertia="{tiny} {tiny} {tiny}"/>
        <geom type="capsule" fromto="0 0 0 {p.l_E} 0 0" size="0.012" material="carbon" contype="0" conaffinity="0"/>
        <geom type="cylinder" pos="{p.l_E} 0 0" size="0.03 0.025" euler="0 1.5708 0" material="mass" contype="0" conaffinity="0"/>
      </body>"""
    if ring and p.pole != "none":
        mt = 1560.0 * 7.225e-5 * 2 * p.l_E
        mp = 2 * p.m_e + mt
        Iyy = 2 * p.m_e * p.l_E**2 + mt * (2 * p.l_E)**2 / 12
        joints = ("" if p.pole == "rigid" else
                  '        <joint name="pole_r" type="hinge" axis="1 0 0" limited="false" damping="0.002"/>\n'
                  '        <joint name="pole_p" type="hinge" axis="0 1 0" limited="false" damping="0.002"/>\n')
        mass_bodies += f"""
      <body name="pole" pos="0 0 {p.pole_mount}">
{joints}        <inertial pos="0 0 {-p.pole_drop}" mass="{mp}" diaginertia="1e-5 {Iyy} {Iyy}"/>
        <geom type="capsule" fromto="{-p.l_E} 0 {-p.pole_drop} {p.l_E} 0 {-p.pole_drop}" size="0.012" material="carbon" contype="0" conaffinity="0"/>
        <geom type="cylinder" pos="{-p.l_E} 0 {-p.pole_drop}" size="0.03 0.025" euler="0 1.5708 0" material="wheel" contype="0" conaffinity="0"/>
        <geom type="cylinder" pos="{p.l_E} 0 {-p.pole_drop}" size="0.03 0.025" euler="0 1.5708 0" material="wheel" contype="0" conaffinity="0"/>
        <geom type="sphere" size="0.018" material="alu" contype="0" conaffinity="0"/>
      </body>"""
    tip_geom = (f'      <geom type="capsule" fromto="0 0 0.004 0 0 {zr + 0.06}" size="0.006" material="alu" contype="0" conaffinity="0"/>'
                if zr > 0 else "")
    extra_body = ""
    if p.core_extra_mass > 0:   # bigger motor: extra mass at the wheel centre
        me = p.core_extra_mass
        extra_body = (f'        <body name="motor_extra" pos="0 0 {p.l_P}">\n'
                      f'          <inertial pos="0 0 0" mass="{me}" diaginertia="{me * 1e-3} {me * 1e-3} {me * 1e-3}"/>\n'
                      f'        </body>')
    target_z = max(0.17, (p.ring_center * 0.8) if ring else 0.17)
    imu_sites = "\n".join(
        f'          <site name="imu{i}" pos="{x} {y} {z}" size="0.006" rgba="0.1 0.8 0.2 1"/>'
        for i, (x, y, z) in enumerate(IMU_POS))
    imu_sensors = "\n".join(
        f'    <accelerometer name="acc{i}" site="imu{i}"/>\n'
        f'    <gyro name="gyr{i}" site="imu{i}"/>' for i in range(len(IMU_POS)))

    return f"""
<mujoco model="one_wheel_cubli">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep}" integrator="RK4" gravity="0 0 {-p.g0}"/>
  <!-- every geom is visual only; all mass comes from explicit <inertial> elements -->
  <default><geom density="0"/></default>
  <visual>
    <global offwidth="1280" offheight="720" azimuth="135" elevation="-15"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.65 0.8" rgb2="0.1 0.12 0.18" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".92 .92 .92" rgb2=".82 .82 .84" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <material name="carbon" rgba="0.12 0.12 0.13 1"/>
    <material name="alu" rgba="0.72 0.74 0.78 1"/>
    <material name="wheel" rgba="0.85 0.35 0.1 1"/>
    <material name="mass" rgba="0.2 0.35 0.8 1"/>
  </asset>
  <worldbody>
    <light pos="0.5 -0.5 2.0" dir="-0.2 0.2 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-1 1 1.5" dir="0.5 -0.5 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <camera name="front" pos="0.95 0.8 0.5" mode="targetbody" target="target"/>
    <camera name="side_beta" pos="0 -1.9 0.35" xyaxes="1 0 0 0 0 1"/>
    <body name="target" pos="0 0 {target_z}"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="grid" contype="0" conaffinity="0"/>
    <body name="housing" pos="0 0 0">
      <joint name="alpha" type="hinge" axis="1 0 0" {lim_a} damping="0"/>
      <joint name="beta"  type="hinge" axis="0 1 0" {lim_b}/>
      <joint name="gamma" type="hinge" axis="0 0 1" limited="false" damping="{p.yaw_damping}" frictionloss="{p.yaw_friction}"/>
      <inertial pos="{cx} {cy} {l_S + zr}" mass="{m_h}" diaginertia="{Ihx} {Ihy} {Ihz}"/>
      <!-- visuals only (inertial above overrides geom mass) -->
{tip_geom}
      <body name="core" pos="0 0 {zr}">
{frame_geoms}
        <geom type="cylinder" fromto="0 0 {p.l_Q-0.02} 0 0 {p.l_Q+0.005}" size="0.018" material="alu" contype="0" conaffinity="0"/>
        <geom type="box" pos="0 0 {p.l_P}" size="0.003 0.05 0.05" quat="{wq}" material="alu" contype="0" conaffinity="0"/>
{imu_sites}
{extra_body}
        <body name="wheel" pos="0 0 {p.l_P}" quat="{wheel_q}">
          <joint name="phi" type="hinge" axis="1 0 0" limited="false"/>
          <inertial pos="0 {p.wheel_ecc} 0" mass="{p.m_w}" diaginertia="{p.I_wx} {Iwt} {Iwt}"/>
          <geom type="cylinder" fromto="0.012 0 0 0.024 0 0" size="0.068" material="wheel" contype="0" conaffinity="0"/>
          <geom type="box" pos="0.0185 0 0.04" size="0.007 0.01 0.02" rgba="1 1 1 1" contype="0" conaffinity="0"/>
        </body>
      </body>
{mass_bodies}
    </body>
  </worldbody>
  <actuator>
    <!-- motor torque T_m on the wheel; reaction acts on the housing -->
    <motor name="motor" joint="phi" gear="1" ctrlrange="{-p.tau_peak} {p.tau_peak}"/>
  </actuator>
  <sensor>
{imu_sensors}
    <jointvel name="wheel_speed" joint="phi"/>
  </sensor>
</mujoco>
"""


def load(p: CubliParams = NOMINAL, timestep=5e-4, ground_limits=False):
    m = mujoco.MjModel.from_xml_string(build_xml(p, timestep, ground_limits))
    d = mujoco.MjData(m)
    return m, d
