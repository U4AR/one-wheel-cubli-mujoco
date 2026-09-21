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


def build_xml(p: CubliParams = NOMINAL, timestep=5e-4) -> str:
    cx, cy = p.com_offset_xy
    # housing inertia about its CoM (parallel-axis theorem from the pivot values)
    Ihx = p.I_hx - p.m_h * (p.l_S**2 + cy**2)
    Ihy = p.I_hy - p.m_h * (p.l_S**2 + cx**2)
    Ihz = p.I_hz - p.m_h * (cx**2 + cy**2)
    wq = f"{np.cos(p.eta/2)} 0 0 {np.sin(p.eta/2)}"  # rotate D about z by eta
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
    imu_sites = "\n".join(
        f'        <site name="imu{i}" pos="{x} {y} {z}" size="0.006" rgba="0.1 0.8 0.2 1"/>'
        for i, (x, y, z) in enumerate(IMU_POS))
    imu_sensors = "\n".join(
        f'    <accelerometer name="acc{i}" site="imu{i}"/>\n'
        f'    <gyro name="gyr{i}" site="imu{i}"/>' for i in range(len(IMU_POS)))

    return f"""
<mujoco model="one_wheel_cubli">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep}" integrator="RK4" gravity="0 0 {-p.g0}"/>
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
    <body name="target" pos="0 0 0.17"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="grid" contype="0" conaffinity="0"/>
    <body name="housing" pos="0 0 0">
      <joint name="alpha" type="hinge" axis="1 0 0" limited="false"/>
      <joint name="beta"  type="hinge" axis="0 1 0" limited="false"/>
      <joint name="gamma" type="hinge" axis="0 0 1" limited="false" damping="{p.yaw_damping}"/>
      <inertial pos="{cx} {cy} {p.l_S}" mass="{p.m_h}" diaginertia="{Ihx} {Ihy} {Ihz}"/>
      <!-- visuals only (inertial above overrides geom mass) -->
{frame_geoms}
      <geom type="cylinder" fromto="0 0 {p.l_Q-0.02} 0 0 {p.l_Q+0.005}" size="0.018" material="alu" contype="0" conaffinity="0"/>
      <geom type="box" pos="0 0 {p.l_P}" size="0.003 0.05 0.05" quat="{wq}" material="alu" contype="0" conaffinity="0"/>
{imu_sites}
      <body name="wheel" pos="0 0 {p.l_P}" quat="{wq}">
        <joint name="phi" type="hinge" axis="1 0 0" limited="false"/>
        <inertial pos="0 0 0" mass="{p.m_w}" diaginertia="{p.I_wx} {Iwt} {Iwt}"/>
        <geom type="cylinder" fromto="0.012 0 0 0.024 0 0" size="0.068" material="wheel" contype="0" conaffinity="0"/>
        <geom type="box" pos="0.0185 0 0.04" size="0.007 0.01 0.02" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      </body>
      <body name="endmass1" pos="0 0 {p.l_Q}">
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
      </body>
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


def load(p: CubliParams = NOMINAL, timestep=5e-4):
    m = mujoco.MjModel.from_xml_string(build_xml(p, timestep))
    d = mujoco.MjData(m)
    return m, d
