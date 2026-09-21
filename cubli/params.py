"""Physical parameters of the One-Wheel Cubli.

All values are from Table A.1 of
  M. Hofer, M. Muehlebach, R. D'Andrea, "The One-Wheel Cubli: A 3D inverted
  pendulum that can balance with a single reaction wheel", Mechatronics 91 (2023).
unless marked ASSUMED (not given in the paper).
"""
from dataclasses import dataclass, field, replace
import numpy as np


@dataclass(frozen=True)
class CubliParams:
    # geometry (m), measured along the body z-axis from the pivot O
    l_P: float = 0.133        # pivot -> reaction wheel centre P
    l_S: float = 0.155        # pivot -> housing centre of mass S
    l_Q: float = 0.214        # pivot -> cantilever attachment Q
    l_E: float = 0.5975       # Q -> end masses E1, E2 (along body x)
    eta: float = np.pi / 4    # wheel axis angle w.r.t. body x, about body z
    # --- design variations (not in the paper; 0 = paper design) -----------------
    wheel_tilt: float = 0.0   # rad, wheel axis tilted up out of the x-y plane
    wheel_ecc: float = 0.0    # m, radial offset of the wheel's centre of mass (unbalance)
    # --- "ring" variant: replace the cantilever + end masses by a thin hoop in the
    # pitch (x-z) plane. The carbon tube (App. C: A = 7.225e-5 m^2, rho = 1560,
    # 2*l_E long, at height l_Q) is removed from the housing's mass/inertia.
    layout: str = "bar"       # "bar" (paper) or "ring"
    ring_mass: float = 0.0    # kg
    ring_radius: float = 0.0  # m, horizontal semi-axis (radius for a circle)
    ring_b: float = 0.0       # m, vertical semi-axis (0 = circle, same as ring_radius)
    ring_center: float = 0.0  # m, height of the hoop centre above the pivot
    core_raise: float = 0.0   # m, housing (+wheel, IMUs) raised above the pivot
    ring_weights: float = 0.0 # kg, each of two point masses on the hoop at 3 and 9 o'clock
    core_extra_mass: float = 0.0  # kg, extra motor mass in the housing (at the wheel centre)
    # balancing pole on the ring variant: the paper's tube + end masses, mounted at
    # height pole_mount (above the pivot, in the body frame)
    pole: str = "none"        # "none" | "rigid" | "gimbal" (2-axis gimbal, stays level)
    pole_mount: float = 0.0   # m
    pole_drop: float = 0.0    # m, pole CoM below the gimbal centre (pendulous if > 0)

    def scaled_actuator(self, k_motor=1.0, k_wheel=1.0, motor_mass=0.36):
        """Bigger motor (torques x k_motor, motor mass grows proportionally) and
        heavier flywheel (inertia and mass x k_wheel, same radius)."""
        return self.with_(tau_peak=self.tau_peak * k_motor, tau_cont=self.tau_cont * k_motor,
                          tau_stall=self.tau_stall * k_motor,
                          i2t_budget=self.i2t_budget * k_motor**2,
                          core_extra_mass=motor_mass * (k_motor - 1.0),
                          m_w=self.m_w * k_wheel, I_wx=self.I_wx * k_wheel,
                          I_wy=self.I_wy * k_wheel, I_wz=self.I_wz * k_wheel)

    # masses (kg)
    m_h: float = 1.101        # housing (incl. motor, electronics, beam)
    m_w: float = 0.228        # reaction wheel
    m_e: float = 0.304        # one end mass

    # housing inertia about the PIVOT O, body frame (kg m^2)
    I_hx: float = 2.992e-2
    I_hy: float = 5.836e-2
    I_hz: float = 3.023e-2
    # wheel inertia about P, wheel frame D (x = spin axis; incl. rotor)
    I_wx: float = 9.044e-4
    I_wy: float = 4.117e-4
    I_wz: float = 4.117e-4

    # cantilever bending (torsional spring/damper about body z at Q)
    k: float = 1.080e4        # N m / rad  (f0 = 57.2 Hz, identified)
    d: float = 0.435          # N m s / rad

    g0: float = 9.81

    # --- actuator: Maxon EC 60 flat 200 W @ 48 V + EPOS4 50/15 -------------
    tau_peak: float = 3.4     # N m, peak (paper)
    tau_cont: float = 0.5     # N m, continuous (paper)
    w_noload: float = 450.0   # rad/s, ASSUMED no-load speed (~4300 rpm @ 48 V)
    tau_stall: float = 8.0    # N m, ASSUMED stall torque (sets back-EMF envelope)
    i2t_budget: float = 11.0  # N^2 m^2 s, ASSUMED I^2t overload budget
                              # (~1 s at peak torque before derating)

    # --- things the real system has that the ideal CAD model does not ------
    # (used to make the simulation honest, all ASSUMED)
    com_offset_xy: tuple = (0.002, 0.002)  # m, CoM shift toward the motor
    yaw_damping: float = 2e-3              # N m s/rad, viscous pivot friction in yaw
    # Coulomb (dry) friction torque of the corner on the ground about the vertical:
    # ~ (2/3) mu N r_contact = 0.67 * 0.3 * 19 N * 1 mm ~ 4e-3 N m. Nothing else can
    # stop a yaw spin: gravity and the pivot force have no vertical torque about the
    # pivot and the motor torque is internal, so vertical angular momentum is conserved.
    yaw_friction: float = 4e-3             # N m

    def hoop_inertia(self):
        """(I_xx, I_yy, I_zz) about the hoop centre of a thin uniform (per arc
        length) elliptical hoop in the x-z plane, semi-axes a (x) and b (z)."""
        a = self.ring_radius
        b = self.ring_b or a
        t = np.linspace(0, 2 * np.pi, 4001)[:-1]
        x, z = a * np.sin(t), -b * np.cos(t)
        ds = np.sqrt((a * np.cos(t))**2 + (b * np.sin(t))**2)
        w = self.ring_mass * ds / ds.sum()
        return float((w * z**2).sum()), float((w * (x**2 + z**2)).sum()), float((w * x**2).sum())

    def housing_without_tube(self):
        """(mass, l_S, I_x, I_y, I_z about the pivot) of the housing minus the
        cantilever tube, for the ring variant."""
        mt = 1560.0 * 7.225e-5 * 2 * self.l_E
        L = 2 * self.l_E
        m = self.m_h - mt
        lS = (self.m_h * self.l_S - mt * self.l_Q) / m
        Ix = self.I_hx - mt * self.l_Q**2
        Iy = self.I_hy - mt * (L**2 / 12 + self.l_Q**2)
        Iz = self.I_hz - mt * L**2 / 12
        return m, lS, Ix, Iy, Iz

    def with_(self, **kw):
        return replace(self, **kw)

    @property
    def m_total(self):
        if self.layout == "ring":
            mp = (2 * self.m_e + 1560.0 * 7.225e-5 * 2 * self.l_E) if self.pole != "none" else 0.0
            return (self.housing_without_tube()[0] + self.m_w + self.ring_mass
                    + 2 * self.ring_weights + self.core_extra_mass + mp)
        return self.m_h + self.m_w + 2 * self.m_e

    def beam_freq_hz(self, l_free=0.523):
        """Cantilever natural frequency as defined in the paper (App. C)."""
        return np.sqrt(self.k / (self.m_e * l_free**2)) / (2 * np.pi)

    def k_from_freq(self, f0, l_free=0.523):
        return self.m_e * l_free**2 * (2 * np.pi * f0) ** 2


NOMINAL = CubliParams()
