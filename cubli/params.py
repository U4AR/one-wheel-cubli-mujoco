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

    def with_(self, **kw):
        return replace(self, **kw)

    @property
    def m_total(self):
        return self.m_h + self.m_w + 2 * self.m_e

    def beam_freq_hz(self, l_free=0.523):
        """Cantilever natural frequency as defined in the paper (App. C)."""
        return np.sqrt(self.k / (self.m_e * l_free**2)) / (2 * np.pi)

    def k_from_freq(self, f0, l_free=0.523):
        return self.m_e * l_free**2 * (2 * np.pi * f0) ** 2


NOMINAL = CubliParams()
