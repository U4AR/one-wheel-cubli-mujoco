"""A small, low-cost, buildable Pogo Cubli from drone parts.

One motor, everything on board (batteries are the end weights). About half the
size of the paper's device (0.9 m span, ~0.3 kg), which is dynamically similar
(Froude scaling: torque needs scale with length^4, times with sqrt(length)), so a
2806-class drone motor on a small FOC board is enough.

Body frame: origin at the top of the leg, z up the leg, the cantilever along x.
Part names are representative, widely sold classes; masses/specs are typical
catalogue values to be checked against the exact part bought.

Drivetrain (one motor):
  2x 3S LiPo (end pods) - leads along the bar - FOC board - 2806 drone motor
  motor bell  =  steel flywheel ring (direct drive; the reaction wheel)
  flywheel hub - printed centrifugal clutch (2 shoes, engage ~20 rad/s)
               - one-way (sprag) needle bearing, drives forward only
               - printed 2-stage spur gearbox 20:1 (module 0.5)
               - printed snail cam (20 mm rise over 144 deg, then a sharp step)
               - roller follower on a bell-crank - braided line - foot slider
  foot slider: 4 mm steel rod in two printed/PTFE bushings, inside the leg spring;
  rebound escapement: an inertial pawl trips at touchdown onto a printed rack on
  the slider; the rack re-extends through a one-way silicone rotary damper; a trip
  dog kicks the pawl out at full extension.
Roll skids (CF rods with rubber tips) and the low battery pods are the kickstands a
fall ends on; the balance controller lifts it back up from there.
"""
from dataclasses import dataclass, field
import numpy as np

from .params import NOMINAL as P

ETA = np.pi / 4                # wheel axis direction in the body x-y plane (as in the paper)


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


@dataclass
class Part:
    name: str
    mass: float
    shape: str                  # box (full dims) | cyl (r, h) | rod (p0, p1) | point | shell (edge)
    pos: tuple = (0.0, 0.0, 0.0)
    size: tuple = ()
    axis: str = "z"             # cylinder axis: x | y | z | w (wheel axis)
    rgba: str = "0.3 0.3 0.3 1"
    collide: bool = False
    cost: float = 0.0           # USD, rough street price
    note: str = ""

    def inertia(self):
        """3x3 inertia about the body-frame origin, and the part's centre."""
        m, r = self.mass, np.asarray(self.pos, float)
        if self.shape == "box":
            a, b, c = self.size
            Ic = m / 12 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
        elif self.shape == "shell":
            Ic = 5.0 / 18.0 * m * self.size[0] ** 2 * np.eye(3)
        elif self.shape == "cyl":
            rad, h = self.size
            Il = np.diag([m * rad * rad / 2, m * (3 * rad * rad + h * h) / 12, m * (3 * rad * rad + h * h) / 12])
            R = {"x": np.eye(3), "y": _Rz(np.pi / 2), "w": _Rz(ETA),
                 "z": np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])}[self.axis]
            Ic = R @ Il @ R.T
        elif self.shape == "rod":
            p0, p1 = np.asarray(self.size[0], float), np.asarray(self.size[1], float)
            r = (p0 + p1) / 2
            d = p1 - p0
            L = np.linalg.norm(d)
            u = d / L
            Ic = m * L * L / 12 * (np.eye(3) - np.outer(u, u))
        else:
            Ic = np.zeros((3, 3))
        return Ic + m * (r @ r * np.eye(3) - np.outer(r, r)), r

    def geom(self, col):
        c = col if self.collide else ""
        if self.shape == "box":
            s = np.asarray(self.size) / 2
            return (f'<geom type="box" pos="{self.pos[0]} {self.pos[1]} {self.pos[2]}" '
                    f'size="{s[0]} {s[1]} {s[2]}" rgba="{self.rgba}" {c}/>')
        if self.shape == "cyl":
            rad, h = self.size
            u = np.asarray({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1),
                            "w": (np.cos(ETA), np.sin(ETA), 0)}[self.axis], float)
            p0 = np.asarray(self.pos) - u * h / 2
            p1 = np.asarray(self.pos) + u * h / 2
            return (f'<geom type="cylinder" fromto="{p0[0]} {p0[1]} {p0[2]} {p1[0]} {p1[1]} {p1[2]}" '
                    f'size="{rad}" rgba="{self.rgba}" {c}/>')
        if self.shape == "rod":
            p0, p1 = self.size
            return (f'<geom type="capsule" fromto="{p0[0]} {p0[1]} {p0[2]} {p1[0]} {p1[1]} {p1[2]}" '
                    f'size="0.003" rgba="{self.rgba}" {c}/>')
        return ""


@dataclass
class Motor:
    """Direct-drive drone outrunner + small FOC board, 3S battery."""
    name: str = "2806-class drone outrunner, KV1300 (~48 g, e.g. for 6-7 in props)"
    driver: str = "small FOC board, ~40 A peak (e.g. ST B-G431B-ESC1 discovery kit or a SimpleFOC-type board)"
    kv_rpm: float = 1300.0
    R: float = 0.065            # phase-to-phase winding resistance (ohm)
    V: float = 11.1             # 3S LiPo nominal
    I_max: float = 40.0         # driver current limit (A)
    I_cont: float = 12.0        # thermal limit, motor in a vented printed frame (A)
    i2t_s: float = 3.0          # seconds at peak before thermal derating
    w_safe: float = 1000.0      # software speed limit (flywheel rim speed)

    @property
    def kt(self):
        return 60.0 / (2 * np.pi * self.kv_rpm)

    def params(self):
        kt = self.kt
        tau_peak = kt * self.I_max
        tau_cont = kt * self.I_cont
        return P.with_(tau_peak=tau_peak, tau_stall=kt * self.V / self.R, w_noload=self.V / kt,
                       tau_cont=tau_cont, i2t_budget=(tau_peak ** 2 - tau_cont ** 2) * self.i2t_s)

    def electrical_power(self, tau, w):
        """battery-side power (W): mechanical + copper loss; regeneration 70 % efficient"""
        I = tau / self.kt
        p = tau * w + 1.5 * I * I * self.R
        return p if p > 0 else 0.7 * p


@dataclass
class Battery:
    name: str = "3S 450 mAh 75C LiPo (~57x31x19 mm, ~40 g)"
    cells: int = 3
    mAh: float = 450
    n: int = 2                  # one pack in each end pod, in parallel

    @property
    def Wh(self):
        return self.n * self.cells * 3.7 * self.mAh / 1000


@dataclass
class Hardware:
    LP: float = 0.067           # wheel-centre height above the leg top
    LQ: float = 0.107           # bar height
    cube: float = 0.075         # housing edge
    l_E: float = 0.45           # half-length of the cantilever
    pod_drop: float = 0.046     # pod top this far below the bar (pod bottom level with the leg top)
    pod_h: float = 0.057
    skid_len: float = 0.12
    skid_drop: float = 0.0
    foot_r: float = 0.004
    electronics_W: float = 1.5  # MCU + radio + FOC board quiescent
    motor: Motor = field(default_factory=Motor)
    battery: Battery = field(default_factory=Battery)
    rim: tuple = (0.030, 0.024, 0.006)      # steel ring OD 60 / ID 48 x 6 mm
    rho_steel: float = 7850.0
    m_hub: float = 0.005
    m_rotor: float = 0.020
    r_rotor: float = 0.014
    parts: list = field(default_factory=list)

    def __post_init__(self):
        if not self.parts:
            self.parts = self.bom()

    def drivetrain_pos(self):
        a = np.array([np.cos(ETA), np.sin(ETA), 0.0])
        return -0.022 * a + np.array([0, 0, self.LP - 0.004])

    # ------------------------------------------------------------------ parts
    def bom(self):
        lE, LQ, LP, drop, ph = self.l_E, self.LQ, self.LP, self.pod_drop, self.pod_h
        zpod = LQ - drop - ph / 2
        return [
            Part("housing: printed PETG cube 75 mm + 2 CF plates", 0.025, "shell", (0, 0, LP), (self.cube,),
                 cost=4),
            Part("tip pyramid struts (printed)", 0.004, "point", (0, 0, 0.015)),
            Part("motor stator + mount", 0.028, "cyl", (0, 0, LP), (0.014, 0.012), "w", "0.25 0.25 0.28 1",
                 cost=20, note=self.motor.name),
            Part("FOC motor board (placed opposite the gearbox to trim the CoM)", 0.010, "box",
                 tuple(np.array([0, 0, LP + 0.012]) + 0.035 * np.array([np.cos(ETA), np.sin(ETA), 0])),
                 (0.03, 0.03, 0.008),
                 rgba="0.1 0.45 0.2 1", cost=20, note=self.motor.driver),
            Part("ESP32-C3 (radio link for the external tracking) + IMU", 0.005, "box", (0, 0, 0.034),
                 (0.025, 0.018, 0.006), rgba="0.1 0.35 0.15 1", cost=6),
            Part("winding drivetrain: clutch, sprag bearing, 20:1 printed gears, cam, follower", 0.016, "box",
                 tuple(self.drivetrain_pos()), (0.018, 0.022, 0.025), rgba="0.55 0.55 0.6 1", cost=8),
            Part("leg: guide, 2 bushings, spring, escapement (pawl, rack, rotary damper)", 0.012, "cyl",
                 (0, 0, -0.006), (0.007, 0.018), "z", "0.7 0.72 0.76 1", cost=8),
            Part("cantilever: CF tube 8 mm x 0.9 m", 0.015 * 2 * lE / 0.9, "rod",
                 size=((-lE, 0, LQ), (lE, 0, LQ)), rgba="0.12 0.12 0.13 1", collide=True, cost=6),
            Part("pod drop strut L (CF rod)", 0.002, "rod", size=((-lE, 0, LQ), (-lE, 0, LQ - drop))),
            Part("pod drop strut R (CF rod)", 0.002, "rod", size=((lE, 0, LQ), (lE, 0, LQ - drop))),
            Part("battery pod L (3S LiPo + printed clip)", 0.044, "box", (-lE, 0, zpod), (0.031, 0.019, ph),
                 rgba="0.85 0.7 0.1 1", collide=True, cost=9, note=self.battery.name),
            Part("battery pod R (3S LiPo + printed clip)", 0.044, "box", (lE, 0, zpod), (0.031, 0.019, ph),
                 rgba="0.85 0.7 0.1 1", collide=True, cost=9, note=self.battery.name),
            Part("roll skid +y (CF rod, rubber tip)", 0.002, "rod",
                 size=((0, 0.01, 0.0), (0, self.skid_len, -self.skid_drop)), cost=1),
            Part("roll skid -y (CF rod, rubber tip)", 0.002, "rod",
                 size=((0, -0.01, 0.0), (0, -self.skid_len, -self.skid_drop)), cost=1),
            Part("harness, XT30 connectors, switch", 0.008, "point", (0, 0, LQ), cost=4),
            Part("tracking markers / AprilTag plate", 0.002, "point", (0, 0, LP + 0.04), cost=1),
        ]

    # ------------------------------------------------------------ flywheel
    def wheel(self):
        ro, ri, w = self.rim
        m_rim = self.rho_steel * np.pi * (ro * ro - ri * ri) * w
        I_rim = 0.5 * m_rim * (ro * ro + ri * ri)
        I_ax = I_rim + 0.5 * self.m_hub * ri * ri + self.m_rotor * self.r_rotor ** 2
        m = m_rim + self.m_hub + self.m_rotor
        return dict(m=m, I_ax=I_ax, I_tr=0.5 * I_ax * 1.0001 + m_rim * w * w / 12, m_rim=m_rim, r=ro, width=w)

    # ------------------------------------------------------- mass properties
    def body_props(self):
        """(mass, CoM (3,), inertia about the CoM (3x3)) without flywheel and foot."""
        m, mc, I0 = 0.0, np.zeros(3), np.zeros((3, 3))
        for p in self.parts:
            I, r = p.inertia()
            m += p.mass
            mc += p.mass * r
            I0 += I
        c = mc / m
        return m, c, I0 - m * (c @ c * np.eye(3) - np.outer(c, c))

    # ------------------------------------------------------------ geometry
    def geoms(self, col):
        h, z0 = self.cube / 2, self.LP
        corners = [(sx * h, sy * h, z0 + sz * h) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        edges = [(a, b) for i, a in enumerate(corners) for b in corners[i + 1:]
                 if sum(abs(u - v) > 1e-9 for u, v in zip(a, b)) == 1]
        edges += [((0, 0, 0.0), c) for c in corners if c[2] < z0]
        g = [f'<geom type="capsule" fromto="{a[0]} {a[1]} {a[2]} {b[0]} {b[1]} {b[2]}" size="0.0025" '
             f'material="carbon" {col}/>' for a, b in edges]
        g += [p.geom(col) for p in self.parts]
        for s in (1, -1):
            g.append(f'<geom name="skid{"p" if s > 0 else "n"}" type="sphere" pos="0 {s * self.skid_len} '
                     f'{-self.skid_drop}" size="0.004" rgba="0.1 0.1 0.1 1" {col}/>')
        q = self.cube / 2 * 0.8
        for p in [(q, q), (-q, q), (q, -q), (-q * 0.6, -q * 0.4)]:
            g.append(f'<geom type="sphere" pos="{p[0]} {p[1]} {self.LP + self.cube / 2 + 0.004}" size="0.0035" '
                     f'rgba="0.95 0.95 0.95 1"/>')
        return "\n      ".join(x for x in g if x)

    def wheel_xml(self):
        w = self.wheel()
        wq = f"{np.cos(ETA / 2)} 0 0 {np.sin(ETA / 2)}"
        ro, ri, wd = self.rim
        return f'''      <body name="wheel" pos="0 0 {self.LP}" quat="{wq}">
        <joint name="phi" type="hinge" axis="1 0 0"/>
        <inertial pos="0 0 0" mass="{w['m']}" diaginertia="{w['I_ax']} {w['I_tr']} {w['I_tr']}"/>
        <geom type="cylinder" fromto="0.006 0 0 {0.006 + wd} 0 0" size="{ro}" rgba="0.62 0.64 0.68 1"/>
        <geom type="cylinder" fromto="0.003 0 0 0.006 0 0" size="{ri}" rgba="0.8 0.35 0.1 1"/>
        <geom type="cylinder" fromto="-0.010 0 0 0.003 0 0" size="{self.r_rotor}" rgba="0.15 0.15 0.18 1"/>
        <geom type="box" pos="{0.007 + wd} 0 {0.8 * ro}" size="0.001 0.003 0.004" rgba="1 1 1 1"/>
      </body>'''

    def cam_xml(self):
        p = self.drivetrain_pos() + np.array([0, 0, -0.018])
        return f'''      <body name="cam" pos="{p[0]} {p[1]} {p[2]}">
        <joint name="camj" type="hinge" axis="0 1 0" armature="1e-6"/>
        <inertial pos="0 0 0" mass="1e-5" diaginertia="1e-10 1e-10 1e-10"/>
        <geom type="cylinder" fromto="0 -0.003 0 0 0.003 0" size="0.011" rgba="0.75 0.2 0.2 1"/>
        <geom type="box" pos="0.009 0 0" size="0.003 0.0035 0.002" rgba="0.95 0.95 0.95 1"/>
      </body>'''

    # ------------------------------------------------------------ report
    def report(self, pp):
        m_b, c, Ic = self.body_props()
        w = self.wheel()
        mp = self.motor.params()
        F_max = pp.k_leg * (pp.preload + pp.stroke - pp.s_min)
        E_spring = 0.5 * pp.k_leg * ((pp.preload + pp.stroke - pp.s_min) ** 2 - pp.preload ** 2)
        cam_arm = (pp.stroke - pp.s_min) / (2 * np.pi * pp.f_wind)
        parts = [dict(name=p.name, mass_g=round(1e3 * p.mass, 1), cost_usd=p.cost, note=p.note) for p in self.parts]
        parts += [dict(name="flywheel: steel ring OD60/ID48 x 6 mm (bolted to the motor bell)",
                       mass_g=round(1e3 * (w["m"] - self.m_rotor), 1), cost_usd=3,
                       note=f"I = {w['I_ax'] * 1e6:.1f} g cm^2 incl. rotor bell"),
                  dict(name="foot slider + rubber foot", mass_g=round(1e3 * pp.m_foot, 1), cost_usd=1, note=""),
                  dict(name="bearings, screws, spring, line, glue", mass_g=0.0, cost_usd=8, note="")]
        return dict(
            parts=parts,
            total_mass_kg=round(m_b + w["m"] + pp.m_foot, 3),
            total_cost_usd=round(sum(p["cost_usd"] for p in parts)),
            span_m=2 * self.l_E,
            body_com_m=[round(float(v), 4) for v in c],
            flywheel=dict(I_kg_m2=float(f"{w['I_ax']:.3g}"), mass_g=round(1e3 * w["m"], 1),
                          software_speed_limit_rad_s=self.motor.w_safe,
                          rim_speed_at_limit_m_s=round(w["r"] * self.motor.w_safe, 1),
                          energy_at_limit_J=round(0.5 * w["I_ax"] * self.motor.w_safe ** 2, 1)),
            motor=dict(kt_Nm_per_A=round(self.motor.kt, 5), peak_torque_Nm=round(mp.tau_peak, 3),
                       cont_torque_Nm=round(mp.tau_cont, 3), no_load_rad_s=round(mp.w_noload),
                       stall_line_Nm=round(mp.tau_stall, 2)),
            battery=dict(pack=self.battery.name, packs=self.battery.n, energy_Wh=round(self.battery.Wh, 1)),
            spring=dict(k_N_per_mm=pp.k_leg / 1e3, preload_N=round(pp.k_leg * pp.preload, 1),
                        max_N=round(F_max, 1), stored_J=round(E_spring, 3)),
            cam=dict(rise_mm=round(1e3 * (pp.stroke - pp.s_min), 1), wind_deg=round(360 * pp.f_wind), gear=pp.G,
                     cam_torque_Nm=round(F_max * cam_arm, 3),
                     motor_torque_while_winding_Nm=round(F_max * cam_arm / pp.G / 0.8, 4), efficiency_assumed=0.8),
        )


SMALL = Hardware()


def small_params(hw=None, **kw):
    """PogoParams for the small drone-parts robot (leg and spring scaled from the
    paper-size design: lengths x1/2, forces x1/8, spring rate x1/4)."""
    from .pogo import PogoParams
    hw = hw or SMALL
    base = dict(l_E=hw.l_E, L0=0.015, stroke=0.025, s_min=0.005, preload=0.010, k_leg=2000.0,
                b_leg=0.35, b_comp=26.0, m_foot=0.008, G=20.0, f_wind=0.4, w_clutch=20.0, mu=0.9,
                v_pay=0.07, hw=hw, Ts=0.005, q_wheel=10.0, w_park=-200.0, catch_window=0.15)
    base.update(kw)
    return PogoParams(**base)
