"""Replace the cantilever + two end masses by a hoop in the pitch plane, keeping
the paper's parameters as far as physically possible, and check it balances.

A ring cannot reproduce the bar exactly: about its own centre a thin hoop in the
x-z plane has I_yy = M R^2 but I_xx = M R^2 / 2, so it adds a lot of ROLL inertia
that the end masses never had. The pitch/roll inertia ratio needed for the silver
ratio (~5.4) is out of reach; the best a hoop standing on its bottom gives is 4/3.

  A "hoop, housing in the middle": housing + flywheel at the hoop centre, pivot
    tip just below the rim. Keeps total mass and the PITCH inertia.
  B "hoop through the pivot": housing where it was (pivot = its corner), hoop
    passes just above the pivot. Keeps the PITCH inertia AND the gravity torque.

    python scripts/ring_design.py
"""
import json, sys
sys.path.insert(0, ".")
import numpy as np
import mujoco
from scipy.optimize import brentq, fsolve

from cubli.model import load
from cubli.params import NOMINAL
from cubli.linearize import continuous_reduced
from cubli.sim import run, Disturbance
run = run
from cubli.tunings import get

GAP = 1.05          # hoop centre at 1.05 R: rim clears the ground up to ~17 deg tilt


def pivot_props(p):
    """total mass, gravity moment sum(m h), I_x and I_y about the pivot."""
    m, d = load(p.with_(com_offset_xy=(0.0, 0.0)))
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, d, M)
    b = m.body("housing").id
    mt = m.body_subtreemass[b]
    return dict(mass=float(mt), N=float(mt * d.subtree_com[b][2]),
                I_x=float(M[0, 0]), I_y=float(M[1, 1]))


def lin_props(p):
    A, B = continuous_reduced(p.with_(com_offset_xy=(0.0, 0.0)))
    pa2, pb2 = A[1, 0], A[3, 2]
    pa, pb = np.sqrt(pa2), np.sqrt(pb2)
    eta = p.eta
    sigma = B[1, 0] / (pa2 * np.cos(eta))
    vol = (sigma * np.sin(eta) * np.cos(eta) * pa * pb / 4 * (pa - pb) / (pa + pb)) ** 2
    return dict(pi_a2=float(pa2), pi_b2=float(pb2), eps=float(pb / pa),
                f_hz=(float(pa / 2 / np.pi), float(pb / 2 / np.pi)),
                sigma=float(sigma), B_alpha=float(B[1, 0]), B_beta=float(B[3, 0]),
                controllability_vol=float(vol))


def ring_A(R, M):
    mh, lS, *_ = NOMINAL.housing_without_tube()
    c = GAP * R
    return NOMINAL.with_(layout="ring", ring_mass=M, ring_radius=R, ring_center=c,
                         core_raise=c - lS)


def ring_B(R, M):
    return NOMINAL.with_(layout="ring", ring_mass=M, ring_radius=R, ring_center=GAP * R,
                         core_raise=0.0)


def design():
    bar = pivot_props(NOMINAL)
    mh, lS, *_ = NOMINAL.housing_without_tube()
    # A: same total mass, same pitch inertia
    M_A = bar["mass"] - mh - NOMINAL.m_w
    R_A = brentq(lambda R: pivot_props(ring_A(R, M_A))["I_y"] - bar["I_y"], 0.05, 1.5)
    # B: same pitch inertia and same gravity moment
    def eqs(v):
        R, M = v
        q = pivot_props(ring_B(R, M))
        return [q["I_y"] - bar["I_y"], q["N"] - bar["N"]]
    R_B, M_B = fsolve(eqs, [0.9, 0.15])
    return {"bar (paper)": NOMINAL, "A: hoop, housing in middle": ring_A(R_A, M_A),
            "B: hoop through pivot": ring_B(R_B, M_B)}


def edge_point(p):
    """body + local point for 'drop an object on the far end' (+x extreme)."""
    if p.layout == "ring":
        return "ring", (p.ring_radius, 0.0, 0.0)
    return "endmass2", None


def balance_tests(p, tuning):
    out = {}
    r = run(T=30.0, plant=p, tuning=tuning, x0=(np.deg2rad(1), np.deg2rad(-1)), seed=0)
    sel = r.t > 10
    out["balances_30s"] = not r.fell
    if not r.fell:
        out["std_alpha_deg"] = float(np.rad2deg(r.x[sel, 0].std()))
        out["std_beta_deg"] = float(np.rad2deg(r.x[sel, 2].std()))
        out["rms_torque"] = float(np.sqrt((r.u[sel] ** 2).mean()))
    body, pt = edge_point(p)
    def max_force(make):
        lo, hi = 0.0, 30.0
        for _ in range(9):
            mid = (lo + hi) / 2
            f = run(T=7.0, plant=p, tuning=tuning, disturbances=make(mid), seed=0).fell
            lo, hi = (mid, hi) if not f else (lo, mid)
        return round(lo, 2)
    out["max_drop_on_far_end_N"] = max_force(lambda F: [Disturbance(2.0, 0.05, body, (0, 0, -F), pt)])
    out["max_side_push_housing_N"] = max_force(lambda F: [Disturbance(2.0, 0.05, "housing", (0, F, 0))])
    return out


if __name__ == "__main__":
    designs = design()
    res = {}
    for name, p in designs.items():
        row = dict(pivot_props(p)) | lin_props(p)
        if p.layout == "ring":
            row["ring"] = dict(R=p.ring_radius, M=p.ring_mass, centre=p.ring_center,
                               core_raise=p.core_raise)
        row |= balance_tests(p, get("tuned"))
        res[name] = row
        print(name, json.dumps(row, indent=None, default=float), "\n", flush=True)
    json.dump(res, open("results/ring_design.json", "w"), indent=1, default=float)
