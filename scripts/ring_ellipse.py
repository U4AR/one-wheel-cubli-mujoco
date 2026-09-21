"""Elliptical hoop: how flat must the ring be to balance with ONE motor?

Housing + flywheel at the hoop centre (the "ring with a flywheel in the middle"
look). Every shape keeps the paper's total mass and pitch inertia about the pivot;
the hoop is squashed from a circle (b/a = 1) toward a flat ellipse. The pivot tip
sits low enough below the rim that the rim clears the ground up to 15 deg tilt.
Full realistic stack: sensor noise/bias, 10 ms delay, CoM offset, motor limits.
    python scripts/ring_ellipse.py
"""
import itertools, json, sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
from scipy.optimize import brentq

import ring_design as RD
from cubli.params import NOMINAL
from cubli.tunings import get

M_HOOP = RD.pivot_props(NOMINAL)["mass"] - NOMINAL.housing_without_tube()[0] - NOMINAL.m_w
TILT_CLEAR = np.deg2rad(15)


def ellipse(a, k, f_tips=0.0):
    """f_tips: fraction of the hoop mass moved into two weights at 3 and 9 o'clock."""
    b = k * a
    lS = NOMINAL.housing_without_tube()[1]
    # hoop centre high enough that the rim clears the ground up to TILT_CLEAR
    c = max(1.05 * b, np.sqrt((a * np.sin(TILT_CLEAR))**2 + (b * np.cos(TILT_CLEAR))**2) + 0.01)
    return NOMINAL.with_(layout="ring", ring_mass=M_HOOP * (1 - f_tips), ring_radius=a, ring_b=b,
                         ring_weights=M_HOOP * f_tips / 2, ring_center=c, core_raise=c - lS)


def solve(k, f_tips=0.0):
    target = RD.pivot_props(NOMINAL)["I_y"]
    a = brentq(lambda a: RD.pivot_props(ellipse(a, k, f_tips))["I_y"] - target, 0.05, 2.0)
    return ellipse(a, k, f_tips)


def weights(qa, qb, r):
    t = get("tuned"); Q = t.Q.copy(); Q[0] *= qa; Q[2] *= qb
    return replace(t, Q=Q, R=t.R * r)


def evaluate(args):
    k, f_tips = args if isinstance(args, tuple) else (args, 0.0)
    p = solve(k, f_tips)
    row = dict(aspect_b_over_a=k, tip_weight_fraction=f_tips, tip_weights_kg=round(p.ring_weights, 3), a=round(p.ring_radius, 3), b=round(p.ring_b, 3),
               centre_height=round(p.ring_center, 3), width_m=round(2 * p.ring_radius, 2))
    lp = RD.lin_props(p)
    row |= dict(eps=round(lp["eps"], 3), controllability_vol=float(f"{lp['controllability_vol']:.3g}"))
    # default tuned weights first, then a small search
    for w in [(1, 1, 1)] + [w for w in itertools.product([0.3, 1, 3], [0.3, 1, 3], [0.3, 1, 3]) if w != (1, 1, 1)]:
        t = weights(*w)
        try:
            ok = all(not RD.run(T=20.0, plant=p, tuning=t, x0=(np.deg2rad(1), np.deg2rad(-1)), seed=s).fell
                     for s in range(2))
        except Exception:
            ok = False
        if ok:
            row["weights"] = w
            row |= RD.balance_tests(p, t)
            break
    else:
        row["balances_30s"] = False
    return row


if __name__ == "__main__":
    uniform = [(k, 0.0) for k in [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12]]
    tipped = [(k, f) for k in [1.0, 0.7, 0.5, 0.35, 0.25, 0.18] for f in [0.5, 0.75, 0.9]]
    if "--flat" in sys.argv:
        tipped = [(k, f) for k in [0.25, 0.18] for f in [0.75, 0.9]]
    jobs = uniform + tipped if "--all" in sys.argv else tipped
    with Pool(len(jobs)) as pool:
        rows = pool.map(evaluate, jobs)
    for r in rows:
        print(r, flush=True)
    json.dump(rows, open("results/ring_ellipse.json", "w"), indent=1, default=float)
