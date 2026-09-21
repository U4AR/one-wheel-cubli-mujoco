"""Design A (hoop, housing + flywheel in the middle) with a bigger motor/flywheel.

Motor torques x k_motor (its mass grows: +0.36 kg per x1), flywheel inertia and
mass x k_wheel. For each size the LQR weights are re-searched; balancing ones are
scored on recoverable initial tilt and disturbances.
    python scripts/ring_bigmotor.py
"""
import itertools, json, sys
sys.path.insert(0, ".")
from dataclasses import replace
from multiprocessing import Pool
import numpy as np

from cubli.params import NOMINAL
from cubli.sim import run, Disturbance
from cubli.tunings import get

R, M = 0.3196, 0.7427                 # design A hoop (keeps the paper's pitch inertia)


def hoop_A(k_motor, k_wheel):
    lS = NOMINAL.housing_without_tube()[1]
    return NOMINAL.with_(layout="ring", ring_mass=M, ring_radius=R, ring_center=1.05 * R,
                         core_raise=1.05 * R - lS).scaled_actuator(k_motor, k_wheel)


def weights(qa, qb, qw, r):
    t = get("paper"); Q = t.Q.copy()
    Q[0] *= qa; Q[2] *= qb; Q[4] *= qw
    return replace(t, Q=Q, R=t.R * r)


GRID = list(itertools.product([0.1, 1, 10], [0.1, 1, 10], [0.1, 1, 10], [0.03, 0.3, 3]))


def first_balancing(args):
    km, kw = args
    p = hoop_A(km, kw)
    ok = []
    for w in GRID:
        try:
            r = run(T=20.0, plant=p, tuning=weights(*w), x0=(np.deg2rad(1), np.deg2rad(-1)), seed=0)
        except Exception:
            continue
        if not r.fell:
            ok.append((w, float(np.sqrt((r.u[r.t > 5] ** 2).mean()))))
    return km, kw, ok


def score(args):
    km, kw, w = args
    p, t = hoop_A(km, kw), weights(*w)
    out = dict(k_motor=km, k_wheel=kw, weights=w, mass=round(p.m_total, 2))
    fell = [run(T=30.0, plant=p, tuning=t, x0=(np.deg2rad(1), np.deg2rad(-1)), seed=s).fell
            for s in range(3)]
    out["balances_3_seeds"] = not any(fell)
    if any(fell):
        return out
    r = run(T=30.0, plant=p, tuning=t, x0=(np.deg2rad(1), np.deg2rad(-1)), seed=0)
    sel = r.t > 5
    out["std_tilt_deg"] = [round(float(np.rad2deg(r.x[sel, i].std())), 3) for i in (0, 2)]
    out["rms_torque"] = round(float(np.sqrt((r.u[sel] ** 2).mean())), 3)
    out["peak_wheel_idle"] = round(float(np.abs(r.x[sel, 4]).max()))
    tilt = 0.0
    for a in [1, 2, 3, 4, 6, 8, 10]:
        if run(T=10.0, plant=p, tuning=t, x0=(np.deg2rad(a), np.deg2rad(-a)), seed=1).fell:
            break
        tilt = a
    out["recovers_from_tilt_deg"] = tilt
    def maxF(make, hi):
        lo = 0.0
        for _ in range(8):
            mid = (lo + hi) / 2
            f = run(T=7.0, plant=p, tuning=t, disturbances=make(mid), seed=0).fell
            lo, hi = (mid, hi) if not f else (lo, mid)
        return round(lo, 2)
    out["max_drop_on_hoop_edge_N"] = maxF(lambda F: [Disturbance(2, 0.05, "ring", (0, 0, -F), (R, 0, 0))], 20.0)
    out["max_side_push_N"] = maxF(lambda F: [Disturbance(2, 0.05, "housing", (0, F, 0))], 60.0)
    return out


if __name__ == "__main__":
    sizes = list(itertools.product([2, 4, 6, 10], [1, 2, 4, 8]))
    with Pool(16) as pool:
        found = pool.map(first_balancing, sizes)
    cand = []
    for km, kw, ok in found:
        print(f"motor x{km:<2} wheel x{kw}: {len(ok)}/{len(GRID)} weight sets balance", flush=True)
        if ok:   # keep the lowest-torque-noise weight set per size
            cand.append((km, kw, min(ok, key=lambda o: o[1])[0]))
    with Pool(max(1, len(cand))) as pool:
        scored = pool.map(score, cand)
    for s in scored:
        print(s, flush=True)
    json.dump(scored, open("results/ring_bigmotor.json", "w"), indent=1)
