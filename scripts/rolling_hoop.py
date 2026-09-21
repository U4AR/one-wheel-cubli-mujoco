"""Rolling hoop (hub design) balanced by ONE reaction wheel, speed-scheduled LQR
(cubli/rolling_kane.py). Benchmarks standing, pushes and rolling at speed, and
compares wheel orientations.   python scripts/rolling_hoop.py"""
import itertools, json, sys
sys.path.insert(0, ".")
import numpy as np
from multiprocessing import Pool

from cubli.rolling import RollingParams
from cubli.rolling_kane import simulate_scheduled, ScheduledRollingController, linear_model_at_speed

DESIGNS = {"paper wheel (45 deg)": dict(eta=np.pi / 4, wheel_tilt=0.0),
           "lean-axis wheel": dict(eta=0.0, wheel_tilt=0.0),
           "lean-axis wheel, tilted 7 deg (final)": dict(eta=0.0, wheel_tilt=np.deg2rad(7))}
SPEEDS = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def max_side_push(args):
    name, v = args
    rp = RollingParams(**DESIGNS[name])
    mk = lambda: ScheduledRollingController(rp)
    T = 25.0 if abs(v) in (1.5, 2.0) else 12.0      # long runs slow down through the critical speed
    if any(simulate_scheduled(rp, T=T, v0=v, lean0_deg=2.0, ctrl=mk(), seed=s)[1] for s in range(2)):
        return name, v, None
    lo, hi = 0.0, 12.0
    for _ in range(6):
        mid = (lo + hi) / 2
        f = simulate_scheduled(rp, T=T, v0=v, lean0_deg=0.0, ctrl=mk(),
                               pushes=[(2.0, 0.05, "housing", (0, mid, 0), None)])[1]
        lo, hi = (mid, hi) if not f else (lo, mid)
    return name, v, round(lo, 1)


def standing(_):
    rp = RollingParams()
    L, f = simulate_scheduled(rp, T=30.0, lean0_deg=2.0)
    sel = L[:, 0] > 10
    cap = 0
    for a in [2, 4, 6, 8, 10, 12]:
        if any(simulate_scheduled(rp, T=10.0, lean0_deg=a, seed=s)[1] for s in range(2)):
            break
        cap = a
    L2, f2 = simulate_scheduled(rp, T=15.0, lean0_deg=0.0, pushes=[(2.0, 0.1, "hoop", (15.0, 0, 0), None)])
    return dict(balances=not f, lean_jitter_deg=round(float(np.rad2deg(L[sel, 1].std())), 3),
                recovers_from_lean_deg=cap, forward_kick_rolls_m=round(float(L2[:, 9].max() - L2[0, 9]), 2),
                stays_up_after_kick=not f2)


if __name__ == "__main__":
    for v in [0.0, 1.0, 2.0]:
        A, B = linear_model_at_speed(v, RollingParams())
        print(f"open-loop eigenvalues at {v} m/s:", np.round(np.sort_complex(np.linalg.eigvals(A)), 2))
    with Pool(len(DESIGNS) * len(SPEEDS) + 1) as pool:
        st = pool.apply_async(standing, (None,))
        res = pool.map(max_side_push, list(itertools.product(DESIGNS, SPEEDS)))
        st = st.get()
    print("final design, standing:", st)
    table = {n: {v: p for q, v, p in res if q == n} for n in DESIGNS}
    for n in DESIGNS:
        print(f"{n:40s}", " ".join(f"{v:+.1f}:{'%4.1fN' % p if p is not None else 'FELL '}" for v, p in table[n].items()))
    json.dump(dict(standing=st, max_side_push_N={n: {str(v): p for v, p in t.items()} for n, t in table.items()}),
              open("results/rolling_hoop.json", "w"), indent=1)
