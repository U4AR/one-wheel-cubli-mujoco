"""Rolling hoop with speed control: reaction wheel (balance) + hub motor (speed).
    python scripts/rolling_drive.py"""
import json, sys
sys.path.insert(0, ".")
import numpy as np
from multiprocessing import Pool
from cubli.rolling import RollingParams
from cubli.rolling_kane import simulate_drive, DriveController

rp = RollingParams()


def cruise(v):
    """drive from rest to v, hold, sideways push while cruising, stop."""
    c = DriveController(rp)
    prof = lambda t: v if 2 <= t < 20 else 0.0
    L, f = simulate_drive(rp, T=35.0, v_target=prof, ctrl=c, lean0_deg=0.5,
                          pushes=[(14.0, 0.05, "housing", (0, 4.0, 0), None)])
    sel = (L[:, 0] > 12) & (L[:, 0] < 20)
    return dict(target=v, fell=f, speed_held=round(float(L[sel, 7].mean()), 3) if sel.any() else None,
                stopped_at=round(float(L[-1, 7]), 3), peak_housing_pitch_deg=round(float(np.rad2deg(np.abs(L[:, 4]).max())), 1))


def max_push_at(v):
    lo, hi = 0.0, 12.0
    for _ in range(6):
        mid = (lo + hi) / 2
        c = DriveController(rp)
        f = simulate_drive(rp, T=18.0, v_target=lambda t: v, v0=v, lean0_deg=0.0, ctrl=c,
                           pushes=[(8.0, 0.05, "housing", (0, mid, 0), None)])[1]
        lo, hi = (mid, hi) if not f else (lo, mid)
    return v, round(lo, 1)


if __name__ == "__main__":
    speeds = [-0.3, 0.5, 1.0, 2.0, 3.0]
    with Pool(len(speeds) + 7) as pool:
        cr = pool.map_async(cruise, speeds)
        mp = pool.map_async(max_push_at, [-0.3, 0.0, 0.5, 1.0, 2.0, 3.0])
        cr, mp = cr.get(), mp.get()
    for r in cr:
        print("cruise", r)
    print("max sideways push while cruising (N, 50 ms):", dict(mp))
    json.dump(dict(cruise=cr, max_push=dict(mp)), open("results/rolling_drive.json", "w"), indent=1)
