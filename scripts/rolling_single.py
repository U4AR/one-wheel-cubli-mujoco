"""ONE motor rolling hoop: the hub motor (hoop <-> hanging housing) is the only
actuator. It drives the rolling speed (pendulum drive, gravity holds the housing)
and balances sideways through a passive gyroscope in the housing (vertical spin
axis, 2x the paper's flywheel inertia, pre-spun to 450 rad/s): swinging the
housing makes the gyro precess, which pushes the hoop sideways.
    python scripts/rolling_single.py"""
import json, sys
sys.path.insert(0, ".")
import numpy as np
from multiprocessing import Pool
from cubli.rolling_kane import simulate_drive, gyro_design, gyro_controller, linear_model_drive

rp = gyro_design()


def cruise(v):
    L, f = simulate_drive(rp, T=35.0, v_target=lambda t: v if 2 <= t < 22 else 0.0, ctrl=gyro_controller(rp),
                          lean0_deg=0.5, pushes=[(16.0, 0.05, "housing", (0, 3.0, 0), None)])
    sel = (L[:, 0] > 12) & (L[:, 0] < 16)
    return dict(target=v, fell=f, speed_held=round(float(L[sel, 7].mean()), 3) if sel.any() else None,
                stopped_at=round(float(L[-1, 7]), 3), lean_std_deg=round(float(np.rad2deg(L[L[:, 0] > 5, 1].std())), 2))


def max_push(v):
    lo, hi = 0.0, 12.0
    for _ in range(6):
        mid = (lo + hi) / 2
        f = simulate_drive(rp, T=16.0, v_target=lambda t: v, v0=v, lean0_deg=0.0, ctrl=gyro_controller(rp),
                           pushes=[(6.0, 0.05, "housing", (0, mid, 0), None)])[1]
        lo, hi = (mid, hi) if not f else (lo, mid)
    return v, round(lo, 1)


def capture(_):
    cap = 0
    for a in [1, 2, 3, 4, 6, 8]:
        if any(simulate_drive(rp, T=10.0, lean0_deg=a, seed=s, ctrl=gyro_controller(rp))[1] for s in range(2)):
            break
        cap = a
    return cap


if __name__ == "__main__":
    A, B = linear_model_drive(0.0, rp)
    lam = [l for l in np.linalg.eigvals(A) if np.real(l) > 0.1][0]
    print("unstable lean mode %.2f /s; controllable through the gyro (PBH rank %d/7)"
          % (np.real(lam), np.linalg.matrix_rank(np.hstack([A - lam * np.eye(7), B]), tol=1e-8)))
    with Pool(12) as pool:
        cr = pool.map_async(cruise, [-0.5, 0.5, 1.0, 1.5, 2.0])
        mp = pool.map_async(max_push, [-0.5, 0.0, 0.5, 1.0, 2.0])
        cap = pool.apply_async(capture, (None,))
        cr, mp, cap = cr.get(), mp.get(), cap.get()
    for r in cr:
        print("cruise", r)
    print("max sideways push (N, 50 ms):", dict(mp))
    print("recovers from standing lean of", cap, "deg")
    json.dump(dict(cruise=cr, max_push=dict(mp), capture_deg=cap), open("results/rolling_single.json", "w"), indent=1)
