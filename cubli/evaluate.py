"""Benchmark scenarios and a scalar score for tuning."""
import numpy as np

from .params import NOMINAL
from .sim import run, Disturbance, SensorModel
from .controller import Tuning


def drop_on_endmass(F, t0=2.0, dur=0.05):
    """Fig. 5: object dropped on end mass 2 -> positive beta disturbance."""
    return [Disturbance(t0, dur, "endmass2", (0.0, 0.0, -F))]


def push_housing(F, t0=2.0, dur=0.05):
    """Lateral push at the top of the housing along +y -> roll (alpha) disturbance."""
    return [Disturbance(t0, dur, "housing", (0.0, F, 0.0))]


def max_recoverable(make_dist, tuning=None, plant=NOMINAL, lo=0.0, hi=200.0,
                    iters=10, T=6.0, **kw):
    """Bisection on disturbance amplitude (N over 50 ms)."""
    ok = lambda F: not run(T=T, plant=plant, tuning=tuning, disturbances=make_dist(F), **kw).fell
    if not ok(lo):
        return 0.0
    if ok(hi):
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def steady_metrics(res, t_from=5.0):
    sel = res.t >= t_from
    x = res.x[sel]
    return dict(
        # deviation about the mean: with a CoM offset the true balance point is
        # not alpha = beta = 0, so the mean tilt is not an error
        rms_alpha_deg=float(np.rad2deg(x[:, 0].std())),
        rms_beta_deg=float(np.rad2deg(x[:, 2].std())),
        mean_alpha_deg=float(np.rad2deg(x[:, 0].mean())),
        mean_beta_deg=float(np.rad2deg(x[:, 2].mean())),
        rms_torque=float(np.sqrt((res.u[sel] ** 2).mean())),
        mean_wheel=float(x[:, 4].mean()),
        rms_beam_mrad=float(1e3 * np.sqrt((x[:, [5, 7]] ** 2).mean())),
    )


ROBUST_PLANTS = {
    "beam 56.2 Hz": NOMINAL.with_(k=NOMINAL.k_from_freq(56.2)),
    "beam 58.2 Hz": NOMINAL.with_(k=NOMINAL.k_from_freq(58.2)),
    "end masses +10%": NOMINAL.with_(m_e=NOMINAL.m_e * 1.1),
    "roll inertia +10%": NOMINAL.with_(I_hx=NOMINAL.I_hx * 1.1, m_h=NOMINAL.m_h * 0.97),
}


def metrics(tuning: Tuning, seeds=(0, 1), bisect_iters=7):
    """All benchmark numbers for one tuning."""
    out = {}
    q = []
    for s in seeds:
        r = run(T=60.0, tuning=tuning, x0=(np.deg2rad(1), np.deg2rad(-1)), seed=s)
        q.append(steady_metrics(r, 30.0) | {"fell": r.fell})
    out["quiet"] = {k: float(np.mean([m[k] for m in q])) for k in q[0]}
    out["max_drop_N"] = max_recoverable(drop_on_endmass, tuning, iters=bisect_iters, hi=6.0)
    out["max_push_N"] = max_recoverable(push_housing, tuning, iters=bisect_iters, hi=30.0)
    out["robust"] = {}
    for name, p in ROBUST_PLANTS.items():
        r = run(T=15.0, plant=p, model_p=NOMINAL, tuning=tuning, x0=(np.deg2rad(1), 0.0), seed=3)
        out["robust"][name] = steady_metrics(r, 5.0) | {"fell": r.fell}
    r = run(T=15.0, tuning=tuning, sensors=SensorModel(delay_steps=2), seed=4)
    out["robust"]["2-step delay"] = steady_metrics(r, 5.0) | {"fell": r.fell}
    r = run(T=15.0, tuning=tuning, sensors=SensorModel(acc_std=0.12, gyr_std=0.012, wheel_std=1.0), seed=5)
    out["robust"]["3x sensor noise"] = steady_metrics(r, 5.0) | {"fell": r.fell}
    return out


def cost(mt):
    """Scalar objective (lower is better)."""
    q = mt["quiet"]
    if q["fell"]:
        return 1e3
    c = 2.0 * (q["rms_alpha_deg"] + q["rms_beta_deg"]) + 1.0 * q["rms_torque"] \
        + 0.005 * abs(q["mean_wheel"])
    c -= 0.5 * mt["max_drop_N"] + 0.05 * mt["max_push_N"]   # larger recoverable disturbance
    for name, r in mt["robust"].items():
        c += 20.0 if r["fell"] else 0.5 * (r["rms_alpha_deg"] + r["rms_beta_deg"])
    return float(c)
