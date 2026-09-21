"""Figures + benchmark table for the README.  python scripts/make_figures.py"""
import json, sys
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cubli.evaluate import drop_on_endmass, metrics, cost
from cubli.params import NOMINAL
from cubli.sim import run
from cubli.linearize import continuous_reduced
from cubli.tunings import get

C = {"paper": "#2a78d6", "tuned": "#eb6834", "third": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "lines.linewidth": 1.6,
    "legend.frameon": False, "axes.titlesize": 10, "axes.titleweight": "bold",
})
LABEL = {"paper": "paper weights (Eq. 23–24)", "tuned": "CMA-ES tuned"}


def fig_disturbance(F=1.5):
    T0 = 4.0  # let the CoM-offset estimate settle first
    runs = {k: run(T=T0 + 5.0, tuning=get(k), disturbances=drop_on_endmass(F, t0=T0), seed=0)
            for k in ["paper", "tuned"]}
    fig, ax = plt.subplots(5, 1, figsize=(7, 8.5), sharex=True)
    for k, r in runs.items():
        t = r.t - T0
        ax[0].plot(t, np.rad2deg(r.x[:, 0]), color=C[k], label=LABEL[k])
        ax[1].plot(t, np.rad2deg(r.x[:, 2]), color=C[k])
        ax[2].plot(t, r.u, color=C[k])
        ax[3].plot(t, r.x[:, 4], color=C[k])
        ax[4].plot(t, 100 * r.i2t / NOMINAL.i2t_budget, color=C[k])
    r = runs["paper"]
    ax[2].fill_between(r.t - T0, -r.ulim, r.ulim, color=GRID, alpha=0.6, lw=0,
                       label="available torque (paper run)")
    for a, yl in zip(ax, ["α  [deg]", "β  [deg]", "torque  [N m]", "wheel speed  [rad/s]",
                          "I²t state  [% of budget]"]):
        a.set_ylabel(yl)
        a.axvspan(0.0, 0.05, color="#c3c2b7", alpha=0.5, lw=0)
    ax[2].set_ylim(-3.6, 3.6)
    ax[0].legend(loc="upper right")
    ax[2].legend(loc="lower right")
    ax[-1].set_xlabel("time since disturbance  [s]")
    ax[-1].set_xlim(-1.0, 5.0)
    ax[0].set_title(f"Disturbance rejection: {F:.1f} N for 50 ms on end mass 2 (cf. paper Fig. 5)",
                    loc="left")
    fig.tight_layout()
    fig.savefig("media/disturbance_response.png", dpi=150)

    fig, a = plt.subplots(figsize=(5, 4.2))
    for k, r in runs.items():
        s = r.t >= T0 - 1.0
        a.plot(np.rad2deg(r.x[s, 0]), np.rad2deg(r.x[s, 2]), color=C[k], label=LABEL[k], lw=1.2)
    a.set_xlabel("α  [deg]"); a.set_ylabel("β  [deg]")
    a.set_title("Disturbance in the α–β plane (cf. Fig. 5 right)", loc="left")
    a.legend()
    fig.tight_layout()
    fig.savefig("media/alpha_beta_plane.png", dpi=150)


def fig_beam(which="tuned"):
    """Paper Fig. 6: controller's assumed beam frequency vs. the true 57.2 Hz."""
    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for f0, col in [(56.2, C["paper"]), (57.2, C["tuned"]), (58.2, C["third"])]:
        mp = NOMINAL.with_(k=NOMINAL.k_from_freq(f0), com_offset_xy=(0, 0))
        r = run(T=40.0, model_p=mp, tuning=get(which), seed=7)
        sel = r.t > 10
        for a, sig in zip(ax, [r.u[sel], np.rad2deg(r.x[sel, 0])]):
            sig = sig - sig.mean()
            fr = np.fft.rfftfreq(len(sig), 0.01)
            a.semilogy(fr[1:], 2 * np.abs(np.fft.rfft(sig))[1:] / len(sig), color=col, lw=0.8,
                       label=f"controller assumes f0 = {f0} Hz")
    ax[0].set_ylabel("|torque|  [N m]"); ax[1].set_ylabel("|α|  [deg]")
    ax[1].set_xlabel("frequency  [Hz]  (Nyquist = 50 Hz)")
    ax[0].set_ylim(1e-6, 1e-1); ax[1].set_ylim(1e-7, 1e-1)
    ax[1].legend(loc="upper right")
    ax[0].set_title("Sensitivity to the assumed cantilever frequency (cf. paper Fig. 6)", loc="left")
    fig.tight_layout()
    fig.savefig("media/beam_sensitivity.png", dpi=150)


def table():
    A, B = continuous_reduced()
    out = {"linearization": {"A": np.round(A, 1).tolist(), "B": np.round(B.ravel(), 1).tolist()}}
    for k in ["paper", "tuned"]:
        mt = metrics(get(k), seeds=(0, 1, 2), bisect_iters=9)
        mt["cost"] = cost(mt)
        out[k] = mt
    json.dump(out, open("results/benchmark.json", "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    fig_disturbance()
    fig_beam()
    if "--figs-only" in sys.argv:
        sys.exit()
    out = table()
    for k in ["paper", "tuned"]:
        q = out[k]["quiet"]
        print(k, f"cost={out[k]['cost']:.3f} drop={out[k]['max_drop_N']:.2f}N "
                 f"push={out[k]['max_push_N']:.2f}N std_a={q['rms_alpha_deg']:.3f} "
                 f"std_b={q['rms_beta_deg']:.3f} rms_u={q['rms_torque']:.3f} "
                 f"wheel={q['mean_wheel']:.1f}",
              {n: r["fell"] for n, r in out[k]["robust"].items()})
