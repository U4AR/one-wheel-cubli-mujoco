"""Named controller tunings: the paper's (Eqs. 23-24) and the CMA-ES result."""
import json
import os
import numpy as np

from .controller import Tuning, PAPER_Q, PAPER_R, PAPER_V, PAPER_W

NAMES = ([f"Q_{n}" for n in ["a", "ad", "b", "bd", "w", "d1", "d1d", "d2", "d2d"]]
         + ["Q_delay", "V_tilt", "V_rate", "V_wheel", "V_beam", "V_delay",
            "W_angle", "W_rate", "W_wheel", "comp_w_acc", "com_tau"])

TUNED_FILE = os.path.join(os.path.dirname(__file__), "tuned_params.json")
# written by tuning runs started from the live viewer
LATEST_FILE = os.path.join(os.path.dirname(__file__), "..", "results", "tuned_params_ui.json")


def to_tuning(z):
    """z = log10 multipliers on the paper's weights (see NAMES)."""
    z = np.asarray(z)
    Q = PAPER_Q.copy()
    Q[:9] *= 10 ** z[0:9]
    Q[9:] *= 10 ** z[9]
    V = PAPER_V.copy()
    V[[0, 2]] *= 10 ** z[10]; V[[1, 3]] *= 10 ** z[11]; V[4] *= 10 ** z[12]
    V[5:9] *= 10 ** z[13]; V[9:] *= 10 ** z[14]
    W = PAPER_W.copy()
    W[:2] *= 10 ** z[15]; W[2:4] *= 10 ** z[16]; W[4] *= 10 ** z[17]
    return Tuning(Q=Q, R=PAPER_R, V=V, W=W,
                  comp_w_acc=float(np.clip(0.02 * 10 ** z[18], 1e-3, 0.5)),
                  com_tau=float(20.0 * 10 ** z[19]))


def get(name="tuned"):
    if name == "paper":
        return Tuning()
    if name == "tuned":
        return to_tuning(json.load(open(TUNED_FILE))["z"])
    if name == "ring":
        # weights that balance the small hoop-through-pivot variant
        # (scripts/ring_design.py): paper weights with alpha weight x0.3, R x3
        t = Tuning()
        t.Q = t.Q.copy(); t.Q[0] *= 0.3; t.R *= 3.0
        return t
    if name == "latest-run":
        return to_tuning(json.load(open(LATEST_FILE))["z"])
    raise ValueError(name)
