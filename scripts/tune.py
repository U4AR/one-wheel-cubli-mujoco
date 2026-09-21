"""CMA-ES search over LQR / Kalman-filter / filter weights, starting from the
paper's tuning (Eqs. 23-24). Parameters are log10 multipliers on the paper values."""
import json, sys, time
from multiprocessing import Pool
import numpy as np
import cma

sys.path.insert(0, ".")
from cubli.tunings import NAMES, to_tuning
from cubli.evaluate import metrics, cost

def f(z):
    try:
        return cost(metrics(to_tuning(z), bisect_iters=7))
    except Exception as e:  # e.g. Riccati failure
        return 1e3


if __name__ == "__main__":
    # usage: tune.py [GENS] [WARM_START_JSON] [OUT_JSON] [POPSIZE]
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out = sys.argv[3] if len(sys.argv) > 3 else "cubli/tuned_params.json"
    popsize = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    x0 = np.zeros(len(NAMES))
    if len(sys.argv) > 2 and sys.argv[2] != "none":  # warm start
        x0 = np.array(json.load(open(sys.argv[2]))["z"])
    es = cma.CMAEvolutionStrategy(x0, 0.35,
                                  {"popsize": popsize, "bounds": [-3, 3], "seed": 1})
    best = (np.inf, None)
    with Pool(popsize) as pool:
        for g in range(gens):
            X = es.ask()
            F = pool.map(f, X)
            es.tell(X, F)
            i = int(np.argmin(F))
            if F[i] < best[0]:
                best = (F[i], np.array(X[i]))
            print(f"gen {g:3d}  best {best[0]:.3f}  gen-min {F[i]:.3f}  median {np.median(F):.3f}",
                  flush=True)
            json.dump({"cost": best[0], "z": best[1].tolist(), "names": NAMES},
                      open(out, "w"), indent=1)
