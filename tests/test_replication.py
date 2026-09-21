"""Checks that the MuJoCo model reproduces the paper's linear model (Eq. 11)
and that the closed loop balances."""
import numpy as np

from cubli.linearize import continuous_reduced
from cubli.sim import run
from cubli.tunings import get

# (row, col, value) entries printed in Eq. (11) of the paper; state order
# alpha, alpha_d, beta, beta_d, phi_d, d1, d1_d, d2, d2_d
PAPER_A = [(1, 0, 57.7), (1, 2, 0.1), (1, 6, 4.6), (1, 8, -4.6),
           (3, 2, 10.6),
           (4, 0, -40.9), (4, 2, -7.5), (4, 6, -3.2), (4, 8, 3.2),
           (6, 0, -4.2), (6, 6, -19.8), (6, 8, -12.6),
           (8, 0, 4.2), (8, 6, -12.6), (8, 8, -19.8)]
PAPER_B = [-20.7, -2.3, 1122.0, 7.4, -7.4]


def test_linearization_matches_paper():
    A, B = continuous_reduced()
    for i, j, v in PAPER_A:
        assert abs(A[i, j] - v) <= max(0.06, 0.01 * abs(v)), (i, j, A[i, j], v)
    np.testing.assert_allclose(B[[1, 3, 4, 6, 8], 0], PAPER_B, rtol=0.01, atol=0.06)


def test_silver_ratio():
    A, _ = continuous_reduced()
    eps = np.sqrt(A[3, 2] / A[1, 0])
    assert abs(eps - 0.43) < 0.01


def test_balances_with_paper_and_tuned_weights():
    for name in ["paper", "tuned"]:
        r = run(T=10.0, tuning=get(name), x0=(np.deg2rad(2), np.deg2rad(-2)))
        assert not r.fell, name
        assert np.abs(r.x[-100:, [0, 2]]).max() < np.deg2rad(2.5), name
