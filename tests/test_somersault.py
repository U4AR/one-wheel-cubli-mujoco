"""Free-body One-Wheel Cubli: rest pose and the somersault recipe."""
import numpy as np
from cubli.somersault import Maneuver, tilt_angles


def test_falls_to_upside_down_rest():
    M = Maneuver(); M.settle_upside_down()
    assert tilt_angles(M.m, M.d)[3][2, 2] < -0.8


def test_somersault_full_turn():
    M = Maneuver(); M.settle_upside_down()
    L = M.run(+0.6, 420, catch=False, T=5, w_rev=420)
    assert abs(L[-1, 3]) >= 330 and L[-1, 6] < -0.8      # one full roll, back at rest upside down
