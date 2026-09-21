"""Pogo cross: one motor balances and winds/fires the spring leg."""
import numpy as np

from cubli.pogo import PogoSim, PogoParams, inertia_ratio


def run(sim, T, schedule=()):
    sched = list(schedule)
    while sim.d.time < T:
        if sched and sim.d.time >= sched[0][0]:
            sim.set_mode(sched.pop(0)[1])
        sim.step()
        if sim.state()["fallen"]:
            break
    return sim.state()


def test_inertia_ratio_restored():
    # the leg raises the CoM (hurts the ratio); the longer bar brings it back to the paper's
    eps_short = inertia_ratio(PogoParams(l_E=0.5975))[0]
    eps = inertia_ratio(PogoParams())[0]
    assert eps < eps_short and eps < 0.46


def test_stick_balances_without_jumping():
    sim = PogoSim(seed=3)
    s = run(sim, 6.0)
    assert not s["fallen"] and sim.jumps == 0
    assert abs(s["alpha"]) < 1.0 and abs(s["beta"]) < 1.0


def test_hops_continuously_then_stops_into_a_stick():
    sim = PogoSim(seed=1)
    s = run(sim, 14.0, [(1.0, "hop")])
    assert not s["fallen"] and sim.jumps >= 3
    assert np.mean(sim.apex) > 0.12                    # ~18 cm apex
    s = run(sim, 22.0, [(0.0, "stop")])
    assert not s["fallen"] and s["mode"] == "stick" and s["phase"] == "stick"
    n = sim.jumps
    s = run(sim, 26.0)                                 # stays a stick
    assert not s["fallen"] and sim.jumps == n and abs(s["alpha"]) + abs(s["beta"]) < 1.5


def test_somersault_lands_and_balances():
    sim = PogoSim(seed=2)
    s = run(sim, 14.0, [(1.0, "flip")])
    assert sim.flips == 1
    assert not s["fallen"] and abs(s["alpha"]) + abs(s["beta"]) < 3.0
