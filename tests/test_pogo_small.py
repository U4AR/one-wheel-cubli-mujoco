"""Small drone-parts pogo robot: bill of materials, balance, hop/stop, somersault, get up."""
from cubli.pogo import PogoSim, inertia_ratio
from cubli.pogo_hw import SMALL, small_params

PP = small_params()


def run(sim, T, schedule=()):
    sched = list(schedule)
    while sim.d.time < T:
        if sched and sim.d.time >= sched[0][0]:
            sim.set_mode(sched.pop(0)[1])
        sim.step()
        if sim.state()["fallen"]:
            break
    return sim.state()


def test_bom_is_small_and_cheap():
    r = SMALL.report(PP)
    assert 0.25 < r["total_mass_kg"] < 0.35 and r["total_cost_usd"] < 150
    assert r["cam"]["motor_torque_while_winding_Nm"] < 0.2 * r["motor"]["peak_torque_Nm"]
    assert inertia_ratio(PP)[0] < 0.43          # at least as good as the paper's balance margin


def test_small_stands_and_hops_then_stops():
    sim = PogoSim(PP, seed=2)
    s = run(sim, 14.0, [(1.0, "hop")])
    assert not s["fallen"] and sim.jumps >= 3
    s = run(sim, 20.0, [(0.0, "stop")])
    assert not s["fallen"] and s["phase"] == "stick" and not s["down"]


def test_small_somersault():
    sim = PogoSim(PP, seed=3)
    s = run(sim, 14.0, [(1.0, "flip")])
    assert sim.flips == 1 and not s["fallen"] and s["phase"] == "stick"


def test_small_gets_up_after_falling_on_a_skid():
    sim = PogoSim(PP, seed=0)
    sim.controller_on = False
    sim.reset((-4.0, -3.0))
    run(sim, 1.5)
    assert sim.state()["down"]                  # resting on a skid / pod
    sim.controller_on = True
    s = run(sim, 8.0)
    assert not s["fallen"] and not s["down"] and s["phase"] == "stick"
    assert abs(s["alpha"]) + abs(s["beta"]) < 2.0 and sim.getups >= 1
