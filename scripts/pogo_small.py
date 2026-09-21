"""Small, low-cost Pogo Cubli from drone parts: bill of materials + every behaviour
on 8 noise seeds (stand, hop -> stop, somersault, get up after a fall, hop forward).

  python scripts/pogo_small.py            # results/pogo_small.json, media/pogo_small_drivetrain.png
  python scripts/pogo_small.py --video    # media/pogo_small.mp4 / .gif
"""
import os
import sys
import json
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cubli.pogo import PogoSim, inertia_ratio  # noqa: E402
from cubli.pogo_hw import SMALL, small_params  # noqa: E402

PP = small_params()


def _run_until(sim, T, sched=(), hook=None):
    sched = list(sched)
    while sim.d.time < T:
        if sched and sim.d.time >= sched[0][0]:
            sim.set_mode(sched.pop(0)[1])
        sim.step()
        if hook:
            hook(sim)
        if sim.state()["fallen"]:
            break
    return sim.state()


def stick(seed):
    sim = PogoSim(PP, seed=seed)
    s = _run_until(sim, 8.0)
    return dict(ok=not s["fallen"] and sim.jumps == 0 and abs(s["alpha"]) + abs(s["beta"]) < 1.5)


def hop_stop(seed):
    sim = PogoSim(PP, seed=seed)
    s = _run_until(sim, 21.0, [(1.0, "hop")])
    n, apex = sim.jumps, float(np.mean(sim.apex)) if sim.apex else 0.0
    s = _run_until(sim, 28.0, [(0.0, "stop")])
    ok = not s["fallen"] and s["phase"] == "stick" and not s["down"]
    return dict(ok=ok, hops_in_20s=n, extra_after_stop=sim.jumps - n, apex_cm=round(100 * apex, 1),
                energy_Wh=round(s["energy_Wh"], 4), t=round(s["t"], 1))


def flip(seed):
    sim = PogoSim(PP, seed=seed)
    s = _run_until(sim, 16.0, [(1.0, "flip")])
    return dict(ok=sim.flips == 1 and not s["fallen"] and s["phase"] == "stick" and not s["down"],
                flips=sim.flips)


def fallen(args):
    a, b, seed = args
    sim = PogoSim(PP, seed=seed)
    sim.controller_on = False
    sim.reset((a, b))
    _run_until(sim, 1.5)
    s = sim.state()
    rest = (round(float(s["alpha"]), 1), round(float(s["beta"]), 1))
    sim.controller_on = True
    t_up = [None]

    def hook(sm):
        st = sm.state()
        if t_up[0] is None and st["phase"] == "stick" and not st["down"] and abs(st["alpha"]) + abs(st["beta"]) < 2:
            t_up[0] = round(st["t"] - 1.5, 2)

    s = _run_until(sim, 9.5, hook=hook)
    ok = not s["fallen"] and s["phase"] == "stick" and not s["down"] and abs(s["alpha"]) + abs(s["beta"]) < 2
    return dict(ok=ok, start=(a, b), rest_deg=rest, t_up_s=t_up[0], attempts=sim.getups)


def knock(args):
    ang, F, seed = args
    sim = PogoSim(PP, seed=seed)
    push = np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang)), 0.0]) * F

    def h():
        sim.d.xfrc_applied[:] = 0
        if 1.0 <= sim.d.time < 1.1:
            sim.d.xfrc_applied[sim.body, :3] = push
    sim.substep_hook = h
    down = [False]
    s = _run_until(sim, 9.0, hook=lambda sm: down.__setitem__(0, down[0] or sm.state()["down"]))
    ok = not s["fallen"] and s["phase"] == "stick" and not s["down"] and abs(s["alpha"]) + abs(s["beta"]) < 2
    return dict(ok=ok, push_deg=ang, push_N=F, went_down=down[0], attempts=sim.getups)


def forward(seed):
    """hop forward for 30 s holding the start heading (body +y is forward)"""
    sim = PogoSim(PP, seed=seed)
    _run_until(sim, 1.0)
    psi0 = sim.heading()
    f0 = np.array([-np.sin(psi0), np.cos(psi0)])
    p0 = sim.d.qpos[:2].copy()
    s = _run_until(sim, 31.0, [(0.0, "hop_fwd")])
    dp = sim.d.qpos[:2] - p0
    return dict(ok=not s["fallen"] and float(dp @ f0) > 0.1, jumps=sim.jumps, forward_m=round(float(dp @ f0), 3),
                sideways_m=round(float(dp @ np.array([f0[1], -f0[0]])), 3), skid_touches=sim.getups,
                heading_err_deg=round(float(np.rad2deg(sim.heading_error())), 1))


def go(args):
    """hop forward 24 s, turn (left = +90, right = -90), hop forward again; returns the path"""
    seed, turn = args
    sim = PogoSim(PP, seed=seed)
    _run_until(sim, 1.0)
    psi0 = sim.heading()
    f0 = np.array([-np.sin(psi0), np.cos(psi0)])
    path, turned, p_turn = [], False, None
    sim.set_mode("hop_fwd")
    while sim.d.time < 70.0:
        if not turned and sim.d.time >= 25.0:
            sim.set_mode("turn_left" if turn > 0 else "turn_right")
            turned, p_turn = True, sim.d.qpos[:2].copy()
        sim.step()
        if int(round(sim.d.time / sim.Ts)) % 20 == 0:
            path.append([round(float(v), 4) for v in sim.d.qpos[:2]])
        if sim.state()["fallen"]:
            break
    s = sim.state()
    p0 = np.array(path[0])
    leg1, leg2 = p_turn - p0, sim.d.qpos[:2] - p_turn

    def ang(v):
        return round(float(np.rad2deg(np.arctan2(v[1], v[0]) - np.arctan2(f0[1], f0[0])) + 180) % 360 - 180, 1)
    ok = (not s["fallen"] and np.linalg.norm(leg1) > 0.1 and abs(ang(leg1)) < 15
          and abs(((ang(leg2) - turn) + 180) % 360 - 180) < 30)
    return dict(ok=bool(ok), turn=turn, leg1_m=round(float(np.linalg.norm(leg1)), 3), leg1_dir_deg=ang(leg1),
                leg2_m=round(float(np.linalg.norm(leg2)), 3), leg2_dir_deg=ang(leg2), jumps=sim.jumps,
                skid_touches=sim.getups, path=path, psi0=psi0)


def path_figure(runs, path="media/pogo_small_paths.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, turn in zip(axs, (90, -90)):
        for r in [r for r in runs if r["turn"] == turn]:
            P_ = np.array(r["path"])
            c, s_ = np.cos(-r["psi0"]), np.sin(-r["psi0"])
            Q = (P_ - P_[0]) @ np.array([[c, s_], [-s_, c]])     # rotate: start heading -> +y
            ax.plot(100 * Q[:, 0], 100 * Q[:, 1], lw=1.2)
        ax.plot(0, 0, "ko")
        ax.annotate("", (0, 12), (0, 0), arrowprops=dict(arrowstyle="->", lw=2))
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("sideways (cm)"); ax.set_ylabel("forward (cm)")
        ax.set_title(f"hop forward 24 s, turn {'left 90' if turn > 0 else 'right 90 (long way round)'}, "
                     "hop forward\n8 noise seeds, top view")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def drivetrain_figure(path="media/pogo_small_drivetrain.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    boxes = [
        ("2x 3S 450 mAh LiPo\n(end pods = weights)", 0.02, 0.78, "#d9b21a"),
        ("small FOC board\n~40 A peak", 0.24, 0.78, "#2f8f4e"),
        ("2806 drone motor\nKV1300, ~48 g", 0.46, 0.78, "#555"),
        ("steel ring flywheel\n60 mm, bolted to the bell", 0.68, 0.78, "#8a8f99"),
        ("centrifugal clutch\n(printed shoes, 20 rad/s)", 0.68, 0.52, "#6b6f78"),
        ("sprag one-way\nneedle bearing", 0.46, 0.52, "#6b6f78"),
        ("printed 20:1\n2-stage spur gears", 0.24, 0.52, "#6b6f78"),
        ("printed snail cam\n20 mm rise / 144 deg + step", 0.02, 0.52, "#b33"),
        ("roller follower +\nbell-crank + braided line", 0.02, 0.26, "#6b6f78"),
        ("foot slider (4 mm rod)\nin spring, 2 N/mm", 0.24, 0.26, "#c9a227"),
        ("rebound escapement:\ninertial pawl + rack +\none-way rotary damper", 0.46, 0.26, "#6b6f78"),
        ("external tracking\n(camera/markers) -> ESP32\nradio -> balance MCU", 0.68, 0.26, "#246"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    for text, x, y, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 0.2, 0.16, boxstyle="round,pad=0.01", fc=c, ec="k", alpha=0.9))
        ax.text(x + 0.1, y + 0.08, text, ha="center", va="center", color="w", fontsize=9)
    arrows = [((0.22, 0.86), (0.24, 0.86)), ((0.44, 0.86), (0.46, 0.86)), ((0.66, 0.86), (0.68, 0.86)),
              ((0.78, 0.78), (0.78, 0.68)), ((0.68, 0.60), (0.66, 0.60)), ((0.46, 0.60), (0.44, 0.60)),
              ((0.24, 0.60), (0.22, 0.60)), ((0.12, 0.52), (0.12, 0.42)), ((0.22, 0.34), (0.24, 0.34)),
              ((0.44, 0.34), (0.46, 0.34))]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.5, 0.99, "Small Pogo Cubli drivetrain: ONE motor balances (reaction wheel) and winds / fires the leg spring",
            ha="center", va="top", fontsize=11, weight="bold")
    ax.text(0.5, 0.13, "Wheel above 20 rad/s -> clutch in -> the cam winds the spring for 8 wheel turns (0.4 cam turn x 20) "
            "and drops off its step -> jump.\nWheel parked at -200 rad/s -> clutch out -> balancing 'stick'. "
            "The same wheel flips the body 360 deg in the air and lifts it off the skids after a fall.",
            ha="center", va="center", fontsize=9)
    ax.set_xlim(0, 0.9); ax.set_ylim(0.05, 1.0); ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def video(path="media/pogo_small"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio
    import mujoco
    from PIL import Image, ImageDraw
    sim = PogoSim(PP, seed=0)
    r = mujoco.Renderer(sim.m, 544, 960)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 1.25, 25.0, -12.0
    cam.lookat[:] = [0, 0, 0.12]
    frames, slow, k = [], [], [0]
    push = dict(t0=None, f=np.zeros(3))
    cam.distance = 1.6

    def h():
        sim.d.xfrc_applied[:] = 0
        if push["t0"] is not None and push["t0"] <= sim.d.time < push["t0"] + 0.1:
            sim.d.xfrc_applied[sim.body, :3] = push["f"]
    sim.substep_hook = h
    captions = [(0.0, "stands as a stick"), (1.0, "hop"), (9.0, "stop -> stick"), (12.0, "somersault"),
                (19.0, "knocked over -> gets up by itself"), (25.0, "hop forward (holding heading)"),
                (47.0, "turn left 90 deg (wheel-speed steering)"), (58.0, "hop forward on the new heading"), (80.0, "")]
    sched = [(1.0, "hop"), (9.0, "stop"), (12.0, "flip"), (25.0, "hop_fwd"), (47.0, "turn_left")]

    def cap(t):
        c = ""
        for tc, txt in captions:
            if t >= tc:
                c = txt
        return c

    every = int(round(0.04 / sim.Ts))
    while sim.d.time < 80.0:
        if sched and sim.d.time >= sched[0][0]:
            sim.set_mode(sched.pop(0)[1])
        if push["t0"] is None and sim.d.time >= 19.5:
            push["t0"], push["f"] = sim.d.time, np.array([0.0, 1.2, 0.0])
        sim.step()
        k[0] += 1
        fl = sim.flipping and sim.airborne
        if k[0] % every and not (fl and k[0] % 2 == 0):
            continue
        com = sim.d.xipos[sim.body]
        cam.lookat[:] = 0.9 * np.asarray(cam.lookat) + 0.1 * np.array([com[0], com[1], 0.12])
        r.update_scene(sim.d, camera=cam)
        s = sim.state()
        im = Image.fromarray(r.render().copy())
        dr = ImageDraw.Draw(im)
        hd = float(np.rad2deg(sim.heading()))
        dr.text((16, 12), f"t={s['t']:5.2f}s  heading {hd:+5.0f} deg  {cap(s['t'])}   [{s['mode']}/{s['phase']}]  jumps={s['jumps']}  "
                          f"flips={sim.flips}  get-ups={sim.getups}  wheel={s['wheel']:+.0f} rad/s  "
                          f"battery used {1000 * s['energy_Wh']:.1f} mWh", fill=(255, 255, 255))
        img = np.asarray(im)
        if fl:
            s2 = Image.fromarray(img.copy())
            ImageDraw.Draw(s2).text((16, 30), "somersault replay, 5x slow motion", fill=(255, 220, 90))
            slow.append(np.asarray(s2))
        if k[0] % every == 0:
            frames.append(img)
    allf = frames[:int(19.0 / 0.04)] + slow + frames[int(19.0 / 0.04):]
    imageio.mimsave(path + ".mp4", allf, fps=25, quality=8)
    imageio.mimsave(path + ".gif", [f[::3, ::3] for f in allf[::2]], duration=1 / 12.5, loop=0)
    return len(allf)


if __name__ == "__main__":
    if "--video" in sys.argv:
        print("frames:", video())
        sys.exit()
    bom = SMALL.report(PP)
    eps = round(inertia_ratio(PP)[0], 3)
    fallen_jobs = [(a, b, 0) for a, b in [(5, 0), (-5, 0), (0, 3), (0, -3), (4, 3), (-4, -3), (4, -3), (-4, 3)]]
    fallen_jobs += [(5, 0, s) for s in (1, 2, 3)]
    knock_jobs = [(a, F, 1) for a in (0, 45, 90, 180, 270) for F in (0.6, 1.5)]
    with Pool(24) as pool:
        jobs = dict(stick=pool.map_async(stick, range(8)), hop_stop=pool.map_async(hop_stop, range(8)),
                    flip=pool.map_async(flip, range(8)), fallen=pool.map_async(fallen, fallen_jobs),
                    knock=pool.map_async(knock, knock_jobs), forward=pool.map_async(forward, range(8)),
                    go=pool.map_async(go, [(sd, t) for t in (90, -90) for sd in range(8)]))
        res = {k: v.get() for k, v in jobs.items()}
    drivetrain_figure()
    path_figure(res["go"])
    hs = res["hop_stop"]
    power_W = float(np.mean([h["energy_Wh"] for h in hs])) * 3600 / 28.0
    summary = {k: f"{sum(r['ok'] for r in v)}/{len(v)}" for k, v in res.items()}
    summary.update(
        inertia_ratio_eps=eps,
        hops_per_20s=[h["hops_in_20s"] for h in hs], extra_jumps_after_stop=[h["extra_after_stop"] for h in hs],
        apex_cm=float(np.mean([h["apex_cm"] for h in hs])),
        forward_cm_per_30s=[round(100 * f["forward_m"]) for f in res["forward"]],
        sideways_cm_per_30s=[round(100 * f["sideways_m"]) for f in res["forward"]],
        forward_skid_touches=[f["skid_touches"] for f in res["forward"]],
        go_left=[(g["leg1_m"], g["leg1_dir_deg"], g["leg2_m"], g["leg2_dir_deg"]) for g in res["go"] if g["turn"] > 0],
        go_right=[(g["leg1_m"], g["leg1_dir_deg"], g["leg2_m"], g["leg2_dir_deg"]) for g in res["go"] if g["turn"] < 0],
        getup_time_s=[f["t_up_s"] for f in res["fallen"]],
        mean_power_hopping_W=round(power_W, 2),
        runtime_hopping_h=round(bom["battery"]["energy_Wh"] * 0.8 / power_W, 1),
    )
    print(json.dumps({k: v for k, v in bom.items() if k != "parts"}, indent=1))
    print(json.dumps(summary, indent=1))
    json.dump(dict(bom=bom, params={k: (v if not hasattr(v, "bom") else "SMALL") for k, v in PP.__dict__.items()},
                   summary=summary, runs=res), open("results/pogo_small.json", "w"), indent=1, default=str)
