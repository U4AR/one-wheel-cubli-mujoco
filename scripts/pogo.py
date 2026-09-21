"""Pogo Cubli experiment: one motor balances AND winds/fires the pogo spring.

Scenario (per seed): stand as a stick -> hop continuously -> stop -> stick ->
somersault -> stick -> second somersault -> stick.

  python scripts/pogo.py            # 8 seeds -> results/pogo.json + media/pogo_timeline.png
  python scripts/pogo.py --video    # media/pogo.mp4 / media/pogo.gif
"""
import os
import sys
import json
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cubli.pogo import PogoSim, PogoParams, inertia_ratio  # noqa: E402

SCHEDULE = [(1.0, "hop"), (21.0, "stop"), (32.0, "flip"), (44.0, "flip")]
T_END = 56.0


def scenario(seed, schedule=SCHEDULE, T=T_END, log_every=1, frame_cb=None):
    sim = PogoSim(seed=seed)
    sched = list(schedule)
    log, events = [], []
    hop_jumps = None
    t_stick_after_stop = None
    max_tilt_hop = 0.0
    while sim.d.time < T:
        if sched and sim.d.time >= sched[0][0]:
            t, cmd = sched.pop(0)
            sim.set_mode(cmd)
            events.append((round(sim.d.time, 2), cmd))
            if cmd == "stop":
                hop_jumps = sim.jumps
        sim.step()
        s = sim.state()
        if s["mode"] == "hop":
            max_tilt_hop = max(max_tilt_hop, abs(s["alpha"]), abs(s["beta"]))
        if hop_jumps is not None and t_stick_after_stop is None and s["phase"] == "stick":
            t_stick_after_stop = round(s["t"], 2)
        if len(log) == 0 or int(round(s["t"] / sim.Ts)) % log_every == 0:
            log.append([round(s["t"], 3), round(s["alpha"], 3), round(s["beta"], 3), round(s["z"], 4),
                        round(s["leg"], 4), round(s["wheel"], 1), round(sim.u, 3), s["phase"],
                        round(float(np.rad2deg(sim.flip_angle)) if sim.flipping else 0.0, 1)])
        if frame_cb is not None:
            frame_cb(sim)
        if s["fallen"]:
            break
    s = sim.state()
    return dict(seed=seed, fell=bool(s["fallen"]), t_end=round(s["t"], 2), events=events,
                hops_before_stop=hop_jumps, jumps_total=sim.jumps, flips=sim.flips,
                apex_mean_cm=round(100 * float(np.mean(sim.apex)), 1) if sim.apex else 0.0,
                max_tilt_while_hopping_deg=round(max_tilt_hop, 2), stick_after_stop_at=t_stick_after_stop,
                final_phase=s["phase"], final_tilt_deg=round(abs(s["alpha"]) + abs(s["beta"]), 2),
                drift_m=round(float(np.hypot(s["x"], s["y"])), 3)), log


def _run(seed):
    return scenario(seed, log_every=2)


def timeline(result, log, path="media/pogo_timeline.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = np.array([r[:7] for r in log], float)
    ph = [r[7] for r in log]
    fl = np.array([r[8] for r in log], float)
    t = L[:, 0]
    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(t, 100 * (L[:, 3] - L[0, 3]), lw=0.8, color="tab:blue")
    ax[0].set_ylabel("CoM height\nchange (cm)")
    ax[1].plot(t, L[:, 1], lw=0.8, label="alpha (roll, bar axis)")
    ax[1].plot(t, L[:, 2], lw=0.8, label="beta (pitch)")
    ax[1].set_ylim(-8, 8); ax[1].set_ylabel("tilt (deg)"); ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(t, L[:, 5], lw=0.8, color="tab:orange"); ax[2].axhline(PogoParams().w_clutch, ls=":", c="k", lw=0.8)
    ax[2].set_ylabel("wheel speed\n(rad/s)")
    ax[3].plot(t, fl, lw=0.8, color="tab:purple"); ax[3].set_ylabel("flip angle\n(deg)")
    ax[3].set_xlabel("time (s)")
    colors = {"stick": "#dfe", "settle": "#ffd", "wind": "#fdd", "flight": "#ddf"}
    start = 0
    for i in range(1, len(ph) + 1):
        if i == len(ph) or ph[i] != ph[start]:
            for a in ax:
                a.axvspan(t[start], t[min(i, len(t) - 1)], color=colors.get(ph[start], "w"), lw=0)
            start = i
    for te, cmd in result["events"]:
        for a in ax:
            a.axvline(te, color="k", lw=0.8, ls="--")
        ax[0].text(te, ax[0].get_ylim()[1] * 0.9, " " + cmd, fontsize=9)
    ax[0].set_title("Pogo Cubli: stick -> continuous hopping -> stop -> stick -> somersault x2 "
                    "(green stick, yellow settle, red winding, blue flight)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def video(path="media/pogo"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio
    import mujoco
    from PIL import Image, ImageDraw
    sched = [(0.8, "hop"), (8.0, "stop"), (11.0, "flip")]
    sim0 = PogoSim(seed=0)
    r = mujoco.Renderer(sim0.m, 544, 960)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.4, 15.0, -8.0
    cam.lookat[:] = [0, 0, 0.25]
    frames, slow, k = [], [], [0]

    def label(img, text):
        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((16, 12), text, fill=(255, 255, 255))
        return np.asarray(im)

    def grab(sim):
        k[0] += 1
        flying_flip = sim.flipping and sim.airborne
        if k[0] % 4 and not flying_flip:
            return
        com = sim.d.xipos[sim.body]
        cam.lookat[:] = 0.9 * np.asarray(cam.lookat) + 0.1 * np.array([com[0], com[1], 0.25])
        r.update_scene(sim.d, camera=cam)
        s = sim.state()
        img = label(r.render().copy(), f"t={s['t']:5.2f}s  {s['mode']}/{s['phase']}  jumps={s['jumps']}  "
                                       f"flips={sim.flips}  wheel={s['wheel']:+.0f} rad/s")
        if flying_flip:
            slow.append(label(img, "\n\n4x slow motion"))
        if k[0] % 4 == 0:
            frames.append(img)

    res, _ = scenario(0, schedule=sched, T=22.0, frame_cb=grab)
    allf = frames + slow
    imageio.mimsave(path + ".mp4", allf, fps=25, quality=8)
    small = [f[::3, ::3] for f in frames[::2]] + [f[::3, ::3] for f in slow]
    imageio.mimsave(path + ".gif", small, duration=1 / 12.5, loop=0)
    return len(allf), res


if __name__ == "__main__":
    if "--video" in sys.argv:
        n, res = video()
        print("frames:", n, res)
        sys.exit()
    eps = {f"l_E={lE}": round(inertia_ratio(PogoParams(l_E=lE))[0], 3) for lE in (0.5975, 0.7, 0.8, 0.9)}
    with Pool(8) as pool:
        out = pool.map(_run, range(8))
    results = [o[0] for o in out]
    for r_ in results:
        print(r_)
    timeline(results[0], out[0][1])
    summary = dict(
        seeds=len(results),
        survived=sum(not r_["fell"] for r_ in results),
        flips_landed=sum(r_["flips"] for r_ in results),
        flips_attempted=2 * len(results),
        hops_before_stop=[r_["hops_before_stop"] for r_ in results],
        apex_mean_cm=round(float(np.mean([r_["apex_mean_cm"] for r_ in results])), 1),
        max_tilt_while_hopping_deg=max(r_["max_tilt_while_hopping_deg"] for r_ in results),
        ended_as_stick=sum(r_["final_phase"] == "stick" for r_ in results),
    )
    print(summary)
    json.dump(dict(params=PogoParams().__dict__, inertia_ratio_eps=eps, schedule=SCHEDULE,
                   summary=summary, runs=results), open("results/pogo.json", "w"), indent=1)
