"""Can the paper's One-Wheel Cubli do a somersault?  (free-body model with floor
contacts: cubli/somersault.py)   python scripts/somersault.py [--video]

1. Rest pose: after a fall it rolls over its small housing and lies upside down
   on the cantilever and housing top.
2. Somersault: spin the wheel to 420 rad/s (the reaction presses the body into the
   floor, so it stays put), then drive it hard to -420 rad/s: the momentum dump
   rolls the body a full turn about the cantilever axis and it lands upside down
   again. Checked over floor friction mu = 0.5 / 0.8 / 1.2.
3. Jump-up to balance (flip up and catch on the tip): the catch region of the
   balance LQR on the free body is only ~3-5 deg at rest, and the throw arrives
   within a few degrees but still moving with no wheel authority left. Not achieved.
"""
import json, os, sys
sys.path.insert(0, ".")
import numpy as np
import mujoco
from multiprocessing import Pool

from cubli.somersault import Maneuver, place, tilt_angles
from cubli.sim import Motor
from cubli.params import NOMINAL as P


def rest_pose(_):
    M = Maneuver(); M.settle_upside_down()
    a, b, w, R = tilt_angles(M.m, M.d)
    return dict(alpha_deg=round(float(np.rad2deg(a)), 1), upside_down=bool(R[2, 2] < -0.8))


def somersault(args):
    mu, w_rev = args
    M = Maneuver(mu=mu); M.settle_upside_down()
    L = M.run(+0.6, 420, catch=False, T=6, w_rev=w_rev)
    return dict(mu=mu, reverse_to=w_rev, rolled_deg=round(float(abs(L[-1, 3]))),
                full_turn=bool(abs(L[-1, 3]) >= 330 and L[-1, 6] < -0.8))


def catch_region(tilt):
    M = Maneuver(); m, d = M.m, M.d
    place(m, d, np.deg2rad(tilt), 0.0)
    motor = Motor(P)
    for _ in range(600):
        u = float(-(M.K @ M.state9())); u, _ = motor.limit(u, d.qvel[M.wd], 0.01); d.ctrl[0] = u
        for _ in range(20):
            mujoco.mj_step(m, d)
        if tilt_angles(m, d)[3][2, 2] < 0.8:
            return tilt, False
    return tilt, True


def video(path="media/somersault"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio
    M = Maneuver(); M.settle_upside_down()
    r = mujoco.Renderer(M.m, 540, 960)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0.15]; cam.distance = 2.0; cam.azimuth = 90; cam.elevation = -12
    frames = []
    M.run(+0.6, 420, catch=False, T=4.0, w_rev=420, frames=frames, frame_every=40, renderer=r, camera=cam)
    imageio.mimsave(path + ".mp4", frames, fps=50, quality=8)
    return len(frames)


if __name__ == "__main__":
    if "--video" in sys.argv:
        print("frames:", video())
        sys.exit()
    with Pool(24) as pool:
        rp = pool.apply_async(rest_pose, (None,))
        ss = pool.map_async(somersault, [(mu, w) for mu in (0.5, 0.8, 1.2) for w in (160, 260, 320, 420)])
        cr = pool.map_async(catch_region, [1, 3, 5, 8, 12])
        rp, ss, cr = rp.get(), ss.get(), cr.get()
    print("rest pose after a fall:", rp)
    for s in ss:
        print("somersault", s)
    print("balance LQR catch region on the free body (tilt deg -> recovers):", dict(cr))
    json.dump(dict(rest=rp, somersault=ss, catch_region=dict(cr)), open("results/somersault.json", "w"), indent=1)
