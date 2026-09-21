"""Render the tuned controller balancing and rejecting disturbances.
Usage: MUJOCO_GL=egl python scripts/make_video.py [paper|tuned]"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, ".")
import numpy as np
import mujoco
import imageio

from cubli.model import load
from cubli.sim import run, Disturbance
from cubli.tunings import get

which = sys.argv[1] if len(sys.argv) > 1 else "tuned"
tuning = get(which)
fps, dt = 50, 5e-4
m, _ = load()
renderer = mujoco.Renderer(m, 540, 960)
frames = []
dist = [Disturbance(3.0, 0.05, "endmass2", (0, 0, -1.5)),   # drop on end mass (beta)
        Disturbance(7.0, 0.05, "housing", (0, 6.0, 0)),     # push housing (alpha)
        Disturbance(11.0, 0.05, "endmass1", (0, 0, -1.5))]
r = run(T=15.0, tuning=tuning, x0=(np.deg2rad(3), np.deg2rad(-2)), disturbances=dist,
        frames=frames, frame_every=int(round(1 / (fps * dt))), renderer=renderer,
        camera="front")
print("fell:", r.fell, "frames:", len(frames))
os.makedirs("media", exist_ok=True)
imageio.mimsave(f"media/cubli_{which}.mp4", frames, fps=fps, quality=8)
small = [f[::2, ::2] for f in frames[::2]]
imageio.mimsave(f"media/cubli_{which}.gif", small, duration=1000 * 2 / fps, loop=0)
