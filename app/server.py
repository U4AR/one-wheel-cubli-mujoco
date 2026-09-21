"""Live, interactive One-Wheel Cubli viewer.

Runs the MuJoCo plant + the paper's estimator/controller in real time on this
machine, streams rendered frames and telemetry to the browser over a websocket,
and takes commands back (forces, mouse grab, controller/plant settings, jobs).

    MUJOCO_GL=egl python app/server.py            # then open http://localhost:8765
    (remote machine: ssh -L 8765:localhost:8765 <host>)
"""
import asyncio
import json
import os
import re
import sys
import time
from collections import deque

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from aiohttp import web, WSMsgType

from cubli.live import LiveSim
from cubli.live_rolling import LiveRolling
from cubli.rolling_kane import gyro_design
from cubli.live_somersault import LiveSomersault
from cubli.live_pogo import LivePogo
from cubli.pogo_hw import small_params
from cubli.params import NOMINAL
from cubli.tunings import LATEST_FILE

HOST, PORT = os.environ.get("CUBLI_HOST", "127.0.0.1"), int(os.environ.get("CUBLI_PORT", 8765))
FPS = 30
PY = sys.executable

pivot_sim = LiveSim()
rolling_sim = None          # built on first use (~8 s: speed-scheduled controller)
single_sim = None           # one-motor (hub + passive gyro) rolling hoop
free_sim = None             # free-body cube on the floor (somersault)
pogo_sim = None             # pogo cross: one motor balances + winds the spring
pogo_small_sim = None       # small drone-parts pogo robot
sim = pivot_sim
clients: set = set()
ctl = dict(paused=False, speed=1.0)
job = dict(proc=None, name=None, lines=deque(maxlen=400), tune=[])
GEN_RE = re.compile(r"gen\s+(\d+)\s+best\s+(-?[\d.]+)\s+gen-min\s+(-?[\d.]+)\s+median\s+(-?[\d.]+)")

# hoop variants (scripts/ring_design.py)
RING_PRESETS = {
    # balances, but fragile: small hoop through the pivot + weights at 3 and 9 o'clock
    "ring_weights": dict(R=0.2, M=0.1, w=0.3, middle=False),
    # the "housing in the middle" hoop with the paper's pitch inertia: cannot balance
    "ring_middle": dict(R=0.3196, M=0.7427, middle=True),
    # flattened (elliptical) hoops, housing in the middle, 90 % of the hoop mass in
    # weights at 3 and 9 o'clock; same total mass and pitch inertia as the paper
    # (scripts/ring_ellipse.py). These balance with the paper's motor.
    "oval_035": dict(R=0.5196, b=0.1819, M=0.0743, w=0.3342, c=0.2312, raise_=0.0845, tuning="tuned"),
    "oval_025": dict(R=0.5506, b=0.1377, M=0.0743, w=0.3342, c=0.2049, raise_=0.0581, tuning="tuned"),
    "oval_018": dict(R=0.5686, b=0.1023, M=0.0743, w=0.3342, c=0.1873, raise_=0.0405, tuning="tuned"),
}

# pulse presets: body, unit force direction (world frame)
PRESETS = {
    "drop_right": ("endmass2", (0, 0, -1)),   # object dropped on end mass 2 -> +beta
    "drop_left": ("endmass1", (0, 0, -1)),    # end mass 1 -> -beta
    "push_py": ("housing", (0, 1, 0)),        # sideways on the housing -> roll
    "push_ny": ("housing", (0, -1, 0)),
    "push_px": ("housing", (1, 0, 0)),
    "push_nx": ("housing", (-1, 0, 0)),
    "twist": ("endmass2", (0, 1, 0)),         # horizontal on an end mass -> yaw + beam
}


async def broadcast(msg):
    dead = []
    for ws in list(clients):
        try:
            if isinstance(msg, bytes):
                await ws.send_bytes(msg)
            else:
                await ws.send_str(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


def past_tuning_curve():
    """Best-cost-per-generation of the tuning run that produced the shipped tuning."""
    out = []
    for fn in ["results/tune_log.txt"]:
        p = os.path.join(ROOT, fn)
        if os.path.exists(p):
            for line in open(p):
                m = GEN_RE.search(line)
                if m:
                    out.append([int(m[1]), float(m[2]), float(m[4])])
    return out


def job_state():
    return dict(type="job", name=job["name"], running=job["proc"] is not None,
                tune=job["tune"], latest_available=os.path.exists(LATEST_FILE))


async def sim_loop():
    period = 1.0 / FPS
    carry = 0.0
    last = time.perf_counter()
    while True:
        now = time.perf_counter()
        dt_wall, last = now - last, now
        try:
            if not ctl["paused"]:
                carry += dt_wall * ctl["speed"]
                n = min(int(carry / sim.Ts), 20)
                carry -= n * sim.Ts
                carry = min(carry, sim.Ts)
                for _ in range(n):
                    sim.control_step()
            if clients:
                await broadcast(sim.render_jpeg())
                await broadcast(json.dumps(dict(
                    type="telemetry", samples=sim.pop_samples(), state=sim.state(),
                    paused=ctl["paused"], speed=ctl["speed"],
                    grab=None if sim.grab is None else float(np.linalg.norm(sim.grab["force"])))))
            else:
                sim.pop_samples()
        except Exception as e:  # keep serving even if one step goes wrong
            print(f"sim loop error: {type(e).__name__}: {e}", flush=True)
            await asyncio.sleep(0.5)
        await asyncio.sleep(max(0.0, period - (time.perf_counter() - now)))


async def run_job(name, args):
    if job["proc"] is not None:
        return
    print(f"{time.strftime('%H:%M:%S')} job start: {name} {args}", flush=True)
    job.update(name=name, tune=[])
    job["lines"].clear()
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    proc = await asyncio.create_subprocess_exec(
        "nice", "-n", "15", PY, "-W", "ignore", *args, cwd=ROOT, env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    job["proc"] = proc
    await broadcast(json.dumps(job_state()))
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        job["lines"].append(line)
        m = GEN_RE.search(line)
        if m:
            job["tune"].append([int(m[1]), float(m[2]), float(m[4])])
        await broadcast(json.dumps(dict(type="log", line=line,
                                        tune=job["tune"][-1] if m else None)))
    rc = await proc.wait()
    job["proc"] = None
    await broadcast(json.dumps(dict(type="log", line=f"[{name} finished, exit code {rc}]")))
    await broadcast(json.dumps(job_state()))


def handle(cmd):
    global sim, rolling_sim, single_sim, free_sim, pogo_sim, pogo_small_sim
    c = cmd.get("cmd")
    if c == "pulse" and hasattr(sim, "kick"):
        sim.kick(cmd["preset"], float(cmd.get("force", 1.5)), float(cmd.get("duration", 0.05)))
    elif c == "maneuver" and hasattr(sim, "start"):
        name = cmd.get("name")
        if name in ("somersault", "jump"):
            sim.start(name, w_spin=float(cmd.get("w_spin", 420)),
                      w_rev=None if cmd.get("w_rev") is None else float(cmd["w_rev"]))
        elif name in ("hop", "stop", "stick", "flip", "hop_fwd", "knock"):
            sim.start(name)
        elif name == "settle":
            sim.settle()
        elif name == "stand":
            sim.stand(float(cmd.get("tilt", 2.0)))
    elif c == "speed_target" and hasattr(sim, "set_speed"):
        sim.set_speed(float(cmd.get("speed", 0.0)))
    elif c == "pulse":
        body, direction = PRESETS[cmd["preset"]]
        F = float(cmd.get("force", 1.5))
        point = None
        if sim.plant.layout == "ring" and body.startswith("endmass"):
            # hoop layouts: hit the hoop at 3 / 9 o'clock (where its tip weights sit)
            sx = 1.0 if body == "endmass2" else -1.0
            body, point = "ring", (sx * sim.plant.ring_radius, 0.0, 0.0)
        sim.pulse(body, np.array(direction, float) * F, float(cmd.get("duration", 0.05)), point)
    elif c == "grab_start":
        return dict(type="grabbed", body=sim.grab_start(cmd["x"], cmd["y"]))
    elif c == "grab_move":
        sim.grab_move(cmd["x"], cmd["y"], stiffness=float(cmd.get("k", 15.0)))
    elif c == "grab_end":
        sim.grab_end()
    elif c == "orbit":
        sim.cam.azimuth -= 180 * cmd["dx"]
        sim.cam.elevation = float(np.clip(sim.cam.elevation - 90 * cmd["dy"], -89, 10))
    elif c == "zoom":
        sim.cam.distance = float(np.clip(sim.cam.distance * (1.0015 ** cmd["delta"]), 0.4, 5))
    elif c == "camera_reset":
        sim.reset_camera()
    elif c == "reset":
        sim.reset(tuple(cmd.get("tilt", (2.0, -1.5))))
    elif c == "pause":
        ctl["paused"] = bool(cmd["value"])
    elif c == "speed":
        ctl["speed"] = float(np.clip(cmd["value"], 0.05, 2.0))
    elif c == "yaw":
        sim.set_yaw(bool(cmd.get("on", sim.yaw_on)))
        sim.heading_ref = float(np.deg2rad(cmd.get("heading_deg", np.rad2deg(sim.heading_ref))))
    elif c == "controller":
        sim.controller_on = bool(cmd["value"])
    elif c == "settings":
        # controller-side settings: rebuild the controller, keep the physical state
        sim.tuning_name = cmd.get("tuning", sim.tuning_name)
        sim.com_enable = bool(cmd.get("com_enable", sim.com_enable))
        sim.noise_scale = float(cmd.get("noise", sim.noise_scale))
        new_delay = int(cmd.get("delay", sim.delay_steps))
        sim.make_controller()
        if new_delay != sim.delay_steps:
            sim.delay_steps = new_delay
            sim.buf = [sim.measure()] * (new_delay + 1)
    elif c == "plant":
        # physical changes need a new model -> rebuild and reset
        layout = cmd.get("layout", "bar")
        if layout == "rolling_hoop":
            if rolling_sim is None:
                rolling_sim = LiveRolling()
            sim = rolling_sim
            sim.reset((2.0, 0.0))
            return dict(type="layout", layout=layout)
        if layout == "somersault":
            if free_sim is None:
                free_sim = LiveSomersault()
            sim = free_sim
            sim.settle()
            return dict(type="layout", layout=layout)
        if layout == "pogo_small":
            if pogo_small_sim is None:
                pogo_small_sim = LivePogo(pp=small_params())
            sim = pogo_small_sim
            sim.reset()
            return dict(type="layout", layout=layout)
        if layout == "pogo":
            if pogo_sim is None:
                pogo_sim = LivePogo()
            sim = pogo_sim
            sim.reset()
            return dict(type="layout", layout=layout)
        if layout == "rolling_single":
            if single_sim is None:
                single_sim = LiveRolling(rp=gyro_design())
            sim = single_sim
            sim.reset((2.0, 0.0))
            return dict(type="layout", layout=layout)
        sim = pivot_sim
        p = NOMINAL.with_(m_e=float(cmd["m_e"]), com_offset_xy=(float(cmd["com_mm"]) / 1e3,) * 2,
                          wheel_tilt=float(np.deg2rad(cmd.get("tilt_deg", 0.0))),
                          wheel_ecc=float(cmd.get("ecc_mm", 0.0)) / 1e3)
        if not cmd.get("pivot_friction", True):
            p = p.with_(yaw_friction=0.0, yaw_damping=0.0)
        if layout != "bar":
            mh, lS, *_ = NOMINAL.housing_without_tube()
            ring = RING_PRESETS[layout]
            c = ring.get("c", 1.05 * ring["R"])
            raise_ = ring.get("raise_", (c - lS) if ring.get("middle") else 0.0)
            p = p.with_(layout="ring", ring_mass=ring["M"], ring_radius=ring["R"],
                        ring_b=ring.get("b", 0.0), ring_center=c,
                        ring_weights=ring.get("w", 0.0), core_raise=raise_)
            sim.tuning_name = ring.get("tuning", "ring")
        if layout == "bar" and sim.tuning_name == "ring":
            sim.tuning_name = "tuned"
        p = p.with_(k=p.k_from_freq(float(cmd["f_beam"])))
        sim.plant = p
        sim.build()
        if layout != "bar":
            sim.reset((0.5, -0.5))    # hoops only recover from ~1 deg initial tilt
    elif c == "job":
        if cmd["name"] == "tune":
            gens, pop = str(int(cmd.get("gens", 10))), str(int(cmd.get("pop", 24)))
            warm = "cubli/tuned_params.json" if cmd.get("warm", True) else "none"
            asyncio.ensure_future(run_job("tune", ["scripts/tune.py", gens, warm,
                                                   "results/tuned_params_ui.json", pop]))
        elif cmd["name"] == "benchmark":
            asyncio.ensure_future(run_job("benchmark", ["scripts/make_figures.py"]))
        elif cmd["name"] == "tests":
            asyncio.ensure_future(run_job("tests", ["-m", "pytest", "-q", "tests"]))
    elif c == "job_stop":
        if job["proc"] is not None:
            job["proc"].terminate()
    return None


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=1 << 20)
    await ws.prepare(request)
    clients.add(ws)
    await ws.send_str(json.dumps(dict(type="hello", past_tune=past_tuning_curve(),
                                      log=list(job["lines"]), fps=FPS)))
    await ws.send_str(json.dumps(job_state()))
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    reply = handle(json.loads(msg.data))
                except Exception as e:  # never let a bad command kill the loop
                    reply = dict(type="error", message=f"{type(e).__name__}: {e}")
                if reply:
                    await ws.send_str(json.dumps(reply))
    finally:
        clients.discard(ws)
    return ws


async def media_list(_):
    """Every figure / video / result produced by the experiments, for the UI gallery."""
    out = []
    for folder in ("media", "results"):
        base = os.path.join(ROOT, folder)
        for fn in sorted(os.listdir(base)):
            if fn.endswith((".png", ".gif", ".mp4", ".json")) and not fn.startswith("tuned_params_ui"):
                out.append(dict(url=f"/{folder}/{fn}", name=fn, folder=folder))
    return web.json_response(out)


async def index(_):
    return web.FileResponse(os.path.join(ROOT, "app", "index.html"))


async def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/media", os.path.join(ROOT, "media"))
    app.router.add_static("/results", os.path.join(ROOT, "results"))
    app.router.add_get("/api/media", media_list)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, HOST, PORT).start()
    print(f"One-Wheel Cubli live viewer on http://{HOST}:{PORT}", flush=True)
    await sim_loop()


if __name__ == "__main__":
    asyncio.run(main())
