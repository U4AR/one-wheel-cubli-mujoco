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
from cubli.params import NOMINAL
from cubli.tunings import LATEST_FILE

HOST, PORT = os.environ.get("CUBLI_HOST", "127.0.0.1"), int(os.environ.get("CUBLI_PORT", 8765))
FPS = 30
PY = sys.executable

sim = LiveSim()
clients: set = set()
ctl = dict(paused=False, speed=1.0)
job = dict(proc=None, name=None, lines=deque(maxlen=400), tune=[])
GEN_RE = re.compile(r"gen\s+(\d+)\s+best\s+(-?[\d.]+)\s+gen-min\s+(-?[\d.]+)\s+median\s+(-?[\d.]+)")

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
    c = cmd.get("cmd")
    if c == "pulse":
        body, direction = PRESETS[cmd["preset"]]
        F = float(cmd.get("force", 1.5))
        sim.pulse(body, np.array(direction, float) * F, float(cmd.get("duration", 0.05)))
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
        p = NOMINAL.with_(m_e=float(cmd["m_e"]), com_offset_xy=(float(cmd["com_mm"]) / 1e3,) * 2)
        p = p.with_(k=p.k_from_freq(float(cmd["f_beam"])))
        sim.plant = p
        sim.build()
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


async def index(_):
    return web.FileResponse(os.path.join(ROOT, "app", "index.html"))


async def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/media", os.path.join(ROOT, "media"))
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, HOST, PORT).start()
    print(f"One-Wheel Cubli live viewer on http://{HOST}:{PORT}", flush=True)
    await sim_loop()


if __name__ == "__main__":
    asyncio.run(main())
