"""Build the project home page (index.html at the repo root, served by GitHub Pages)
from the results files, so every number on the page comes from the simulations.

  python scripts/make_site.py
"""
import json
import os
import html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "https://github.com/U4AR/one-wheel-cubli-mujoco"


def load(name):
    with open(os.path.join(ROOT, "results", name)) as f:
        return json.load(f)


def e(x):
    return html.escape(str(x))


small = load("pogo_small.json")
big = load("pogo.json")
bom, S = small["bom"], small["summary"]
B = big["summary"]

parts_rows = "\n".join(
    f"<tr><td>{e(p['name'])}</td><td class=num>{p['mass_g']:.0f}</td>"
    f"<td class=num>{('$' + format(p['cost_usd'], '.0f')) if p['cost_usd'] else '—'}</td>"
    f"<td class=note>{e(p['note'])}</td></tr>" for p in bom["parts"])

fw, gl, gr = S["forward_cm_per_30s"], S["go_left"], S["go_right"]
getup_t = [t for t in S["getup_time_s"] if t]

results_small = [
    ("Stand still as a “stick”", S["stick"], "tilt stays under 1.5°"),
    ("Hop continuously, then stop", S["hop_stop"],
     f"{min(S['hops_per_20s'])}–{max(S['hops_per_20s'])} hops per 20 s, apex ≈ {S['apex_cm']:.0f} cm, "
     f"0–1 jumps after “stop”"),
    ("Somersault in the air and land", S["flip"], "360° about the bar in ≈ 0.3 s"),
    ("Get up after a fall", S["fallen"],
     f"from resting on a skid or battery pod, up in {min(getup_t):.1f}–{max(getup_t):.1f} s"),
    ("Get up after being knocked over", S["knock"], "10 pushes of 0.6–1.5 N from 5 directions"),
    ("Hop forward, holding heading", S["forward"],
     f"{min(fw)}–{max(fw)} cm straight ahead in 30 s, ≤ {max(abs(x) for x in S['sideways_cm_per_30s'])} cm sideways, "
     f"{sum(S['forward_skid_touches'])} skid touches"),
    ("Forward → turn left 90° → forward", f"{sum(1 for g in small['runs']['go'] if g['ok'] and g['turn'] > 0)}/8",
     f"second leg at {min(g[3] for g in gl):.0f}…{max(g[3] for g in gl):.0f}°"),
    ("Forward → turn right 90° → forward", f"{sum(1 for g in small['runs']['go'] if g['ok'] and g['turn'] < 0)}/8",
     f"second leg at {min(g[3] for g in gr):.0f}…{max(g[3] for g in gr):.0f}° (turns the long way round)"),
]
res_rows = "\n".join(f"<tr><td>{e(a)}</td><td class=score>{e(b)}</td><td class=note>{e(c)}</td></tr>"
                     for a, b, c in results_small)

n_tests = sum(open(os.path.join(ROOT, "tests", fn)).read().count("\ndef test_")
              for fn in os.listdir(os.path.join(ROOT, "tests")) if fn.startswith("test_"))
m, f, bat, sp, cam = bom["motor"], bom["flywheel"], bom["battery"], bom["spring"], bom["cam"]

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pogo Cubli</title>
<meta name="description" content="A one-motor robot that balances on a point, hops, somersaults, gets up after falls and steers — simulated in MuJoCo, from the One-Wheel Cubli paper to a $110 drone-parts build.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #f7f6f2; --bg2: #efede6; --card: #ffffff; --ink: #16161a; --ink2: #4a4a52; --ink3: #7a7a84;
    --line: #dedbd2; --accent: #d9481c; --accent2: #2f6fd6; --good: #1e8a4c; --pod: #c99a06;
    --shadow: 0 1px 2px rgba(0,0,0,.05), 0 8px 24px rgba(0,0,0,.06);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #111114; --bg2: #18181c; --card: #1d1d22; --ink: #f2f1ec; --ink2: #c4c3bb; --ink3: #8d8c86;
      --line: #2e2e34; --accent: #ff6a3d; --accent2: #6aa3ff; --good: #3ecf7c; --pod: #f0c230;
      --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.35);
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #111114; --bg2: #18181c; --card: #1d1d22; --ink: #f2f1ec; --ink2: #c4c3bb; --ink3: #8d8c86;
    --line: #2e2e34; --accent: #ff6a3d; --accent2: #6aa3ff; --good: #3ecf7c; --pod: #f0c230;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.35);
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{ margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.6 Inter, system-ui, sans-serif; }}
  a {{ color: var(--accent2); text-decoration: none; }} a:hover {{ text-decoration: underline; }}
  .wrap {{ max-width: 1120px; margin: 0 auto; padding: 0 16px; }}
  nav {{ position: sticky; top: 0; z-index: 10; background: color-mix(in srgb, var(--bg) 88%, transparent);
        backdrop-filter: blur(10px); border-bottom: 1px solid var(--line); }}
  nav .wrap {{ display: flex; align-items: center; gap: 18px; height: 54px; overflow-x: auto; white-space: nowrap; }}
  nav .brand {{ font-weight: 800; color: var(--ink); letter-spacing: -.02em; margin-right: auto; }}
  nav a.l {{ color: var(--ink2); font-size: 14px; font-weight: 500; }}
  nav .gh {{ background: var(--ink); color: var(--bg); padding: 6px 12px; border-radius: 99px; font-size: 13px; font-weight: 600; }}
  header.hero {{ padding: 56px 0 24px; }}
  .hero-grid {{ display: grid; grid-template-columns: 1.05fr 1fr; gap: 40px; align-items: center; }}
  @media (max-width: 880px) {{ .hero-grid {{ grid-template-columns: 1fr; gap: 24px; }} }}
  .kicker {{ font: 600 12px/1 "JetBrains Mono", monospace; letter-spacing: .12em; text-transform: uppercase; color: var(--accent); }}
  h1 {{ font-size: clamp(34px, 5.4vw, 60px); line-height: 1.02; letter-spacing: -.035em; margin: 14px 0 18px; font-weight: 800; }}
  h1 em {{ font-style: normal; color: var(--accent); }}
  .lede {{ font-size: 18px; color: var(--ink2); max-width: 36em; }}
  .cta {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 22px; }}
  .btn {{ display: inline-block; padding: 11px 18px; border-radius: 10px; font-weight: 600; font-size: 15px; }}
  .btn.p {{ background: var(--accent); color: #fff; }} .btn.s {{ border: 1px solid var(--line); color: var(--ink); background: var(--card); }}
  .btn:hover {{ text-decoration: none; filter: brightness(1.06); }}
  .vid {{ border-radius: 16px; overflow: hidden; background: #000; box-shadow: var(--shadow); border: 1px solid var(--line); }}
  .vid video, .vid img {{ display: block; width: 100%; height: auto; }}
  .stats {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 1px; background: var(--line); border: 1px solid var(--line);
           border-radius: 14px; overflow: hidden; margin: 36px 0 8px; }}
  @media (max-width: 880px) {{ .stats {{ grid-template-columns: repeat(3, 1fr); }} }}
  .stat {{ background: var(--card); padding: 16px 14px; }}
  .stat b {{ display: block; font-size: 26px; letter-spacing: -.02em; }}
  .stat span {{ font-size: 13px; color: var(--ink3); }}
  section {{ padding: 56px 0 8px; }}
  h2 {{ font-size: clamp(26px, 3.4vw, 36px); letter-spacing: -.025em; margin: 8px 0 10px; line-height: 1.1; }}
  .sub {{ color: var(--ink2); max-width: 46em; margin: 0 0 24px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .grid3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
  @media (max-width: 880px) {{ .grid2, .grid3 {{ grid-template-columns: 1fr; }} }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 20px; box-shadow: var(--shadow); }}
  .card h3 {{ margin: 0 0 6px; font-size: 18px; letter-spacing: -.01em; }}
  .card p {{ margin: 6px 0 0; color: var(--ink2); font-size: 15px; }}
  .card .vid {{ margin: -4px -4px 14px; border-radius: 12px; box-shadow: none; }}
  .step {{ font: 600 12px/1 "JetBrains Mono", monospace; color: var(--ink3); }}
  .timeline {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }}
  @media (max-width: 880px) {{ .timeline {{ grid-template-columns: 1fr 1fr; }} }}
  .tl {{ border-left: 3px solid var(--accent); padding: 4px 0 4px 12px; }}
  .tl b {{ display: block; font-size: 15px; }} .tl span {{ font-size: 14px; color: var(--ink2); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14.5px; }}
  th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }}
  th {{ font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--ink3); font-weight: 600; }}
  td.num {{ text-align: right; font-family: "JetBrains Mono", monospace; white-space: nowrap; }}
  td.note {{ color: var(--ink3); font-size: 13.5px; }}
  td.score {{ font-weight: 700; color: var(--good); white-space: nowrap; font-family: "JetBrains Mono", monospace; }}
  tfoot td {{ font-weight: 700; border-bottom: 0; }}
  .tablewrap {{ overflow-x: auto; background: var(--card); border: 1px solid var(--line); border-radius: 16px; box-shadow: var(--shadow); }}
  .specs {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 18px; }}
  @media (max-width: 880px) {{ .specs {{ grid-template-columns: 1fr 1fr; }} }}
  .spec {{ background: var(--bg2); border-radius: 12px; padding: 14px; }}
  .spec h4 {{ margin: 0 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: .06em; color: var(--ink3); }}
  .spec dl {{ margin: 0; display: grid; grid-template-columns: auto auto; gap: 3px 10px; font-size: 14px; }}
  .spec dt {{ color: var(--ink2); }} .spec dd {{ margin: 0; text-align: right; font-family: "JetBrains Mono", monospace; }}
  .chain {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 18px 0; }}
  .chain span {{ background: var(--card); border: 1px solid var(--line); padding: 7px 11px; border-radius: 9px; font-size: 14px; font-weight: 500; }}
  .chain i {{ color: var(--ink3); font-style: normal; }}
  .chain .hot {{ border-color: var(--accent); color: var(--accent); }}
  figure {{ margin: 0; }} figcaption {{ font-size: 13.5px; color: var(--ink3); margin-top: 8px; }}
  img.fig {{ width: 100%; border-radius: 12px; border: 1px solid var(--line); background: #fff; }}
  .shots {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
  @media (max-width: 880px) {{ .shots {{ grid-template-columns: 1fr; }} }}
  .shots img {{ width: 100%; border-radius: 10px; border: 1px solid var(--line); display: block; }}
  pre {{ background: #15151a; color: #e8e6df; padding: 16px 18px; border-radius: 12px; overflow-x: auto;
        font: 13.5px/1.55 "JetBrains Mono", monospace; }}
  .modes li {{ margin: 6px 0; color: var(--ink2); }} .modes b {{ color: var(--ink); }}
  .limits li {{ margin: 8px 0; color: var(--ink2); }}
  .tag {{ display: inline-block; font: 600 11px/1 "JetBrains Mono", monospace; padding: 4px 7px; border-radius: 6px;
         background: var(--bg2); color: var(--ink2); margin-right: 6px; }}
  footer {{ border-top: 1px solid var(--line); margin-top: 64px; padding: 28px 0 48px; color: var(--ink3); font-size: 14px; }}
  .theme {{ background: none; border: 1px solid var(--line); color: var(--ink2); border-radius: 99px; padding: 5px 10px; cursor: pointer; font: inherit; font-size: 13px; }}
</style>
</head>
<body>
<nav><div class="wrap">
  <a class="brand" href="#">Pogo Cubli</a>
  <a class="l" href="#journey">Journey</a>
  <a class="l" href="#how">How it works</a>
  <a class="l" href="#parts">Parts</a>
  <a class="l" href="#results">Results</a>
  <a class="l" href="#run">Run it</a>
  <button class="theme" id="theme" aria-label="Toggle theme">◐</button>
  <a class="gh" href="{REPO}">GitHub</a>
</div></nav>

<header class="hero"><div class="wrap">
  <div class="hero-grid">
    <div>
      <div class="kicker">MuJoCo simulation · one motor</div>
      <h1>Balance. Hop. Flip.<br>Fall, <em>get up</em>, turn.</h1>
      <p class="lede">A robot that balances on a single point using <b>one</b> motor — and uses that same motor to wind and fire a pogo spring. It started as a faithful replica of ETH Zürich's <i>One-Wheel Cubli</i> and ended as a 0.3 kg, ≈${bom['total_cost_usd']} design built from drone parts.</p>
      <div class="cta">
        <a class="btn p" href="#parts">See the parts list</a>
        <a class="btn s" href="{REPO}">Source code</a>
        <a class="btn s" href="media/pogo_small.mp4">Full video</a>
      </div>
    </div>
    <div class="vid"><video src="media/pogo_small.mp4" autoplay muted loop playsinline preload="auto"
      poster="media/poster_pogo_small.jpg"></video></div>
  </div>
  <div class="stats">
    <div class="stat"><b>1</b><span>motor does everything</span></div>
    <div class="stat"><b>{bom['total_mass_kg'] * 1000:.0f} g</b><span>total mass</span></div>
    <div class="stat"><b>≈${bom['total_cost_usd']}</b><span>parts cost</span></div>
    <div class="stat"><b>{S['apex_cm']:.0f} cm</b><span>hop height</span></div>
    <div class="stat"><b>{S['flip']}</b><span>somersaults landed</span></div>
    <div class="stat"><b>{S['go']}</b><span>forward-turn-forward runs</span></div>
  </div>
</div></header>

<section id="journey"><div class="wrap">
  <div class="kicker">The journey</div>
  <h2>From a paper replica to a robot you could build</h2>
  <p class="sub">Every step below is a separate experiment in the repo, each with its own script, test and write-up.</p>
  <div class="timeline">
    <div class="tl"><b>1 · Replicate the paper</b><span>Model, 5-IMU estimator, delay-compensating Kalman filter and LQR; the linearisation matches the paper's Eq. 11.</span></div>
    <div class="tl"><b>2 · Tune &amp; explore</b><span>CMA-ES tuning, yaw control with friction, offset flywheels, hoops and ovals instead of end masses.</span></div>
    <div class="tl"><b>3 · Roll &amp; somersault</b><span>A rolling hoop with speed control (two motors, then one plus a gyro); the paper's cube rolls a full somersault.</span></div>
    <div class="tl"><b>4 · One motor, many tricks</b><span>A pogo leg wound by the balance motor; then a small drone-parts robot that gets up and steers.</span></div>
  </div>
  <div class="grid2" style="margin-top:28px">
    <div class="card"><div class="vid"><video src="media/cubli_tuned.mp4" poster="media/poster_cubli_tuned.jpg" muted loop playsinline preload="metadata" controls></video></div>
      <span class="step">01 · PAPER REPLICA</span><h3>The One-Wheel Cubli, balancing</h3>
      <p>One reaction wheel at 45° steers two tilt axes, because the long cantilever makes pitch much slower than roll. Tuned controller recovering from drops and pushes.</p></div>
    <div class="card"><div class="vid"><video src="media/somersault.mp4" poster="media/poster_somersault.jpg" muted loop playsinline preload="metadata" controls></video></div>
      <span class="step">02 · FREE BODY</span><h3>Somersault on the floor</h3>
      <p>Spin the wheel up to 420 rad/s, then dump the momentum: the cube rolls a full turn about its bar. Jumping back up onto the tip stays out of reach for one wheel.</p></div>
    <div class="card"><div class="vid"><video src="media/pogo.mp4" poster="media/poster_pogo.jpg" muted loop playsinline preload="metadata" controls></video></div>
      <span class="step">03 · POGO CROSS</span><h3>Hop, stop, somersault in the air</h3>
      <p>The paper's cube on a spring leg. The balance motor winds a cam through a clutch; the spring fires it ≈18 cm up; a 360° flip lands 16/16 times.</p></div>
    <div class="card"><div class="vid"><video src="media/pogo_small.mp4" poster="media/poster_pogo_small.jpg" muted loop playsinline preload="metadata" controls></video></div>
      <span class="step">04 · BUILDABLE</span><h3>Small robot: gets up, walks, turns</h3>
      <p>Half size, drone parts, batteries as the end weights. Hops, somersaults, gets up from its kickstands, hops forward holding heading and turns 90°.</p></div>
  </div>
</div></section>

<section id="how"><div class="wrap">
  <div class="kicker">How it works</div>
  <h2>One motor, four jobs</h2>
  <p class="sub">The motor drives a flywheel that balances the body. The same shaft, through a clutch that only engages above 20 rad/s, winds the leg spring. The wheel's speed alone decides whether it balances, hops, or steers.</p>
  <div class="chain">
    <span>2× 3S LiPo</span><i>→</i><span>FOC board</span><i>→</i><span class="hot">2806 drone motor</span><i>=</i><span class="hot">steel-ring flywheel</span><i>→</i>
    <span>centrifugal clutch</span><i>→</i><span>one-way bearing</span><i>→</i><span>20:1 printed gears</span><i>→</i><span class="hot">snail cam</span><i>→</i><span>spring leg</span>
  </div>
  <div class="grid2">
    <figure><img class="fig" src="media/pogo_small_drivetrain.png" alt="Drivetrain diagram: battery, FOC board, motor, flywheel, clutch, one-way bearing, gears, cam, follower, spring leg, escapement, tracking"><figcaption>Drivetrain. The cam winds the spring for 8 wheel turns, then drops off its step: the leg fires.</figcaption></figure>
    <div class="card">
      <h3>What the controller does</h3>
      <ul class="modes">
        <li><b>Balance</b> — an LQR on the foot, like the paper's, running at 200 Hz.</li>
        <li><b>Hop</b> — spin above the clutch speed; the cam winds and releases the spring. Park the wheel below it to stand still.</li>
        <li><b>Fly</b> — a foot-contact switch hands the motor to a flight law the instant the foot lifts; a rebound catch stops landings bouncing.</li>
        <li><b>Somersault</b> — brake and reverse the wheel in the air; a guidance law stops the rotation exactly at 360°.</li>
        <li><b>Get up</b> — skids and battery pods are kickstands; the wheel lifts the body along the curve that coasts to upright.</li>
        <li><b>Hop forward</b> — a push-pull torque pulse leans it 1–2° just before the spring fires, with no leftover spin.</li>
        <li><b>Turn</b> — the flywheel axis is tilted 20° up, so the balance wheel speed becomes the steering.</li>
      </ul>
    </div>
  </div>
</div></section>

<section id="parts"><div class="wrap">
  <div class="kicker">Bill of materials</div>
  <h2>Everything on board: {bom['total_mass_kg'] * 1000:.0f} g, ≈${bom['total_cost_usd']}</h2>
  <p class="sub">The simulation is built from this list: the mass, centre of mass and inertia are summed part by part, and the motor model uses its KV, winding resistance, battery voltage and driver current limit. Parts are representative classes with typical catalogue values — check against the exact part you buy.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Part</th><th style="text-align:right">g</th><th style="text-align:right">$</th><th>Notes</th></tr></thead>
    <tbody>
{parts_rows}
    </tbody>
    <tfoot><tr><td>Total</td><td class=num>{bom['total_mass_kg'] * 1000:.0f}</td><td class=num>${bom['total_cost_usd']}</td><td class=note>span {bom['span_m']:.1f} m</td></tr></tfoot>
  </table></div>
  <div class="specs">
    <div class="spec"><h4>Motor</h4><dl>
      <dt>Peak torque</dt><dd>{m['peak_torque_Nm']:.2f} N·m</dd>
      <dt>Continuous</dt><dd>{m['cont_torque_Nm']:.2f} N·m</dd>
      <dt>No-load</dt><dd>{m['no_load_rad_s']} rad/s</dd>
      <dt>Kt</dt><dd>{m['kt_Nm_per_A'] * 1000:.2f} mN·m/A</dd></dl></div>
    <div class="spec"><h4>Flywheel</h4><dl>
      <dt>Inertia</dt><dd>{f['I_kg_m2'] * 1e7:.0f} g·cm²</dd>
      <dt>Mass</dt><dd>{f['mass_g']:.0f} g</dd>
      <dt>Speed limit</dt><dd>{f['software_speed_limit_rad_s']:.0f} rad/s</dd>
      <dt>Rim speed</dt><dd>{f['rim_speed_at_limit_m_s']:.0f} m/s</dd></dl></div>
    <div class="spec"><h4>Battery</h4><dl>
      <dt>Packs</dt><dd>{bat['packs']} × 3S 450 mAh</dd>
      <dt>Energy</dt><dd>{bat['energy_Wh']:.0f} Wh</dd>
      <dt>Hopping power</dt><dd>{S['mean_power_hopping_W']:.1f} W</dd>
      <dt>Runtime</dt><dd>≈{S['runtime_hopping_h']:.1f} h</dd></dl></div>
    <div class="spec"><h4>Spring &amp; cam</h4><dl>
      <dt>Spring</dt><dd>{sp['k_N_per_mm']:.0f} N/mm</dd>
      <dt>Stored</dt><dd>{sp['stored_J']:.1f} J</dd>
      <dt>Cam rise</dt><dd>{cam['rise_mm']:.0f} mm / {cam['wind_deg']}°</dd>
      <dt>Motor load</dt><dd>{cam['motor_torque_while_winding_Nm'] * 1000:.0f} mN·m</dd></dl></div>
  </div>
</div></section>

<section id="results"><div class="wrap">
  <div class="kicker">Results</div>
  <h2>Tested on 8 random-noise runs each</h2>
  <p class="sub">Sensor noise, delays, motor torque–speed limits and thermal derating are all simulated. State comes from external tracking (camera + markers) and a foot load cell.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Small robot</th><th>Passed</th><th>Detail</th></tr></thead>
    <tbody>
{res_rows}
    </tbody>
  </table></div>
  <div class="grid2" style="margin-top:22px">
    <figure><img class="fig" src="media/pogo_small_paths.png" alt="Top-view paths: hop forward, turn left or right 90 degrees, hop forward; eight runs each"><figcaption>Top view of 16 runs: hop forward 24 s, turn 90°, hop forward. Left turns are quick; right turns go the long way round so the winding clutch stays disengaged.</figcaption></figure>
    <figure><img class="fig" src="media/pogo_timeline.png" alt="Timeline of the paper-size pogo cross: hopping, stop, stick, two somersaults"><figcaption>Paper-size pogo cross: {B['survived']}/{B['seeds']} runs survive a 56 s routine, {B['flips_landed']}/{B['flips_attempted']} somersaults land, apex ≈ {B['apex_mean_cm']:.0f} cm.</figcaption></figure>
  </div>
</div></section>

<section id="viewer"><div class="wrap">
  <div class="kicker">Live viewer</div>
  <h2>Push it, knock it over, steer it — in your browser</h2>
  <p class="sub">The repo ships a live web viewer running the real MuJoCo plant and controllers: drag to orbit, shift-drag to pull on any part, buttons for every behaviour, and live charts of tilt, torque, wheel speed and motor heat.</p>
  <div class="shots">
    <figure><img src="media/pogo_small_ui_getup.png" alt="Viewer: robot resting on a skid, getting up"><figcaption>Knocked onto a skid, getting up.</figcaption></figure>
    <figure><img src="media/pogo_small_ui_turn.png" alt="Viewer: robot turning after forward hops"><figcaption>Turning toward a new heading after forward hops.</figcaption></figure>
    <figure><img src="media/pogo_ui_flip.png" alt="Viewer: paper-size robot mid-somersault"><figcaption>Paper-size cross, mid-somersault.</figcaption></figure>
  </div>
</div></section>

<section id="run"><div class="wrap">
  <div class="kicker">Run it</div>
  <h2>Three commands</h2>
  <pre>git clone {REPO}.git && cd one-wheel-cubli-mujoco
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
MUJOCO_GL=egl python app/server.py        # open http://127.0.0.1:8765, Plant → Layout → POGO SMALL

python scripts/pogo_small.py              # parts list + every behaviour on 8 seeds → results/pogo_small.json
python -m pytest -q tests                 # {n_tests} tests</pre>
</div></section>

<section id="limits"><div class="wrap">
  <div class="kicker">Honest limits</div>
  <h2>What this is — and isn't</h2>
  <ul class="limits">
    <li><span class="tag">SIM</span>Everything here is simulated. The cam, clutch and landing catch are idealised, and part specs are typical values.</li>
    <li><span class="tag">SENSING</span>Tilt and heading are assumed to come from external tracking; an on-board-only version would need gyro integration in flight.</li>
    <li><span class="tag">TURNING</span>The turn rate depends on the foot's twisting friction, which is an assumption. Right turns are slow because they would otherwise engage the winding clutch.</li>
    <li><span class="tag">SPEED</span>It is slow: about 3 cm per hop, ≈1 cm/s.</li>
    <li><span class="tag">FALLS</span>Get-up works from the kickstands; a robot flipped completely onto its back cannot right itself.</li>
  </ul>
</div></section>

<footer><div class="wrap">
  Based on M. Hofer, M. Muehlebach and R. D'Andrea, “The One-Wheel Cubli”, <i>Mechatronics</i> 91 (2023), ETH Zürich, CC BY 4.0 —
  <a href="https://doi.org/10.1016/j.mechatronics.2023.102965">doi:10.1016/j.mechatronics.2023.102965</a>.
  This is an independent simulation, not affiliated with the authors. · <a href="{REPO}">Source on GitHub</a>
</div></footer>

<script>
  // autoplay card videos only while they are on screen
  const io = new IntersectionObserver(es => es.forEach(en => {{
    const v = en.target;
    if (en.isIntersecting) v.play().catch(() => {{}}); else if (!v.autoplay) v.pause();
  }}), {{ threshold: 0.4 }});
  document.querySelectorAll(".card video").forEach(v => io.observe(v));
  const root = document.documentElement, btn = document.getElementById("theme");
  try {{ const t = localStorage.getItem("theme"); if (t) root.dataset.theme = t; }} catch (e) {{}}
  btn.onclick = () => {{
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try {{ localStorage.setItem("theme", root.dataset.theme); }} catch (e) {{}}
  }};
</script>
</body>
</html>
"""

with open(os.path.join(ROOT, "index.html"), "w") as fh:
    fh.write(page)
print("wrote index.html", len(page), "bytes;", n_tests, "tests")
