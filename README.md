# One-Wheel Cubli in MuJoCo

A simulation replica of

> M. Hofer, M. Muehlebach, R. D'Andrea, **"The One-Wheel Cubli: A 3D inverted pendulum that can balance with a single reaction wheel"**, *Mechatronics* 91 (2023) 102965. [doi:10.1016/j.mechatronics.2023.102965](https://doi.org/10.1016/j.mechatronics.2023.102965) (open access, CC BY 4.0)

It is built in [MuJoCo](https://mujoco.org), with the paper's full estimation and control stack and a tuned controller.

![balancing](media/cubli_tuned.gif)

*Tuned controller: it recovers from a 3°/−2° initial tilt, a 1.5 N drop on one end mass (t = 3 s), a 6 N sideways push on the housing (t = 7 s) and a drop on the other end mass (t = 11 s). The full-resolution MP4 is [media/cubli_tuned.mp4](media/cubli_tuned.mp4).*

## What the paper does

The system balances on one corner. It has two tilt degrees of freedom (roll α, pitch β) and only **one** reaction wheel, mounted at 45° between the two tilt axes. It is controllable only because a 1.25 m cantilever with two 0.3 kg end masses makes the pitch inertia much larger than the roll inertia. The resulting time-scale separation (natural-frequency ratio ≈ 1/(1+√2), the inverse *silver ratio*) lets a single torque steer both angles. The controller is a delay-compensating steady-state Kalman filter plus an LQR on a 14-state model that includes the cantilever bending modes.

## What is replicated here

| Paper | This repo |
|---|---|
| 4-body model, generalised coords (α, β, γ, φ, δ₁, δ₂), App. A | `cubli/model.py`. MJCF generated from the Table A.1 parameters. The housing gets three stacked hinges x→y→z, which is exactly the paper's Euler chain. The cantilevers are hinges about body-z at Q, with the identified torsional stiffness k and damping d. |
| Linearised, reduced model, Eq. (11) | `cubli/linearize.py`. Finite-difference linearisation of the MuJoCo model, then the same reduction (drop φ, γ). |
| 5-IMU accelerometer tilt estimate, Eq. (8) | `cubli/controller.py:imu_weights`. MuJoCo accelerometer/gyro sites: 4 in a square near the pivot and 1 far away. |
| Complementary filter, 0.98 / 0.02 (Sec. 5.2) | `Controller.tilt` |
| 1-step-delay-augmented steady-state KF, Eqs. (18)–(21) | `Controller.step` |
| Discrete LQR at Ts = 10 ms, Eqs. (22)–(23); KF tuning, Eq. (24) | `cubli/controller.py:design` |
| CoM-offset estimation by low-pass filtering α, β (Sec. 6) | `Controller.step` |
| Simulation with torque limits, I²t overheat protection, delay, noise (Sec. 7.1) | `cubli/sim.py` |
| Fig. 5 (disturbance rejection), Fig. 6 (beam-frequency sensitivity) | `scripts/make_figures.py` |

### Check 1: the linear model matches Eq. (11)

The MuJoCo model is built only from Table A.1, and its linearisation reproduces the paper's A and B matrices to the printed precision (asserted in `tests/test_replication.py`):

| entry | paper | MuJoCo |
|---|---|---|
| ∂α̈/∂α | 57.7 | 57.7 |
| ∂α̈/∂β | 0.1 | 0.1 |
| ∂β̈/∂β | 10.6 | 10.6 |
| ∂φ̈/∂α, ∂φ̈/∂β | −40.9, −7.5 | −40.8, −7.5 |
| ∂δ̈₁/∂α, ∂δ̈₁/∂δ̇₁, ∂δ̈₁/∂δ̇₂ | −4.2, −19.8, −12.6 | −4.2, −19.8, −12.5 |
| ∂α̈/∂δ̇ᵢ, ∂φ̈/∂δ̇ᵢ | ±4.6, ∓3.2 | ±4.6, ∓3.2 |
| B (α̈, β̈, φ̈, δ̈₁, δ̈₂) | −20.7, −2.3, 1122, 7.4, −7.4 | −20.7, −2.3, 1122.0, 7.4, −7.4 |
| ε = π_β/π_α | 0.43 | 0.428 |

The only mismatch is the β̈ ← δ coupling: the paper prints 75.8, MuJoCo gives 83.3. The paper does not print the other large beam entries (∗).

### Check 2: it balances with the paper's design

With the paper's structure and weights (Eqs. 23–24), the simulated system balances indefinitely. This holds with sensor noise, sensor bias, a 10 ms measurement delay, the torque envelope, I²t derating, and a deliberately injected 2 mm CoM offset. The disturbance response has the same qualitative signature as the paper's Fig. 5. A push in +β first makes the controller command a *negative* torque, which drives α up quickly (the fast axis). A positive torque then brings both angles back together along the level lines of the control law.

![disturbance](media/disturbance_response.png)
![alpha-beta](media/alpha_beta_plane.png)

The paper says its Fig. 5 disturbance was "approximately the maximum disturbance from which the system can recover". The same holds here. The recoverable drop on an end mass is small (~2 N for 50 ms), because correcting β takes α excursions about 9× larger (B: −20.7 vs −2.3). Those excursions push the wheel toward its no-load speed.

### Where this simulation disagrees with the paper

**Cantilever sensitivity (Fig. 6) is not reproduced.** The paper reports that a ±1 Hz error in the assumed cantilever frequency causes visible structural oscillations. It also reports that ignoring the flexibility altogether destabilises the loop. In this simulation, with the paper's k and d, neither happens:

- The torque and α spectra are almost identical for controllers that assume 56.2, 57.2 or 58.2 Hz.
- A controller designed on a rigid-beam model (k → 10⁸) also balances.

![beam](media/beam_sensitivity.png)

The mechanisms behind the real-hardware effect are not modelled. Plausible candidates are the motor current-loop dynamics, torque ripple, IMU mounting on the flexible structure, anti-aliasing filters, and the 57 Hz mode sitting above the 50 Hz Nyquist frequency. The beam model and the controller's beam states are still fully implemented, so any of those effects can be added.

## Tuning

`scripts/tune.py` runs CMA-ES (40 generations × 40 candidates, warm-started from a shorter first pass) over 20 log-scale multipliers on the paper's weights:

- the 9 LQR state weights plus a shared weight on the delayed states;
- the process- and measurement-noise groups of the KF;
- the complementary-filter weight;
- the CoM-estimator time constant.

The objective (`cubli/evaluate.py:cost`) combines:

- tilt jitter, RMS torque (heating) and mean wheel speed while balancing for 60 s;
- the largest recoverable drop on an end mass and push on the housing (bisection);
- survival and jitter under model errors: beam at 56.2 / 58.2 Hz, end masses +10 %, roll inertia +10 %, a 2-step delay, and 3× sensor noise.

The result is stored in `cubli/tuned_params.json`.

| metric (sim, 3 noise seeds) | paper weights | tuned | change |
|---|---|---|---|
| max recoverable drop on end mass (N, 50 ms) | 2.10 | **2.38** | +13 % |
| max recoverable push on housing top (N, 50 ms) | 8.73 | **10.49** | +20 % |
| tilt jitter α / β (std, deg) | 0.092 / 0.041 | **0.067 / 0.038** | −27 % / −7 % |
| RMS motor torque while balancing (N m) | 0.083 | **0.044** | −47 % |
| mean wheel speed after 30–60 s (rad/s) | 12.1 | **≈0** | |
| survives all 6 model-error scenarios | ✓ | ✓ | |
| torque RMS with 3× sensor noise (N m) | 0.247 | **0.129** | −48 % |

What changed (see `cubli/tuned_params.json` and `cubli/tunings.py`):

- **LQR weights were redistributed, but the net gains barely moved.** The α weight fell 50× and the β weight 2×. The wheel-speed weight rose 2.4× and the weight on the delayed states rose about 140×. The resulting gains are close to the paper's (α: −133 → −146, β: 268 → 294 N m/rad).
- **The Kalman filter trusts the gyro rates much more.** Rate measurement noise went down about 240× and rate process noise up about 20×. This removes estimator lag on α̇ and β̇ and cuts the noise reaching the motor.
- **Much faster CoM-offset estimator** (τ = 20 s → 0.8 s). It acts as an integrator on the balance point, so the wheel speed now averages to zero within seconds. That keeps full torque headroom available for the next disturbance. Ablation: the paper weights with only τ = 0.83 s already reach a 2.27 N drop / 9.4 N push, about half of the total gain. The tuned weights with the old τ = 20 s fall back to 1.73 N / 8.2 N.
- **Complementary filter leans harder on the gyro** (accelerometer weight 0.02 → 0.002).

One trade-off shows in the disturbance plot. The tuned controller uses less peak torque and about half the I²t heating, but it undershoots more on the way back (β reaches about −3.6° instead of −2.2°).

## Assumptions (not given in the paper)

These are flagged with `ASSUMED` in `cubli/params.py`, `cubli/sim.py` and `cubli/controller.py`:

- **Motor envelope.** No-load speed 450 rad/s, stall torque 8 N m, and an I²t budget of about 1 s at peak torque. The paper gives only the 3.4 N m peak and 0.5 N m continuous figures. The no-load speed matters most: it sets the maximum recoverable disturbance.
- **Sensor noise.** MPU-6050-class accelerometer noise 0.04 m/s², gyro noise 0.004 rad/s, residual biases, and Hall-sensor speed noise 0.3 rad/s.
- **IMU positions.** Only the layout (4 near the pivot + 1 far away) is from the paper.
- **State normalisation used before applying Q and R.** The paper normalises but does not give the scales. The printed LQR gain (Eq. 25) is therefore not directly comparable, although its sign pattern (−, −, +, +, +, −, −, +, +) matches.
- **Model additions.** A 2 mm CoM offset toward the motor, and small yaw friction at the pivot.
- **Wheel inertia.** The paper's wheel inertia includes the motor rotor, so I_wx > I_wy + I_wz, which is not a valid single rigid body. The transverse wheel inertia is raised just enough to be physical. That is a change of about 4·10⁻⁵ kg m² against a housing inertia of about 3·10⁻² kg m², and Check 1 is unaffected.

## Usage

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

pytest -q                                       # replication checks
python -c "from cubli.sim import run; r = run(T=10); print('fell:', r.fell)"
MUJOCO_GL=egl python scripts/make_figures.py    # figures + results/benchmark.json
MUJOCO_GL=egl python scripts/make_video.py tuned  # or: paper
python scripts/tune.py 40 cubli/tuned_params.json # re-run the CMA-ES tuning (uses 40 cores)
```

Interactive viewer:

```bash
python -c "import mujoco.viewer; from cubli.model import load; m, d = load(); mujoco.viewer.launch(m, d)"
```

## Layout

```
cubli/params.py      Table A.1 parameters (+ flagged assumptions)
cubli/model.py       MJCF generator
cubli/linearize.py   linearisation / reduction / ZOH discretisation
cubli/controller.py  IMU fusion, complementary filter, delay-KF, LQR, CoM estimator
cubli/sim.py         closed loop: plant, sensors, delay, motor envelope + I²t
cubli/evaluate.py    benchmark scenarios, metrics, tuning objective
cubli/tunings.py     "paper" and "tuned" controller settings
scripts/             tuning, figures, video
tests/               replication checks
```

## Credit

The system design, model, parameters, and estimation/control architecture are from Hofer, Muehlebach and D'Andrea (2023), ETH Zürich, licensed CC BY 4.0. This repository is an independent simulation re-implementation and is not affiliated with the authors.
