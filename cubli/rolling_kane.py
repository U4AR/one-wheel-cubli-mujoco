"""Linear model of the rolling hoop about straight, upright rolling at speed v,
derived with Kane's method (rolling without slip is a nonholonomic constraint).

Bodies: thin hoop (radius r, mass M), housing hanging from the axle (CoM d below
it), reaction wheel in the housing with its axis at eta between the lean axis
and the axle. Frames follow SymPy's rolling-disk example:
  N -(yaw q1 about N.z)-> Y -(lean q2 about Y.x)-> L -(spin q3 about L.y)-> hoop
  housing H = L rotated by q4 about L.y ; wheel W = H rotated by q5 about u.
Speeds: u1, u2, u3 = hoop angular velocity along L.x, L.y, L.z; u4 = q4'; u5 = q5'.
Linear state used by the controller:  x = (q2, u1, u3, q4, u4, u5)
  lean, lean rate, yaw rate, housing pitch (rel. lean frame), its rate, wheel speed
(the hoop's spin u2 sets the rolling speed v = r*u2 and is the scheduling variable).
"""
import functools
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me

from .rolling import RollingParams, P


@functools.lru_cache(maxsize=1)
def _symbolic():
    q1, q2, q3, q4, q5 = me.dynamicsymbols("q1:6")
    u1, u2, u3, u4, u5 = me.dynamicsymbols("u1:6")
    r, M, d, g, eta, b, tw = sm.symbols("r M d g eta b t_w")
    mh, Icx, Icy, Icz = sm.symbols("m_h I_cx I_cy I_cz")
    mw, Iw, Iwt = sm.symbols("m_w I_w I_wt")
    T = me.dynamicsymbols("T")
    Th = me.dynamicsymbols("T_h")          # hub motor: hoop <-> housing, about the axle

    N = me.ReferenceFrame("N")
    Y = N.orientnew("Y", "Axis", [q1, N.z])
    L = Y.orientnew("L", "Axis", [q2, Y.x])
    Rf = L.orientnew("R", "Axis", [q3, L.y])
    w_R_qd = Rf.ang_vel_in(N)
    Rf.set_ang_vel(N, u1 * L.x + u2 * L.y + u3 * L.z)
    w_L = u1 * L.x + u3 * sm.tan(q2) * L.y + u3 * L.z        # hoop minus its own spin
    L.set_ang_vel(N, w_L)

    C = me.Point("C"); C.set_vel(N, 0)
    Dh = C.locatenew("Dh", r * L.z)
    Dh.v2pt_theory(C, N, Rf)                                 # no-slip rolling

    H = L.orientnew("H", "Axis", [q4, L.y])
    H.set_ang_vel(N, w_L + u4 * L.y)
    Ph = Dh.locatenew("Ph", -d * H.z)
    Ph.v2pt_theory(Dh, N, H)
    uvec = sm.cos(tw) * (sm.cos(eta) * H.x + sm.sin(eta) * H.y) + sm.sin(tw) * H.z
    W = H.orientnew("W", "Axis", [q5, uvec])
    W.set_ang_vel(N, H.ang_vel_in(N) + u5 * uvec)
    Pw = Ph.locatenew("Pw", 0 * H.x)
    Pw.set_vel(N, Ph.vel(N))

    kd = [me.dot(Rf.ang_vel_in(N) - w_R_qd, uv) for uv in L] + [q4.diff() - u4, q5.diff() - u5]

    hoop = me.RigidBody("hoop", Dh, Rf, M, (me.inertia(L, M * r**2 / 2, M * r**2, M * r**2 / 2), Dh))
    house = me.RigidBody("house", Ph, H, mh, (me.inertia(H, Icx, Icy, Icz), Ph))
    Iwd = Iw * (uvec | uvec) + Iwt * (me.inertia(H, 1, 1, 1) - (uvec | uvec))
    wheel = me.RigidBody("wheel", Pw, W, mw, (Iwd, Pw))
    loads = [(Dh, -M * g * N.z), (Ph, -mh * g * N.z), (Pw, -mw * g * N.z),
             (H, -b * u4 * L.y - T * uvec + Th * L.y), (Rf, b * u4 * L.y - Th * L.y), (W, T * uvec)]
    KM = me.KanesMethod(N, q_ind=[q1, q2, q3, q4, q5], u_ind=[u1, u2, u3, u4, u5], kd_eqs=kd)
    KM.kanes_equations([hoop, house, wheel], loads)
    params = [r, M, d, g, eta, b, tw, mh, Icx, Icy, Icz, mw, Iw, Iwt]
    return KM, (q1, q2, q3, q4, q5), (u1, u2, u3, u4, u5), (T, Th), params


@functools.lru_cache(maxsize=1)
def _linearizer():
    KM, qs, us, T, params = _symbolic()
    lin = KM.to_linearizer()
    return lin, qs, us, T, params


def _linearize_full(v, rp):
    """Full linearisation: state (q1..q5, u1..u5), inputs (T wheel, T_h hub)."""
    lin, qs, us, Ts_, params = _linearizer()
    T, Th = Ts_
    m_h, (Icx, Icy, Icz) = rp.housing()
    Iwt = max(P.I_wy, 0.5 * P.I_wx * 1.0001)
    r_eff = rp.R + 0.01                                     # rim capsule radius
    vals = dict(zip(params, [r_eff, rp.M_hoop, rp.d, P.g0, rp.eta, rp.bearing_damping,
                             rp.wheel_tilt, m_h, Icx, Icy, Icz, P.m_w, P.I_wx, Iwt]))
    q1, q2, q3, q4, q5 = qs
    u1, u2, u3, u4, u5 = us
    op = {q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, u1: 0, u2: v / r_eff, u3: 0, u4: 0, u5: 0,
          T: 0, Th: 0}
    for q in qs:
        op[q.diff()] = 0
    op[u2.diff()] = 0
    for u in (u1, u3, u4, u5):
        op[u.diff()] = 0
    A, B = lin.linearize(op_point=[op, vals], A_and_B=True)
    A = np.array(A.subs(vals).evalf(), dtype=float)
    B = np.array(B.subs(vals).evalf(), dtype=float)
    order = list(lin.r)                  # input order chosen by SymPy
    B = B[:, [order.index(T), order.index(Th)]]
    return A, B


def linear_model_at_speed(v, rp: RollingParams = RollingParams()):
    """Continuous A (6x6), B (6x1) for x = (lean, lean_rate, yaw_rate, pitch, pitch_rate,
    wheel_speed) about straight upright rolling at speed v (m/s); wheel torque only."""
    A, B = _linearize_full(v, rp)
    # full state order: q1..q5, u1..u5
    keep = [1, 5, 7, 3, 8, 9]            # q2, u1, u3, q4, u4, u5
    return A[np.ix_(keep, keep)], B[keep][:, :1]


def linear_model_drive(v, rp: RollingParams = RollingParams()):
    """With the hub motor: A (7x7), B (7x2) for x = (lean, lean_rate, yaw_rate, pitch,
    pitch_rate, wheel_speed, hoop_spin), inputs (wheel torque, hub torque)."""
    A, B = _linearize_full(v, rp)
    keep = [1, 5, 7, 3, 8, 9, 6]         # ... + u2 (hoop spin = speed / r)
    return A[np.ix_(keep, keep)], B[keep]


# ---------------------------------------------------------------------------
from scipy.linalg import expm, solve_discrete_are
import mujoco

_VP = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
V_GRID = np.concatenate([-_VP[:0:-1], _VP])          # -5 ... 5 m/s
SCALE = np.array([np.deg2rad(1), np.deg2rad(10), np.deg2rad(10), np.deg2rad(5), np.deg2rad(20), 50.0])


def _c2d(A, B, Ts):
    n, m_ = B.shape
    Mx = np.zeros((n + m_, n + m_)); Mx[:n, :n] = A; Mx[:n, n:] = B
    E = expm(Mx * Ts)
    return E[:n, :n], E[:n, n:]


class ScheduledRollingController:
    """LQR gains scheduled on rolling speed (Kane linearisation at each speed), with
    one-step measurement-delay compensation.

    A rolling hoop has a conserved quantity the wheel cannot change: a combination
    c = w'x of yaw rate and lean (vertical angular momentum; at standstill simply
    the yaw rate). With c != 0 the only torque-free state is a steady lean-into-
    the-turn circle x_eq(c) -- like a coin rolling in circles -- not "straight and
    upright". The controller therefore regulates x - x_eq(c): forcing lean -> 0
    against the conservation law needs a constant torque and runs the wheel away.
    """

    def __init__(self, rp: RollingParams = RollingParams(), Ts=0.01,
                 Q=(1.0, 0.1, 0.01, 0.3, 0.1, 0.05), Rw=20.0, use_yaw=True,
                 v_fade=None, fade_pow=4):
        self.rp, self.Ts = rp, Ts
        S = np.diag(SCALE)
        self.K, self.Ad, self.Bd, self.W, self.Xeq = [], [], [], [], []
        Si = np.linalg.inv(S)
        for v in V_GRID:
            A, B = linear_model_at_speed(v, rp)
            w, xeq = self._conserved(A, B)
            Ad, Bd = _c2d(A, B, Ts)
            An, Bn = Si @ Ad @ S, Si @ Bd                        # scaled, discrete
            Rv = Rw * (1 + (abs(v) / v_fade) ** fade_pow) if v_fade else Rw
            if w is None:
                Pr = solve_discrete_are(An, Bn, np.diag(Q), np.array([[Rv]]))
                K = np.linalg.solve(Rv + Bn.T @ Pr @ Bn, Bn.T @ Pr @ An) @ Si
                w, xeq = np.zeros(6), np.zeros(6)
            else:
                # remove the conserved mode exactly: z = x - c*x_eq lies in ker(w'),
                # reduce to a 5-D system there and design the LQR on it
                wn, xn = S @ w, Si @ xeq                        # scaled versions, wn.xn = 1
                Pn = np.eye(6) - np.outer(xn, wn)
                _, _, Vt = np.linalg.svd(wn[None, :])
                V = Vt[1:].T                                    # orthonormal basis of ker(wn')
                Ar, Br = V.T @ Pn @ An @ V, V.T @ Pn @ Bn
                Qr = V.T @ np.diag(Q) @ V
                Pr = solve_discrete_are(Ar, Br, Qr, np.array([[Rv]]))
                Kr = np.linalg.solve(Rv + Br.T @ Pr @ Br, Br.T @ Pr @ Ar)
                K = Kr @ V.T @ Si                               # acts on (x - c*x_eq)
            self.K.append(K); self.Ad.append(Ad); self.Bd.append(Bd)
            self.W.append(w); self.Xeq.append(xeq)
        self.K, self.Ad, self.Bd, self.W, self.Xeq = map(np.array, (self.K, self.Ad, self.Bd, self.W, self.Xeq))
        self.Rmap = np.array([np.outer(xe, w) / (w @ xe) if np.any(w) else np.zeros((6, 6))
                              for w, xe in zip(self.W, self.Xeq)])
        self.ref_lim = np.array([np.deg2rad(6), np.inf, 1.0, np.inf, np.inf, np.inf])
        if not use_yaw:
            self.K[:, :, 2] = 0.0
        self.u = 0.0

    @staticmethod
    def _conserved(A, B, tol=1e-8):
        """left null vector w of [A B] (w'A = 0, w'B = 0) and the torque-free
        equilibrium direction x_eq (A x_eq = 0, wheel speed 0, w'x_eq = 1)."""
        M = np.hstack([A, B])
        U, sv, _ = np.linalg.svd(M)
        null = U[:, sv < tol * sv.max()] if (sv < tol * sv.max()).any() else None
        if null is None:
            # rank deficiency beyond the 6 singular values (M is 6x7): check the last
            if sv[-1] > tol * sv.max():
                return None, None
            null = U[:, -1:]
        w = null[:, 0]
        # fix the SVD's arbitrary sign so w (and x_eq) vary smoothly with speed:
        # at standstill w is the yaw-rate axis, so keep its yaw component positive
        if w[2] < 0:
            w = -w
        _, sa, Vt = np.linalg.svd(A)
        NA = Vt[sa < 1e-8 * sa.max()].T                 # null space of A (torque-free states)
        # choose the torque-free equilibrium with the same conserved value that
        # is nicest: small lean and turn rate first, wheel speed second (with a
        # tilted wheel the turning momentum can be parked in the wheel instead)
        Wt = np.diag([1e4, 1.0, 1e3, 1.0, 1.0, 1e-4])
        G = NA.T @ Wt @ NA
        a_ = np.linalg.solve(G + 1e-12 * np.eye(len(G)), NA.T @ w)
        xeq = NA @ a_
        xeq /= (w @ xeq)
        return w, xeq

    def _interp(self, arr, v):
        v = float(np.clip(v, V_GRID[0], V_GRID[-1]))
        i = int(np.clip(np.searchsorted(V_GRID, v) - 1, 0, len(V_GRID) - 2))
        wgt = (v - V_GRID[i]) / (V_GRID[i + 1] - V_GRID[i])
        return (1 - wgt) * arr[i] + wgt * arr[i + 1]

    def step(self, x_delayed, v):
        """x_delayed: (lean, lean_rate, yaw_rate, pitch, pitch_rate, wheel) from the
        previous step; v: rolling speed (m/s, signed)."""
        x = np.array(x_delayed, float)
        Ad, Bd, K = self._interp(self.Ad, v), self._interp(self.Bd, v), self._interp(self.K, v)
        xhat = Ad @ x + Bd[:, 0] * self.u
        xref = np.clip(self._interp(self.Rmap, v) @ xhat, -self.ref_lim, self.ref_lim)
        self.u = float(-(K @ (xhat - xref))[0])
        return self.u

    def applied(self, u):
        self.u = u


def measure6(m, d, r_eff):
    """(lean, lean_rate, yaw_rate, pitch_rel, pitch_rate_rel, wheel), rolling speed."""
    hb, hs = m.body("hoop").id, m.body("housing").id
    Rh = d.xmat[hb].reshape(3, 3); Rs = d.xmat[hs].reshape(3, 3)
    a = Rh[:, 1]
    zw = np.array([0.0, 0.0, 1.0])
    f = np.cross(a, zw); f /= np.linalg.norm(f)
    Lz = np.cross(f, a)
    lean = np.arcsin(np.clip(a @ zw, -1, 1))
    w = Rh @ d.qvel[3:6]                                 # hoop angular velocity (world)
    u1, u2, u3 = w @ f, w @ a, w @ Lz
    h = Rs[:, 2]
    pitch = np.arctan2(h @ f, h @ Lz)
    w_house = w + a * d.qvel[m.joint("axle").dofadr[0]]
    u4 = w_house @ a - u3 * np.tan(lean)
    return np.array([lean, u1, u3, pitch, u4, d.qvel[m.joint("phi").dofadr[0]]]), u2 * r_eff


def simulate_scheduled(rp=RollingParams(), T=12.0, lean0_deg=0.5, v0=0.0, pushes=(),
                       seed=0, Ts=0.01, dt=5e-4, angle_std=np.deg2rad(0.03),
                       angle_bias_std=np.deg2rad(0.1), rate_std=0.005, wheel_std=0.3,
                       ctrl=None, frames=None, frame_every=None, renderer=None, camera=None):
    from .rolling import load
    from .sim import Motor
    rng = np.random.default_rng(seed)
    m, d = load(rp, dt)
    q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([1.0, 0, 0]), np.deg2rad(lean0_deg))
    d.qpos[3:7] = q
    d.qpos[2] = (rp.R + 0.01) * np.cos(np.deg2rad(lean0_deg)) + 1e-4
    d.qvel[0] = v0
    d.qvel[4] = v0 / (rp.R + 0.01)
    d.qvel[m.joint("axle").dofadr[0]] = -v0 / (rp.R + 0.01)    # housing hangs still
    mujoco.mj_forward(m, d)
    ctrl = ctrl or ScheduledRollingController(rp, Ts)
    motor = Motor(P)
    bias = np.array([rng.normal(0, angle_bias_std), 0, 0, rng.normal(0, angle_bias_std), 0, 0])
    noise = np.array([angle_std, rate_std, rate_std, angle_std, rate_std, wheel_std])
    ids = {b: m.body(b).id for _, _, b, _, _ in pushes}
    r_eff = rp.R + 0.01

    def meas():
        x, v = measure6(m, d, r_eff)
        return x + bias + rng.normal(0, 1, 6) * noise, v + rng.normal(0, 0.01), x

    y, vm, _ = meas()
    log, fell = [], False
    for k in range(int(T / Ts)):
        u, lim = motor.limit(ctrl.step(y, vm), d.qvel[m.joint("phi").dofadr[0]], Ts)
        ctrl.applied(u)
        y, vm, x_true = meas()
        d.ctrl[0] = u
        for _ in range(int(round(Ts / dt))):
            d.xfrc_applied[:] = 0
            for t0, dur, bname, F, pt in pushes:
                if t0 <= d.time < t0 + dur:
                    bid = ids[bname]
                    d.xfrc_applied[bid, :3] += F
            mujoco.mj_step(m, d)
            if frames is not None and renderer is not None and int(round(d.time / dt)) % frame_every == 0:
                renderer.update_scene(d, camera=camera)
                frames.append(renderer.render())
        _, v_true = measure6(m, d, r_eff)
        log.append([d.time, *x_true, v_true, u, d.qpos[0], d.qpos[1]])
        if abs(x_true[0]) > np.deg2rad(30):
            fell = True
            break
    # columns: t, lean, lean_rate, yaw_rate, pitch, pitch_rate, wheel, v, u, x, y
    return np.array(log), fell


# ---------------------------------------------------------------------------
# Speed control: reaction wheel (lean) + hub motor (rolling speed)
V_DRIVE = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
# state: lean, lean_rate, yaw_rate, pitch, pitch_rate, wheel, hoop_spin, int(spin error)
SCALE_D = np.array([np.deg2rad(1), np.deg2rad(10), np.deg2rad(10), np.deg2rad(10), np.deg2rad(30),
                    50.0, 0.3, 0.3])


def _conserved_general(A, B, weights, tol=1e-8):
    """Left null vector w of [A B] and the 'nicest' torque-free equilibrium x_eq
    (A x_eq = 0, w'x_eq = 1, minimising x_eq' diag(weights) x_eq)."""
    U, sv, _ = np.linalg.svd(np.hstack([A, B]))
    n = A.shape[0]
    if len(sv) == n and sv[-1] > tol * sv.max():
        return None, None
    w = U[:, -1]
    if w[2] < 0:
        w = -w
    _, sa, Vt = np.linalg.svd(A)
    NA = Vt[sa < 1e-8 * sa.max()].T
    if NA.shape[1] == 0:
        return None, None
    G = NA.T @ np.diag(weights) @ NA
    xeq = NA @ np.linalg.solve(G + 1e-12 * np.eye(len(G)), NA.T @ w)
    return w, xeq / (w @ xeq)


class DriveController:
    """Two inputs (wheel torque, hub torque), gains scheduled on rolling speed.
    Tracks a rolling-speed command with integral action (rolling friction needs a
    steady housing tilt), balances sideways, handles the conserved yaw/lean mode
    exactly as ScheduledRollingController does, and compensates the 10 ms delay."""

    def __init__(self, rp: RollingParams = RollingParams(), Ts=0.01,
                 Q=(1.0, 0.1, 0.01, 0.1, 0.05, 0.05, 1.0, 0.3), R=(20.0, 20.0), accel=0.35,
                 v_min=-0.3, v_max=3.0, park_in_wheel=False):
        # accel: the housing pitches ~ accel to drive / brake; that pitch must stay
        # below the wheel tilt, else the wheel's vertical component flips sign.
        # v_min: backwards the tilted wheel acts like a negative tilt -> slow reverse only
        # park_in_wheel=False: speed-dependent handling of the conserved turning
        # momentum (wheel at speed, turning in place near standstill) -- 0/8 falls
        # on a long drive/brake/reverse/push sequence vs 4-7/8 when always parked
        self.rp, self.Ts, self.accel = rp, Ts, accel
        self.v_min, self.v_max = v_min, v_max
        self.r = rp.R + 0.01
        S = np.diag(SCALE_D); Si = np.linalg.inv(S)
        self.K, self.Ad, self.Bd, self.Rmap = [], [], [], []
        for v in V_DRIVE:
            A7, B7 = linear_model_drive(v, rp)
            A = np.zeros((8, 8)); A[:7, :7] = A7; A[7, 6] = 1.0     # integrator of spin error
            B = np.zeros((8, 2)); B[:7] = B7
            # where to put conserved turning momentum: at speed it must go into the
            # wheel (turning would mean leaning into a circle); near standstill the
            # hoop can just turn in place, which keeps the wheel unloaded
            fast = 1.0 if park_in_wheel else min(abs(v) / 1.5, 1.0)
            w_yaw = 10 ** (-2 + 5 * fast)          # 1e-2 at rest -> 1e3 at >= 1.5 m/s
            w_wheel = 10 ** (0 - 4 * fast)         # 1 at rest   -> 1e-4 at >= 1.5 m/s
            w, xeq = _conserved_general(A, B, weights=[1e4, 1, w_yaw, 1, 1, w_wheel, 1e4, 1e4])
            Ad, Bd = _c2d(A, B, Ts)
            An, Bn = Si @ Ad @ S, Si @ Bd
            Rm = np.diag(R)
            if w is None:
                Pr = solve_discrete_are(An, Bn, np.diag(Q), Rm)
                K = np.linalg.solve(Rm + Bn.T @ Pr @ Bn, Bn.T @ Pr @ An) @ Si
                Rmap = np.zeros((8, 8))
            else:
                wn, xn = S @ w, Si @ xeq
                Pn = np.eye(8) - np.outer(xn, wn)
                V = np.linalg.svd(wn[None, :])[2][1:].T
                Ar, Br = V.T @ Pn @ An @ V, V.T @ Pn @ Bn
                Pr = solve_discrete_are(Ar, Br, V.T @ np.diag(Q) @ V, Rm)
                K = np.linalg.solve(Rm + Br.T @ Pr @ Br, Br.T @ Pr @ Ar) @ V.T @ Si
                Rmap = np.outer(xeq, w)
            self.K.append(K); self.Ad.append(Ad[:7, :7]); self.Bd.append(Bd[:7]); self.Rmap.append(Rmap)
        self.K, self.Ad, self.Bd, self.Rmap = map(np.array, (self.K, self.Ad, self.Bd, self.Rmap))
        self.ref_lim = np.array([np.deg2rad(6), np.inf, 1.5, np.inf, np.inf, 300.0, np.inf, np.inf])
        self.reset()

    def reset(self, v=0.0):
        self.u = np.zeros(2)
        self.integ = 0.0
        self.v_cmd = v                 # rate-limited speed command actually tracked
        self.v_target = v

    def _interp(self, arr, v):
        v = float(np.clip(v, V_DRIVE[0], V_DRIVE[-1]))
        i = int(np.clip(np.searchsorted(V_DRIVE, v) - 1, 0, len(V_DRIVE) - 2))
        a = (v - V_DRIVE[i]) / (V_DRIVE[i + 1] - V_DRIVE[i])
        return (1 - a) * arr[i] + a * arr[i + 1]

    def step(self, x_delayed, v_meas):
        """x_delayed: 7-state measurement from the previous step (spin = v/r)."""
        tgt = float(np.clip(self.v_target, self.v_min, self.v_max))
        self.v_cmd += float(np.clip(tgt - self.v_cmd, -self.accel * self.Ts, self.accel * self.Ts))
        x = np.asarray(x_delayed, float)
        Ad, Bd, K, Rm = (self._interp(a, v_meas) for a in (self.Ad, self.Bd, self.K, self.Rmap))
        xhat = Ad @ x + Bd @ self.u
        spin_ref = self.v_cmd / self.r
        self.integ = float(np.clip(self.integ + (xhat[6] - spin_ref) * self.Ts, -2.0, 2.0))
        xa = np.concatenate([xhat, [self.integ]])
        ref = np.clip(Rm @ xa, -self.ref_lim, self.ref_lim)
        ref[6] += spin_ref
        self.u = -(K @ (xa - ref))
        return self.u.copy()

    def applied(self, u):
        self.u = np.asarray(u, float)


def simulate_drive(rp=RollingParams(), T=20.0, v_target=lambda t: 0.0, lean0_deg=1.0, v0=0.0,
                   pushes=(), seed=0, Ts=0.01, dt=5e-4, ctrl=None):
    """Closed loop with speed commands. Returns log columns:
    t, lean, lean_rate, yaw_rate, pitch, pitch_rate, wheel, v, v_cmd, u_wheel, u_hub, x, y."""
    from .rolling import load
    from .sim import Motor
    rng = np.random.default_rng(seed)
    m, d = load(rp, dt)
    q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([1.0, 0, 0]), np.deg2rad(lean0_deg))
    d.qpos[3:7] = q
    r = rp.R + 0.01
    d.qpos[2] = r * np.cos(np.deg2rad(lean0_deg)) + 1e-4
    d.qvel[0] = v0; d.qvel[4] = v0 / r; d.qvel[m.joint("axle").dofadr[0]] = -v0 / r
    mujoco.mj_forward(m, d)
    ctrl = ctrl or DriveController(rp, Ts)
    ctrl.reset(v0)
    motor = Motor(P)
    bias = np.array([rng.normal(0, np.deg2rad(0.1)), 0, 0, rng.normal(0, np.deg2rad(0.1)), 0, 0, 0])
    noise = np.array([np.deg2rad(0.03), 0.005, 0.005, np.deg2rad(0.03), 0.005, 0.3, 0.01])
    ids = {b: m.body(b).id for _, _, b, _, _ in pushes}

    def meas():
        x6, v = measure6(m, d, r)
        x = np.append(x6, v / r)
        return x + bias + rng.normal(0, 1, 7) * noise, v, x

    y, vm, _ = meas()
    log, fell = [], False
    for k in range(int(T / Ts)):
        ctrl.v_target = v_target(d.time)
        u = ctrl.step(y, vm)
        uw, _ = motor.limit(float(u[0]), d.qvel[m.joint("phi").dofadr[0]], Ts)
        uh = float(np.clip(u[1], -rp.hub_tau, rp.hub_tau))
        ctrl.applied([uw, uh])
        y, vm, x_true = meas()
        d.ctrl[0], d.ctrl[1] = uw, uh
        for _ in range(int(round(Ts / dt))):
            d.xfrc_applied[:] = 0
            for t0, dur, bname, F, pt in pushes:
                if t0 <= d.time < t0 + dur:
                    d.xfrc_applied[ids[bname], :3] += F
            mujoco.mj_step(m, d)
        log.append([d.time, *x_true[:6], x_true[6] * r, ctrl.v_cmd, uw, uh, d.qpos[0], d.qpos[1]])
        if abs(x_true[0]) > np.deg2rad(30):
            fell = True
            break
    return np.array(log), fell
