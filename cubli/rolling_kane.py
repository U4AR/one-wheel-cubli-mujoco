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
             (H, -b * u4 * L.y - T * uvec), (Rf, b * u4 * L.y), (W, T * uvec)]
    KM = me.KanesMethod(N, q_ind=[q1, q2, q3, q4, q5], u_ind=[u1, u2, u3, u4, u5], kd_eqs=kd)
    KM.kanes_equations([hoop, house, wheel], loads)
    params = [r, M, d, g, eta, b, tw, mh, Icx, Icy, Icz, mw, Iw, Iwt]
    return KM, (q1, q2, q3, q4, q5), (u1, u2, u3, u4, u5), T, params


@functools.lru_cache(maxsize=1)
def _linearizer():
    KM, qs, us, T, params = _symbolic()
    lin = KM.to_linearizer()
    return lin, qs, us, T, params


def linear_model_at_speed(v, rp: RollingParams = RollingParams()):
    """Continuous A (6x6), B (6x1) for x = (lean, lean_rate, yaw_rate, pitch, pitch_rate,
    wheel_speed) about straight upright rolling at speed v (m/s)."""
    lin, qs, us, T, params = _linearizer()
    m_h, (Icx, Icy, Icz) = rp.housing()
    Iwt = max(P.I_wy, 0.5 * P.I_wx * 1.0001)
    r_eff = rp.R + 0.01                                     # rim capsule radius
    vals = dict(zip(params, [r_eff, rp.M_hoop, rp.d, P.g0, rp.eta, rp.bearing_damping,
                             rp.wheel_tilt, m_h, Icx, Icy, Icz, P.m_w, P.I_wx, Iwt]))
    q1, q2, q3, q4, q5 = qs
    u1, u2, u3, u4, u5 = us
    op = {q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, u1: 0, u2: v / r_eff, u3: 0, u4: 0, u5: 0,
          T: 0}
    for q in qs:
        op[q.diff()] = 0
    op[u2.diff()] = 0
    for u in (u1, u3, u4, u5):
        op[u.diff()] = 0
    A, B = lin.linearize(op_point=[op, vals], A_and_B=True)
    A = np.array(A.subs(vals).evalf(), dtype=float)
    B = np.array(B.subs(vals).evalf(), dtype=float)
    # full state order: q1..q5, u1..u5
    keep = [1, 5, 7, 3, 8, 9]            # q2, u1, u3, q4, u4, u5
    return A[np.ix_(keep, keep)], B[keep]


# ---------------------------------------------------------------------------
from scipy.linalg import expm, solve_discrete_are
import mujoco

_VP = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
V_GRID = np.concatenate([-_VP[:0:-1], _VP])          # -5 ... 5 m/s
SCALE = np.array([np.deg2rad(1), np.deg2rad(10), np.deg2rad(10), np.deg2rad(5), np.deg2rad(20), 50.0])


def _c2d(A, B, Ts):
    n = A.shape[0]
    Mx = np.zeros((n + 1, n + 1)); Mx[:n, :n] = A; Mx[:n, n:] = B
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
