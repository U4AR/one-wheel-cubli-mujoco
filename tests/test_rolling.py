"""Rolling hoop: model checks and a short closed-loop run."""
import numpy as np

from cubli.rolling import RollingParams, linear_model
from cubli.rolling_kane import linear_model_at_speed, simulate_scheduled


def test_kane_matches_hand_model_at_rest():
    rp = RollingParams()
    ev_k = np.max(np.real(np.linalg.eigvals(linear_model_at_speed(0.0, rp)[0])))
    ev_h = np.max(np.real(np.linalg.eigvals(linear_model(rp)[0])))
    assert abs(ev_k - ev_h) / ev_h < 0.05          # lean instability ~5.1-5.2 /s


def test_self_stable_above_critical_speed():
    rp = RollingParams()
    assert np.max(np.real(np.linalg.eigvals(linear_model_at_speed(0.5, rp)[0]))) > 1.0
    # above ~1.25 m/s the lean instability becomes a (near-)neutral weave
    assert np.max(np.real(np.linalg.eigvals(linear_model_at_speed(2.0, rp)[0]))) < 0.05


def test_balances_standing_and_rolling():
    for v in (0.0, 1.0):
        L, fell = simulate_scheduled(RollingParams(), T=6.0, v0=v, lean0_deg=2.0)
        assert not fell, v


def test_hub_motor_tracks_speed():
    from cubli.rolling_kane import simulate_drive
    L, fell = simulate_drive(RollingParams(), T=12.0, v_target=lambda t: 1.0 if t > 1 else 0.0, lean0_deg=1.0)
    assert not fell
    assert abs(L[-100:, 7].mean() - 1.0) < 0.05


def test_single_motor_gyro_balances_and_tracks_speed():
    from cubli.rolling_kane import simulate_drive, gyro_design, gyro_controller
    rp = gyro_design()
    L, fell = simulate_drive(rp, T=14.0, v_target=lambda t: 0.5 if t > 1 else 0.0,
                             ctrl=gyro_controller(rp), lean0_deg=1.0)
    assert not fell
    assert abs(L[-100:, 7].mean() - 0.5) < 0.08
