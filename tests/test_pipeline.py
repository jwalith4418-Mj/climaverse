"""Fast, deterministic tests. No real data required; the IMD reader is exercised
via a manufactured format-identical fixture (round-trip), so the parser is verified
even though the portal file is not bundled."""
import numpy as np
from climaverse.config import RAIN_GRID, TEMP_GRID, PILOT, subset_indices
from climaverse.data import load_variable
from climaverse.data.imd_grid import read_grd, write_grd
from climaverse.twin import TwinState, SyncLedger, synchronise, twin_health
from climaverse.validation import walk_forward, BiasCorrector
from climaverse.simulation import ScenarioEngine, Scenario


def test_official_grid_shapes():
    assert RAIN_GRID.shape == (129, 135)      # IMD 0.25 deg rainfall
    assert TEMP_GRID.shape == (31, 31)        # IMD 1.0 deg temperature


def test_imd_grd_roundtrip(tmp_path):
    arr = np.random.rand(5, *TEMP_GRID.shape).astype("float32")
    arr[0, 0, 0] = np.nan
    p = tmp_path / "tmax_test.grd"
    write_grd(p, arr, TEMP_GRID)
    back = read_grd(p, TEMP_GRID)
    assert back.shape == arr.shape
    assert np.isnan(back[0, 0, 0])
    np.testing.assert_allclose(back[1:], arr[1:], atol=1e-4)


def test_sync_chain_is_tamper_evident():
    obs = load_variable("rain", n_days=30)
    st = TwinState.empty(RAIN_GRID, PILOT, "rain"); led = SyncLedger()
    for d in range(30):
        synchronise(st, obs.data[d], d, led)
    assert led.verify()
    led._rows[5].mean_value += 1.0          # tamper
    assert not led.verify()


def test_model_beats_climatology_on_rain():
    obs = load_variable("rain", n_days=540)
    la, lo = subset_indices(RAIN_GRID, PILOT)
    bt = walk_forward(obs.data[:, la, lo], split=0.7, seq_len=7, epochs=4)
    assert bt.clim_rmse > 0
    assert bt.skill_vs_clim > 0.0           # genuine skill over climatology


def test_health_never_invents_skill():
    st = TwinState.empty(RAIN_GRID, PILOT, "rain")
    h = twin_health(st, days_since_obs=0, skill_score=None)
    assert h.skill == 0.0                   # honesty invariant


def test_recalibrator_identity_below_min_n():
    bc = BiasCorrector().fit(np.zeros(10), np.ones(10))
    assert not bc.active                    # too few pairs -> identity
    np.testing.assert_array_equal(bc.apply(np.array([1.0, 2.0])), np.array([1.0, 2.0]))


def test_whatif_directions():
    obs_t = load_variable("tmax", n_days=210); obs_r = load_variable("rain", n_days=210)
    la, lo = subset_indices(TEMP_GRID, PILOT); la2, lo2 = subset_indices(RAIN_GRID, PILOT)
    res = ScenarioEngine().run(obs_t.data[-1, la, lo], obs_r.data[-1, la2, lo2],
                               Scenario("deficit", temp_delta_c=2.0, rain_pct=-20))
    assert res.deltas["mean_tmax_c"] == 2.0
    assert res.deltas["mean_rain_mm"] < 0
