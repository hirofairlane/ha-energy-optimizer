"""Tests for eo.forecasters.house_forecaster."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pytest

from eo.forecasters.house_forecaster import (
    DEFAULT_HOUSE_QUANTILES,
    HOUSE_FEATURE_NAMES,
    HOUSE_KWH_PER_HOUR_CAP,
    HouseFeatures,
    HouseForecaster,
    make_house_features_for_hour,
)


# ── Featurisation ───────────────────────────────────────────────────────────
class TestMakeFeatures:
    def test_saturday_is_weekend(self):
        f = make_house_features_for_hour(
            target_ts=datetime(2026, 5, 23, 12),  # 2026-05-23 = Saturday
            t_outdoor=20.0, t_outdoor_lag24h=18.0,
            house_kwh_yesterday_same_hour=1.2,
            house_kwh_last_observed=1.5,
        )
        assert f.is_weekend == 1

    def test_monday_is_not_weekend(self):
        f = make_house_features_for_hour(
            target_ts=datetime(2026, 5, 25, 12),
            t_outdoor=20.0, t_outdoor_lag24h=18.0,
            house_kwh_yesterday_same_hour=1.2,
            house_kwh_last_observed=1.5,
        )
        assert f.is_weekend == 0

    def test_to_array_shape_matches_feature_names(self):
        f = make_house_features_for_hour(
            target_ts=datetime(2026, 5, 20, 14),
            t_outdoor=22.0, t_outdoor_lag24h=20.0,
            house_kwh_yesterday_same_hour=0.8,
            house_kwh_last_observed=1.1,
            custom_loads_planned_watts=2000.0,
            deferred_loads_planned_watts=500.0,
        )
        arr = f.to_array()
        assert arr.shape == (len(HOUSE_FEATURE_NAMES),)
        assert arr.dtype == float

    def test_defaults_for_loads(self):
        f = make_house_features_for_hour(
            target_ts=datetime(2026, 5, 20, 14),
            t_outdoor=22.0, t_outdoor_lag24h=20.0,
            house_kwh_yesterday_same_hour=0.8,
            house_kwh_last_observed=1.1,
        )
        assert f.custom_loads_planned_watts == 0.0
        assert f.deferred_loads_planned_watts == 0.0


# ── Synthetic dataset ──────────────────────────────────────────────────────
def _synthetic_house_dataset(n=500, seed=0):
    """Synthetic data with a known pattern:
        kwh = base
            + 0.0005 * deferred_watts                # planned loads consume
            + 0.0005 * custom_watts
            + 0.02 * abs(t_outdoor - 20)             # HVAC pull at extremes
            + 0.4 * is_evening                       # peak-hour usage
            + noise
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        hour = rng.integers(0, 24)
        weekday = rng.integers(0, 7)
        month = rng.integers(1, 13)
        is_weekend = 1 if weekday >= 5 else 0
        t_out = rng.uniform(-5, 35)
        t_out_lag = rng.uniform(-5, 35)
        ksh = rng.uniform(0.4, 1.5)  # yesterday same hour
        klo = rng.uniform(0.5, 2.0)  # last observed
        cust = rng.uniform(0, 3000)
        defr = rng.uniform(0, 2500)
        is_evening = 1 if 18 <= hour <= 22 else 0
        truth = (
            0.7
            + 0.0005 * (cust + defr)
            + 0.02 * abs(t_out - 20)
            + 0.4 * is_evening
            + 0.3 * ksh
            + rng.normal(0, 0.15)
        )
        truth = float(max(0.0, truth))
        X.append([hour, weekday, month, is_weekend, t_out, t_out_lag, ksh, klo, cust, defr])
        y.append(truth)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


# ── Model contract ──────────────────────────────────────────────────────────
class TestModelContract:
    def test_constructor_rejects_invalid_quantiles(self):
        with pytest.raises(ValueError):
            HouseForecaster(quantiles=())
        with pytest.raises(ValueError):
            HouseForecaster(quantiles=(0.0, 0.5))

    def test_constructor_rejects_non_positive_cap(self):
        with pytest.raises(ValueError):
            HouseForecaster(kwh_cap_per_hour=0)
        with pytest.raises(ValueError):
            HouseForecaster(kwh_cap_per_hour=-1)

    def test_quantiles_sorted(self):
        m = HouseForecaster(quantiles=(0.9, 0.1, 0.5))
        assert m.quantiles == (0.1, 0.5, 0.9)
        assert m.quantile_names == ["p10", "p50", "p90"]

    def test_fitted_false_then_true(self):
        m = HouseForecaster()
        assert m.fitted is False
        X, y = _synthetic_house_dataset(n=50)
        m.fit(X, y)
        assert m.fitted is True

    def test_predict_before_fit_raises(self):
        m = HouseForecaster()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((1, len(HOUSE_FEATURE_NAMES))))

    def test_fit_rejects_wrong_feature_count(self):
        m = HouseForecaster()
        with pytest.raises(ValueError):
            m.fit(np.zeros((10, 4)), np.zeros(10))

    def test_fit_rejects_mismatched_y_length(self):
        m = HouseForecaster()
        with pytest.raises(ValueError):
            m.fit(np.zeros((10, len(HOUSE_FEATURE_NAMES))), np.zeros(11))

    def test_predict_shape(self):
        m = HouseForecaster()
        X, y = _synthetic_house_dataset(n=100)
        m.fit(X, y)
        out = m.predict(X[:7])
        assert set(out.keys()) == {"p10", "p50", "p90"}
        for v in out.values():
            assert v.shape == (7,)

    def test_predict_quantile_ordering(self):
        m = HouseForecaster()
        X, y = _synthetic_house_dataset(n=200)
        m.fit(X, y)
        out = m.predict(X)
        assert np.all(out["p10"] <= out["p50"] + 1e-9)
        assert np.all(out["p50"] <= out["p90"] + 1e-9)

    def test_predict_non_negative(self):
        m = HouseForecaster()
        X, y = _synthetic_house_dataset(n=200)
        m.fit(X, y)
        out = m.predict(X)
        for v in out.values():
            assert v.min() >= 0.0

    def test_predict_clipped_to_cap(self):
        m = HouseForecaster(kwh_cap_per_hour=5.0)
        X, y = _synthetic_house_dataset(n=100)
        # Push target up so the model wants to predict high.
        y_high = np.full_like(y, 50.0)
        m.fit(X, y_high)
        out = m.predict(X)
        for v in out.values():
            assert v.max() <= 5.0

    def test_predict_one_returns_scalars(self):
        m = HouseForecaster()
        X, y = _synthetic_house_dataset(n=100)
        m.fit(X, y)
        feat = make_house_features_for_hour(
            target_ts=datetime(2026, 5, 20, 19),  # evening
            t_outdoor=22.0, t_outdoor_lag24h=20.0,
            house_kwh_yesterday_same_hour=1.5,
            house_kwh_last_observed=1.8,
            custom_loads_planned_watts=2000.0,
        )
        out = m.predict_one(feat)
        assert set(out.keys()) == {"p10", "p50", "p90"}
        for v in out.values():
            assert isinstance(v, float)


# ── Synthetic learning ─────────────────────────────────────────────────────
class TestSyntheticLearning:
    def test_evening_higher_than_night(self):
        m = HouseForecaster()
        X, y = _synthetic_house_dataset(n=600, seed=1)
        m.fit(X, y)
        evening = np.array([[20, 1, 6, 0, 22, 20, 1.0, 1.5, 0, 0]])
        night = np.array([[3, 1, 6, 0, 22, 20, 1.0, 1.5, 0, 0]])
        ev_p50 = m.predict(evening)["p50"][0]
        n_p50 = m.predict(night)["p50"][0]
        assert ev_p50 > n_p50, (
            f"expected evening > night, got {ev_p50:.3f} vs {n_p50:.3f}"
        )

    def test_more_planned_loads_means_higher_prediction(self):
        m = HouseForecaster()
        X, y = _synthetic_house_dataset(n=600, seed=2)
        m.fit(X, y)
        base = np.array([[14, 1, 6, 0, 22, 20, 1.0, 1.5, 0, 0]])
        loaded = base.copy()
        loaded[0, 8] = 3000  # custom_loads_planned_watts
        loaded[0, 9] = 2000  # deferred_loads_planned_watts
        assert m.predict(loaded)["p50"][0] > m.predict(base)["p50"][0]


# ── Persistence ────────────────────────────────────────────────────────────
class TestPersistence:
    def test_round_trip(self, tmp_path: Path):
        m = HouseForecaster()
        X, y = _synthetic_house_dataset(n=150, seed=3)
        m.fit(X, y)
        path = tmp_path / "house.joblib"
        m.save(path)
        m2 = HouseForecaster.load(path)
        np.testing.assert_array_equal(
            m.predict(X[:20])["p50"], m2.predict(X[:20])["p50"],
        )

    def test_load_rejects_schema_mismatch(self, tmp_path: Path):
        m = HouseForecaster()
        X, y = _synthetic_house_dataset(n=80, seed=4)
        m.fit(X, y)
        path = tmp_path / "house.joblib"
        m.save(path)

        data = joblib.load(path)
        data["feature_names"] = ["x", "y", "z"]
        joblib.dump(data, path)

        with pytest.raises(RuntimeError, match="schema"):
            HouseForecaster.load(path)

    def test_load_preserves_cap(self, tmp_path: Path):
        m = HouseForecaster(kwh_cap_per_hour=7.0)
        X, y = _synthetic_house_dataset(n=80, seed=5)
        m.fit(X, y)
        path = tmp_path / "house.joblib"
        m.save(path)
        m2 = HouseForecaster.load(path)
        assert m2.kwh_cap_per_hour == 7.0


# ── Module constants ───────────────────────────────────────────────────────
class TestConstants:
    def test_default_quantiles(self):
        assert DEFAULT_HOUSE_QUANTILES == (0.10, 0.50, 0.90)

    def test_cap_constant_sane(self):
        assert 10 < HOUSE_KWH_PER_HOUR_CAP < 100
