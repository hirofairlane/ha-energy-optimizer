"""Tests for eo.forecasters.atmospheric_factor.

Splits into:
  * Pure featurisation (no ML involved).
  * Model contract (fit/predict/save/load shape and error handling).
  * Synthetic learning (validates the pipeline actually fits a known signal —
    not a real-data accuracy test).
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from eo.forecasters.atmospheric_factor import (
    AtmosphericFactorFeatures,
    AtmosphericFactorModel,
    DEFAULT_QUANTILES,
    FEATURE_NAMES,
    WEATHER_CONDITION_CODES,
    make_features_for_hour,
    weather_to_int,
)


# ── Weather encoding ────────────────────────────────────────────────────────
class TestWeatherToInt:
    def test_known_conditions_round_trip(self):
        for cond, code in WEATHER_CONDITION_CODES.items():
            assert weather_to_int(cond) == code

    def test_unknown_string_falls_back_to_unknown(self):
        assert weather_to_int("blizzard") == WEATHER_CONDITION_CODES["unknown"]

    def test_none_returns_unknown_code(self):
        assert weather_to_int(None) == WEATHER_CONDITION_CODES["unknown"]

    def test_empty_string_returns_unknown_code(self):
        assert weather_to_int("") == WEATHER_CONDITION_CODES["unknown"]

    def test_case_insensitive(self):
        assert weather_to_int("SUNNY") == weather_to_int("sunny")
        assert weather_to_int("Partlycloudy") == weather_to_int("partlycloudy")

    def test_sunny_is_lower_code_than_cloudy(self):
        assert weather_to_int("sunny") < weather_to_int("cloudy") < weather_to_int("rainy")


# ── Featurisation ───────────────────────────────────────────────────────────
class TestMakeFeatures:
    def test_returns_dataclass(self):
        f = make_features_for_hour(
            ts=datetime(2026, 5, 20, 12),
            weather_condition="sunny",
            weather_condition_lag1d="cloudy",
            t_outdoor=22.5,
            humidity=45.0,
            yield_factor_similar_days=[0.85, 0.90, 0.80],
        )
        assert isinstance(f, AtmosphericFactorFeatures)
        assert f.hour == 12
        assert f.month == 5
        assert f.day_of_year == 140  # 2026-05-20 is day 140
        assert f.weather_condition_int == WEATHER_CONDITION_CODES["sunny"]
        assert f.weather_condition_lag1d_int == WEATHER_CONDITION_CODES["cloudy"]
        assert f.t_outdoor == 22.5
        assert f.humidity == 45.0

    def test_yield_history_mean_and_std(self):
        f = make_features_for_hour(
            ts=datetime(2026, 5, 20, 12),
            weather_condition="sunny",
            weather_condition_lag1d="sunny",
            t_outdoor=20.0, humidity=40.0,
            yield_factor_similar_days=[0.8, 0.9, 1.0],
        )
        assert f.yield_factor_similar_days_mean == pytest.approx(0.9, rel=1e-3)
        assert f.yield_factor_similar_days_std > 0

    def test_empty_history_uses_uninformative_prior(self):
        f = make_features_for_hour(
            ts=datetime(2026, 5, 20, 12),
            weather_condition="sunny",
            weather_condition_lag1d="sunny",
            t_outdoor=20.0, humidity=40.0,
            yield_factor_similar_days=[],
        )
        assert f.yield_factor_similar_days_mean == 0.5
        assert f.yield_factor_similar_days_std == 0.3

    def test_none_history_uses_uninformative_prior(self):
        f = make_features_for_hour(
            ts=datetime(2026, 5, 20, 12),
            weather_condition=None,
            weather_condition_lag1d=None,
            t_outdoor=20.0, humidity=40.0,
            yield_factor_similar_days=None,
        )
        assert f.yield_factor_similar_days_mean == 0.5
        assert f.yield_factor_similar_days_std == 0.3

    def test_single_sample_history_uses_default_std(self):
        f = make_features_for_hour(
            ts=datetime(2026, 5, 20, 12),
            weather_condition="sunny",
            weather_condition_lag1d="sunny",
            t_outdoor=20.0, humidity=40.0,
            yield_factor_similar_days=[0.7],
        )
        assert f.yield_factor_similar_days_mean == 0.7
        assert f.yield_factor_similar_days_std == 0.3

    def test_to_array_shape_matches_feature_names(self):
        f = make_features_for_hour(
            ts=datetime(2026, 5, 20, 12),
            weather_condition="sunny",
            weather_condition_lag1d="cloudy",
            t_outdoor=22.5, humidity=45.0,
            yield_factor_similar_days=[0.85],
        )
        arr = f.to_array()
        assert arr.shape == (len(FEATURE_NAMES),)
        assert arr.dtype == float


# ── Model contract ──────────────────────────────────────────────────────────
def _synthetic_dataset(n=400, seed=0):
    """Generate a synthetic dataset with a known underlying pattern:
        factor = clamp(0.9 - 0.15 * cloud_code + noise, 0, 1.1)
    plus weak dependence on humidity and yield prior mean.
    """
    rng = np.random.default_rng(seed)
    X = []
    y = []
    for _ in range(n):
        hour = rng.integers(6, 19)
        weekday = rng.integers(0, 7)
        month = rng.integers(1, 13)
        doy = rng.integers(1, 366)
        wc = rng.integers(0, 7)
        wc_lag = rng.integers(0, 7)
        t_out = rng.uniform(-5, 35)
        hum = rng.uniform(20, 90)
        prior_mean = rng.uniform(0.2, 0.95)
        prior_std = rng.uniform(0.05, 0.3)
        # Underlying truth: clearer sky → higher factor.
        truth = 0.9 - 0.13 * wc + 0.001 * (hum - 50) * (-1)  # humid → lower
        truth += 0.5 * (prior_mean - 0.5)  # follow prior weakly
        truth += rng.normal(0, 0.06)
        truth = float(np.clip(truth, 0, 1.1))
        X.append([hour, weekday, month, doy, wc, wc_lag, t_out, hum, prior_mean, prior_std])
        y.append(truth)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


class TestModelContract:
    def test_constructor_rejects_invalid_quantiles(self):
        with pytest.raises(ValueError):
            AtmosphericFactorModel(quantiles=())
        with pytest.raises(ValueError):
            AtmosphericFactorModel(quantiles=(0.0, 0.5))
        with pytest.raises(ValueError):
            AtmosphericFactorModel(quantiles=(0.5, 1.5))

    def test_quantiles_get_sorted(self):
        m = AtmosphericFactorModel(quantiles=(0.9, 0.1, 0.5))
        assert m.quantiles == (0.1, 0.5, 0.9)

    def test_quantile_names(self):
        m = AtmosphericFactorModel(quantiles=(0.1, 0.5, 0.9))
        assert m.quantile_names == ["p10", "p50", "p90"]

    def test_fitted_property(self):
        m = AtmosphericFactorModel()
        assert m.fitted is False
        X, y = _synthetic_dataset(n=50)
        m.fit(X, y)
        assert m.fitted is True

    def test_predict_before_fit_raises(self):
        m = AtmosphericFactorModel()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((1, len(FEATURE_NAMES))))

    def test_fit_rejects_wrong_feature_count(self):
        m = AtmosphericFactorModel()
        with pytest.raises(ValueError):
            m.fit(np.zeros((10, 3)), np.zeros(10))

    def test_fit_rejects_1d_X(self):
        m = AtmosphericFactorModel()
        with pytest.raises(ValueError):
            m.fit(np.zeros(len(FEATURE_NAMES)), np.zeros(1))

    def test_fit_rejects_mismatched_y_length(self):
        m = AtmosphericFactorModel()
        with pytest.raises(ValueError):
            m.fit(np.zeros((10, len(FEATURE_NAMES))), np.zeros(11))

    def test_predict_shape(self):
        m = AtmosphericFactorModel(quantiles=(0.1, 0.5, 0.9))
        X, y = _synthetic_dataset(n=100)
        m.fit(X, y)
        out = m.predict(X[:5])
        assert set(out.keys()) == {"p10", "p50", "p90"}
        for v in out.values():
            assert v.shape == (5,)

    def test_predict_clipped_to_physical_range(self):
        m = AtmosphericFactorModel()
        X, y = _synthetic_dataset(n=100)
        # Saturate target with very high values to push prediction high
        y_high = np.full_like(y, 2.0)
        m.fit(X, y_high)
        out = m.predict(X)
        for v in out.values():
            assert v.min() >= 0.0
            assert v.max() <= 1.1

    def test_predict_quantile_ordering_enforced(self):
        m = AtmosphericFactorModel(quantiles=(0.1, 0.5, 0.9))
        X, y = _synthetic_dataset(n=200)
        m.fit(X, y)
        out = m.predict(X)
        # Element-wise: p10 ≤ p50 ≤ p90
        assert np.all(out["p10"] <= out["p50"] + 1e-9)
        assert np.all(out["p50"] <= out["p90"] + 1e-9)

    def test_predict_one_returns_scalars(self):
        m = AtmosphericFactorModel()
        X, y = _synthetic_dataset(n=100)
        m.fit(X, y)
        feat = make_features_for_hour(
            ts=datetime(2026, 5, 20, 12),
            weather_condition="sunny",
            weather_condition_lag1d="sunny",
            t_outdoor=20.0, humidity=40.0,
            yield_factor_similar_days=[0.85],
        )
        out = m.predict_one(feat)
        assert set(out.keys()) == {"p10", "p50", "p90"}
        for v in out.values():
            assert isinstance(v, float)


# ── Synthetic learning sanity ──────────────────────────────────────────────
class TestSyntheticLearning:
    """Validate the pipeline actually learns a signal. NOT a real-data test."""

    def test_learns_negative_cloud_effect(self):
        # Generate a clear monotone pattern: cloudier weather → lower factor.
        m = AtmosphericFactorModel()
        X, y = _synthetic_dataset(n=600, seed=1)
        m.fit(X, y)
        # Two test inputs differing only in weather code.
        sunny = np.array([[12, 1, 6, 172, 1, 1, 22, 40, 0.85, 0.1]])
        cloudy = sunny.copy()
        cloudy[0, 4] = 5  # rainy
        pred_sunny = m.predict(sunny)["p50"][0]
        pred_cloudy = m.predict(cloudy)["p50"][0]
        assert pred_sunny > pred_cloudy, (
            f"expected sunny factor > cloudy factor, got {pred_sunny:.3f} vs {pred_cloudy:.3f}"
        )

    def test_p90_above_p10(self):
        m = AtmosphericFactorModel()
        X, y = _synthetic_dataset(n=400, seed=2)
        m.fit(X, y)
        pred = m.predict(X[:50])
        # On average p90 must be > p10 by a clear margin.
        assert (pred["p90"] - pred["p10"]).mean() > 0.05


# ── Persistence ────────────────────────────────────────────────────────────
class TestPersistence:
    def test_round_trip(self, tmp_path: Path):
        m = AtmosphericFactorModel()
        X, y = _synthetic_dataset(n=200, seed=3)
        m.fit(X, y)
        path = tmp_path / "atmf.joblib"
        m.save(path)
        m2 = AtmosphericFactorModel.load(path)
        # Predictions match exactly.
        np.testing.assert_array_equal(m.predict(X[:20])["p50"], m2.predict(X[:20])["p50"])

    def test_load_rejects_schema_mismatch(self, tmp_path: Path):
        m = AtmosphericFactorModel()
        X, y = _synthetic_dataset(n=100, seed=4)
        m.fit(X, y)
        path = tmp_path / "atmf.joblib"
        m.save(path)

        # Tamper with the saved feature_names to simulate a schema change.
        data = joblib.load(path)  # type: ignore[name-defined]
        data["feature_names"] = ["x", "y"]
        joblib.dump(data, path)  # type: ignore[name-defined]

        with pytest.raises(RuntimeError, match="schema"):
            AtmosphericFactorModel.load(path)


# Late import to keep the schema-mismatch test self-contained.
import joblib  # noqa: E402
