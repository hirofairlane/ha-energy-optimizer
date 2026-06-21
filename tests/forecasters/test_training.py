"""Tests for eo.forecasters.training."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from eo.forecasters.atmospheric_factor import (
    AtmosphericFactorModel,
    make_features_for_hour,
)
from eo.forecasters.house_forecaster import (
    HouseForecaster,
    make_house_features_for_hour,
)
from eo.forecasters.training import (
    MIN_TRAIN_SAMPLES,
    TrainingReport,
    compute_atmospheric_factor,
    train_atmospheric_factor_model,
    train_house_forecaster,
)


def _utc(h):
    return datetime(2026, 5, 20, 0, tzinfo=timezone.utc) + timedelta(hours=h)


# ── compute_atmospheric_factor ─────────────────────────────────────────────
class TestComputeAtmosphericFactor:
    def test_normal_ratio(self):
        assert compute_atmospheric_factor(3.0, 5.0) == pytest.approx(0.6)

    def test_clipped_to_zero_at_night(self):
        assert compute_atmospheric_factor(0.0, 0.0) == 0.0
        assert compute_atmospheric_factor(1.0, 0.0) == 0.0

    def test_clipped_to_max_1_1(self):
        # broken-cloud enhancement: actual > clear-sky
        assert compute_atmospheric_factor(8.0, 5.0) == 1.1

    def test_clipped_to_min_zero_on_negative_input(self):
        assert compute_atmospheric_factor(-1.0, 5.0) == 0.0


# ── Atmospheric factor training ────────────────────────────────────────────
def _make_atm_sample(h: int, sunny: bool):
    """Build a synthetic (ts, features, target) triple."""
    feat = make_features_for_hour(
        ts=_utc(h),
        weather_condition="sunny" if sunny else "rainy",
        weather_condition_lag1d="sunny" if sunny else "rainy",
        t_outdoor=22.0 if sunny else 14.0,
        humidity=40.0 if sunny else 80.0,
        yield_factor_similar_days=[0.85] if sunny else [0.3],
    )
    factor = 0.9 if sunny else 0.3
    return (_utc(h), feat, factor)


class TestTrainAtmospheric:
    def test_basic_training_returns_fitted_model_and_report(self):
        samples = [_make_atm_sample(h, sunny=(h % 2 == 0)) for h in range(80)]
        model, report = train_atmospheric_factor_model(samples)
        assert isinstance(model, AtmosphericFactorModel)
        assert model.fitted
        assert isinstance(report, TrainingReport)
        assert report.model_kind == "atmospheric_factor"
        assert report.samples_used == 80
        assert report.samples_dropped == 0
        assert math.isfinite(report.in_sample_mae_p50)

    def test_rejects_too_few_samples(self):
        samples = [_make_atm_sample(h, sunny=True) for h in range(10)]
        with pytest.raises(ValueError, match="Not enough"):
            train_atmospheric_factor_model(samples)

    def test_custom_min_samples_threshold(self):
        samples = [_make_atm_sample(h, sunny=True) for h in range(60)]
        # Lower the bar — should accept.
        model, report = train_atmospheric_factor_model(samples, min_samples=20)
        assert model.fitted
        assert report.samples_used == 60

    def test_inf_and_nan_samples_dropped(self):
        good = [_make_atm_sample(h, sunny=(h % 2 == 0)) for h in range(60)]
        # Inject a corrupt sample at index 0
        good_array_feat = good[0][1].to_array()

        class _BadFeat:
            def to_array(self):
                arr = good_array_feat.copy()
                arr[3] = float("nan")
                return arr
        good[0] = (_utc(0), _BadFeat(), 0.5)

        # Inject a NaN target too.
        class _OkFeat:
            def to_array(self):
                return good_array_feat.copy()
        good[1] = (_utc(1), _OkFeat(), float("nan"))

        model, report = train_atmospheric_factor_model(good, min_samples=10)
        assert report.samples_dropped == 2
        assert report.samples_used == len(good) - 2

    def test_p50_prediction_close_to_actual_for_clear_signal(self):
        # Strong, clean signal: 0.9 on sunny days, 0.3 on rainy.
        samples = []
        for h in range(200):
            sunny = (h % 3 != 0)
            samples.append(_make_atm_sample(h, sunny=sunny))
        model, report = train_atmospheric_factor_model(samples)
        # In-sample MAE on a clean synthetic signal should be small.
        assert report.in_sample_mae_p50 < 0.15


# ── House forecaster training ──────────────────────────────────────────────
def _make_house_sample(h: int, evening: bool):
    feat = make_house_features_for_hour(
        target_ts=_utc(h),
        t_outdoor=22.0,
        t_outdoor_lag24h=20.0,
        house_kwh_yesterday_same_hour=1.5 if evening else 0.6,
        house_kwh_last_observed=1.6 if evening else 0.7,
        custom_loads_planned_watts=2000.0 if evening else 0.0,
        deferred_loads_planned_watts=0.0,
    )
    target = 2.2 if evening else 0.7
    return (_utc(h), feat, target)


class TestTrainHouse:
    def test_basic_training(self):
        samples = [_make_house_sample(h, evening=(h % 4 == 0)) for h in range(80)]
        model, report = train_house_forecaster(samples)
        assert isinstance(model, HouseForecaster)
        assert model.fitted
        assert report.model_kind == "house"
        assert report.samples_used == 80
        assert math.isfinite(report.in_sample_mae_p50)

    def test_rejects_too_few_samples(self):
        samples = [_make_house_sample(h, evening=True) for h in range(20)]
        with pytest.raises(ValueError, match="Not enough"):
            train_house_forecaster(samples)

    def test_custom_kwh_cap(self):
        samples = [_make_house_sample(h, evening=True) for h in range(80)]
        model, _ = train_house_forecaster(samples, kwh_cap_per_hour=10.0)
        assert model.kwh_cap_per_hour == 10.0

    def test_custom_min_samples_threshold(self):
        samples = [_make_house_sample(h, evening=(h % 2 == 0)) for h in range(30)]
        model, report = train_house_forecaster(samples, min_samples=20)
        assert model.fitted
        assert report.samples_used == 30


# ── TrainingReport serialisation ────────────────────────────────────────────
class TestReportSerialisation:
    def test_to_dict(self):
        report = TrainingReport(
            model_kind="atmospheric_factor",
            samples_used=100, samples_dropped=2,
            in_sample_mae_p50=0.123456789,
            in_sample_bias_p50=-0.987654321,
        )
        d = report.to_dict()
        assert d["model_kind"] == "atmospheric_factor"
        assert d["samples_used"] == 100
        assert d["samples_dropped"] == 2
        # Rounded to 5 places
        assert d["in_sample_mae_p50"] == 0.12346
        assert d["in_sample_bias_p50"] == -0.98765


# ── Constants ──────────────────────────────────────────────────────────────
class TestConstants:
    def test_min_train_samples_sane(self):
        assert 20 < MIN_TRAIN_SAMPLES < 500
