"""Tests for eo.forecasters.solar_forecaster.

The forecaster is a pure composition layer, so the tests focus on:
  * Contract correctness (timezone awareness, horizon range, unfitted model).
  * Hard physical constraints (night → zero across all quantiles).
  * Quantile ordering preserved through the multiplication.
  * Feature provider integration (the seam where the addon plugs in).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from eo.forecasters.atmospheric_factor import (
    AtmosphericFactorModel,
    make_features_for_hour,
)
from eo.forecasters.clear_sky import ClearSkyConfig, ClearSkyModel
from eo.forecasters.solar_forecaster import (
    HourForecast,
    SolarForecaster,
)


GUADARRAMA_LAT = 40.65
GUADARRAMA_LON = -4.0


def _clear_sky(capacity_kwp=5.0) -> ClearSkyModel:
    return ClearSkyModel(ClearSkyConfig(
        latitude=GUADARRAMA_LAT,
        longitude=GUADARRAMA_LON,
        capacity_kwp=capacity_kwp,
    ))


def _fitted_atmospheric(seed=0) -> AtmosphericFactorModel:
    """Train an atmospheric factor model on synthetic but realistic data so
    that downstream tests can call predict_hourly without raising."""
    from tests.forecasters.test_atmospheric_factor import _synthetic_dataset
    m = AtmosphericFactorModel()
    X, y = _synthetic_dataset(n=400, seed=seed)
    m.fit(X, y)
    return m


def _default_provider(ts):
    """A reasonable default feature provider for tests."""
    return make_features_for_hour(
        ts=ts,
        weather_condition="sunny",
        weather_condition_lag1d="sunny",
        t_outdoor=20.0,
        humidity=40.0,
        yield_factor_similar_days=[0.85, 0.88, 0.82],
    )


def _utc(year, month, day, hour=12, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# ── Contract ─────────────────────────────────────────────────────────────────
class TestContract:
    def test_naive_start_raises(self):
        sf = SolarForecaster(_clear_sky(), _fitted_atmospheric())
        with pytest.raises(ValueError):
            sf.predict_hourly(datetime(2026, 6, 21, 8), 24, _default_provider)

    def test_zero_horizon_raises(self):
        sf = SolarForecaster(_clear_sky(), _fitted_atmospheric())
        with pytest.raises(ValueError):
            sf.predict_hourly(_utc(2026, 6, 21, 8), 0, _default_provider)

    def test_negative_horizon_raises(self):
        sf = SolarForecaster(_clear_sky(), _fitted_atmospheric())
        with pytest.raises(ValueError):
            sf.predict_hourly(_utc(2026, 6, 21, 8), -1, _default_provider)

    def test_unfitted_atmospheric_model_raises(self):
        sf = SolarForecaster(_clear_sky(), AtmosphericFactorModel())
        with pytest.raises(RuntimeError, match="not fitted"):
            sf.predict_hourly(_utc(2026, 6, 21, 8), 24, _default_provider)

    def test_horizon_length(self):
        sf = SolarForecaster(_clear_sky(), _fitted_atmospheric())
        forecasts = sf.predict_hourly(_utc(2026, 6, 21, 0), 48, _default_provider)
        assert len(forecasts) == 48

    def test_start_is_snapped_to_top_of_hour(self):
        sf = SolarForecaster(_clear_sky(), _fitted_atmospheric())
        forecasts = sf.predict_hourly(
            _utc(2026, 6, 21, 8, 42), 3, _default_provider
        )
        assert forecasts[0].hour_start == _utc(2026, 6, 21, 8)
        assert forecasts[1].hour_start == _utc(2026, 6, 21, 9)
        assert forecasts[2].hour_start == _utc(2026, 6, 21, 10)


# ── Physical constraints ────────────────────────────────────────────────────
class TestPhysicalConstraints:
    def test_night_hours_are_zero_across_all_quantiles(self):
        sf = SolarForecaster(_clear_sky(), _fitted_atmospheric())
        forecasts = sf.predict_hourly(_utc(2026, 6, 21, 0), 24, _default_provider)
        # 02:00 UTC ≈ 04:00 local Madrid summer (still dark)
        for f in forecasts[:3] + forecasts[22:]:
            assert f.clear_sky_kwh == 0.0
            assert f.p10_kwh == 0.0
            assert f.p50_kwh == 0.0
            assert f.p90_kwh == 0.0

    def test_daytime_hours_have_positive_kwh(self):
        sf = SolarForecaster(_clear_sky(), _fitted_atmospheric())
        forecasts = sf.predict_hourly(_utc(2026, 6, 21, 0), 24, _default_provider)
        # 12:00 UTC at lon=-4 → past solar noon
        midday = [f for f in forecasts if f.hour_start.hour == 12][0]
        assert midday.clear_sky_kwh > 1.0
        assert midday.p50_kwh > 0.0

    def test_quantile_ordering_per_hour(self):
        sf = SolarForecaster(_clear_sky(), _fitted_atmospheric())
        forecasts = sf.predict_hourly(_utc(2026, 6, 21, 0), 48, _default_provider)
        for f in forecasts:
            # On hours with positive production, p10 ≤ p50 ≤ p90.
            assert f.p10_kwh <= f.p50_kwh + 1e-9
            assert f.p50_kwh <= f.p90_kwh + 1e-9

    def test_p90_never_exceeds_capacity_times_factor_cap(self):
        # Capacity 5 kWp → max 1.1 × 5 kWh/h baseline cap.
        sf = SolarForecaster(_clear_sky(capacity_kwp=5.0), _fitted_atmospheric())
        forecasts = sf.predict_hourly(_utc(2026, 6, 21, 0), 24, _default_provider)
        for f in forecasts:
            assert f.p90_kwh <= 1.1 * f.clear_sky_kwh + 1e-6

    def test_higher_capacity_produces_more(self):
        atm = _fitted_atmospheric()  # share model
        sf5 = SolarForecaster(_clear_sky(capacity_kwp=5.0), atm)
        sf10 = SolarForecaster(_clear_sky(capacity_kwp=10.0), atm)
        f5 = sf5.predict_hourly(_utc(2026, 6, 21, 11), 1, _default_provider)[0]
        f10 = sf10.predict_hourly(_utc(2026, 6, 21, 11), 1, _default_provider)[0]
        assert f10.p50_kwh == pytest.approx(2.0 * f5.p50_kwh, rel=1e-6)


# ── Feature-provider integration ────────────────────────────────────────────
class TestFeatureProvider:
    def test_provider_is_called_for_every_hour(self):
        sf = SolarForecaster(_clear_sky(), _fitted_atmospheric())
        calls = []

        def provider(ts):
            calls.append(ts)
            return _default_provider(ts)

        sf.predict_hourly(_utc(2026, 6, 21, 0), 24, provider)
        assert len(calls) == 24
        # Each call has a unique timestamp.
        assert len(set(calls)) == 24

    def test_cloudy_provider_lowers_kwh_vs_sunny_provider(self):
        atm = _fitted_atmospheric(seed=10)
        sf = SolarForecaster(_clear_sky(), atm)

        def sunny(ts):
            return make_features_for_hour(
                ts, "sunny", "sunny", 22.0, 40.0,
                yield_factor_similar_days=[0.85, 0.9],
            )

        def cloudy(ts):
            return make_features_for_hour(
                ts, "rainy", "rainy", 12.0, 80.0,
                yield_factor_similar_days=[0.25, 0.30],
            )

        sunny_total = sf.total_kwh(
            sf.predict_hourly(_utc(2026, 6, 21, 0), 24, sunny)
        )["p50_kwh"]
        cloudy_total = sf.total_kwh(
            sf.predict_hourly(_utc(2026, 6, 21, 0), 24, cloudy)
        )["p50_kwh"]
        assert cloudy_total < sunny_total


# ── Helpers / serialisation ────────────────────────────────────────────────
class TestHelpers:
    def test_hour_forecast_to_dict(self):
        f = HourForecast(
            hour_start=_utc(2026, 6, 21, 12),
            clear_sky_kwh=4.0,
            p10_kwh=2.0, p50_kwh=3.0, p90_kwh=4.0,
            factor_p10=0.5, factor_p50=0.75, factor_p90=1.0,
        )
        d = f.to_dict()
        assert d["hour_start"].startswith("2026-06-21T12:00")
        assert d["clear_sky_kwh"] == 4.0
        assert d["p50_kwh"] == 3.0

    def test_total_kwh_aggregates(self):
        sf = SolarForecaster(_clear_sky(), _fitted_atmospheric())
        forecasts = sf.predict_hourly(_utc(2026, 6, 21, 0), 12, _default_provider)
        totals = sf.total_kwh(forecasts)
        assert totals["p10_kwh"] >= 0
        assert totals["p50_kwh"] >= totals["p10_kwh"]
        assert totals["p90_kwh"] >= totals["p50_kwh"]
        assert totals["clear_sky_kwh"] >= totals["p50_kwh"] / 1.1
