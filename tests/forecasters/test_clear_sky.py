"""Tests for eo.forecasters.clear_sky.

Sanity checks on physical behaviour rather than exact numeric matching against
a reference library (we don't depend on pvlib). The validation strategy:

  * Power must be zero at solar midnight and through the night.
  * Power must peak somewhere near solar noon on a clear day.
  * Daily energy must scale linearly with capacity_kwp.
  * Summer days must produce more than winter days at mid-latitudes.
  * A south-facing panel in the northern hemisphere must outperform a
    north-facing one on average.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from eo.forecasters.clear_sky import (
    BEAM_ATTENUATION,
    DIFFUSE_FRACTION,
    SOLAR_CONSTANT_W_M2,
    ClearSkyConfig,
    ClearSkyModel,
    solar_position,
)

GUADARRAMA_LAT = 40.65
GUADARRAMA_LON = -4.0


def _config(**kw) -> ClearSkyConfig:
    return ClearSkyConfig(
        latitude=kw.pop("latitude", GUADARRAMA_LAT),
        longitude=kw.pop("longitude", GUADARRAMA_LON),
        capacity_kwp=kw.pop("capacity_kwp", 5.0),
        tilt_deg=kw.pop("tilt_deg", None),
        azimuth_deg=kw.pop("azimuth_deg", 180.0),
        system_efficiency=kw.pop("system_efficiency", 0.85),
    )


def _utc(year, month, day, hour=12, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# ── Solar position ──────────────────────────────────────────────────────────
class TestSolarPosition:
    def test_naive_datetime_raises(self):
        with pytest.raises(ValueError):
            solar_position(datetime(2026, 6, 21, 12), GUADARRAMA_LAT, GUADARRAMA_LON)

    def test_sun_high_at_summer_noon_north_hemisphere(self):
        sp = solar_position(_utc(2026, 6, 21, 12), 40.65, -4.0)
        assert sp.elevation_deg > 65.0
        # Around solar noon at lon=-4 the sun is roughly south-ish.
        assert 150 < sp.azimuth_deg < 210

    def test_sun_below_horizon_at_midnight(self):
        sp = solar_position(_utc(2026, 6, 21, 0), 40.65, -4.0)
        assert sp.elevation_deg < 0
        assert sp.air_mass == float("inf")

    def test_declination_extremes(self):
        # Northern hemisphere summer solstice → +23.4°-ish
        sp_summer = solar_position(_utc(2026, 6, 21, 12), 0.0, 0.0)
        assert 22.5 < sp_summer.declination_deg < 24.0
        # Winter solstice → -23.4°-ish
        sp_winter = solar_position(_utc(2026, 12, 21, 12), 0.0, 0.0)
        assert -24.0 < sp_winter.declination_deg < -22.5

    def test_morning_east_afternoon_west(self):
        sp_morning = solar_position(_utc(2026, 6, 21, 8), 40.65, -4.0)
        sp_afternoon = solar_position(_utc(2026, 6, 21, 16), 40.65, -4.0)
        # Morning sun is to the east (azimuth 90 ± wide)
        assert sp_morning.azimuth_deg < 180
        # Afternoon sun is to the west (azimuth > 180)
        assert sp_afternoon.azimuth_deg > 180


# ── Clear-sky irradiance & power ───────────────────────────────────────────
class TestClearSkyModel:
    def test_zero_irradiance_at_night(self):
        model = ClearSkyModel(_config())
        # 3:00 UTC ≈ 4:00 local Madrid winter; sun below horizon
        for h in (0, 1, 2, 3, 22, 23):
            assert model.irradiance_w_m2(_utc(2026, 1, 15, h)) == 0.0
            assert model.power_w(_utc(2026, 1, 15, h)) == 0.0

    def test_irradiance_peak_around_noon(self):
        model = ClearSkyModel(_config())
        # Sample every hour on the longest day; max should fall between 10 and 14 UTC.
        irr_by_hour = {
            h: model.irradiance_w_m2(_utc(2026, 6, 21, h)) for h in range(24)
        }
        peak_h = max(irr_by_hour, key=irr_by_hour.get)
        assert 10 <= peak_h <= 14
        # Peak should be substantial on a clear summer day.
        assert irr_by_hour[peak_h] > 500

    def test_power_scales_linearly_with_capacity(self):
        m1 = ClearSkyModel(_config(capacity_kwp=5.0))
        m2 = ClearSkyModel(_config(capacity_kwp=10.0))
        ts = _utc(2026, 6, 21, 12)
        assert m2.power_w(ts) == pytest.approx(2.0 * m1.power_w(ts), rel=1e-6)

    def test_summer_day_beats_winter_day_at_midlatitude(self):
        # Tilt defaults to latitude, which intentionally biases the panel
        # toward catching low winter sun → summer/winter ratio is muted.
        # 1.5× is still a robust signal that the geometry works.
        model = ClearSkyModel(_config())
        june = sum(
            model.power_w(_utc(2026, 6, 21, h)) for h in range(24)
        )
        december = sum(
            model.power_w(_utc(2026, 12, 21, h)) for h in range(24)
        )
        assert june > 1.5 * december

    def test_south_facing_beats_north_facing_in_northern_hemisphere(self):
        south = ClearSkyModel(_config(azimuth_deg=180))
        north = ClearSkyModel(_config(azimuth_deg=0))
        ts = _utc(2026, 6, 21, 12)
        assert south.power_w(ts) > north.power_w(ts)

    def test_default_tilt_equals_latitude(self):
        cfg = _config(latitude=40.0, tilt_deg=None)
        assert cfg.effective_tilt_deg() == 40.0

    def test_explicit_tilt_used_when_set(self):
        cfg = _config(latitude=40.0, tilt_deg=10.0)
        assert cfg.effective_tilt_deg() == 10.0

    def test_efficiency_factor_applied(self):
        m_full = ClearSkyModel(_config(system_efficiency=1.0))
        m_half = ClearSkyModel(_config(system_efficiency=0.5))
        ts = _utc(2026, 6, 21, 12)
        assert m_half.power_w(ts) == pytest.approx(0.5 * m_full.power_w(ts), rel=1e-6)


# ── Energy integration ─────────────────────────────────────────────────────
class TestEnergyIntegration:
    def test_kwh_for_hour_is_non_negative(self):
        model = ClearSkyModel(_config())
        for h in range(24):
            kwh = model.kwh_for_hour(_utc(2026, 6, 21, h))
            assert kwh >= 0

    def test_kwh_for_night_hour_is_zero(self):
        model = ClearSkyModel(_config())
        assert model.kwh_for_hour(_utc(2026, 12, 21, 2)) == 0.0

    def test_daily_kwh_sane_for_5kwp_summer_solstice(self):
        """A 5 kWp south-facing array at 40° latitude on June 21 should produce
        somewhere in the 20-40 kWh range on a perfectly clear day."""
        model = ClearSkyModel(_config(capacity_kwp=5.0))
        series = model.kwh_series(
            _utc(2026, 6, 21, 0), _utc(2026, 6, 22, 0)
        )
        total = sum(kwh for _, kwh in series)
        assert 15 < total < 50, f"Daily summer kWh out of range: {total:.1f}"

    def test_daily_kwh_lower_in_winter(self):
        model = ClearSkyModel(_config(capacity_kwp=5.0))
        june = sum(
            kwh for _, kwh in model.kwh_series(
                _utc(2026, 6, 21, 0), _utc(2026, 6, 22, 0)
            )
        )
        december = sum(
            kwh for _, kwh in model.kwh_series(
                _utc(2026, 12, 21, 0), _utc(2026, 12, 22, 0)
            )
        )
        assert december < june

    def test_kwh_series_naive_datetimes_raise(self):
        model = ClearSkyModel(_config())
        with pytest.raises(ValueError):
            model.kwh_series(
                datetime(2026, 6, 21),
                datetime(2026, 6, 22),
            )

    def test_kwh_series_aligned_to_hour_boundaries(self):
        model = ClearSkyModel(_config())
        series = model.kwh_series(
            _utc(2026, 6, 21, 8, 37),  # not a whole hour
            _utc(2026, 6, 21, 11, 0),
        )
        # Should produce entries at 08:00, 09:00, 10:00 → 3 entries
        assert len(series) == 3
        starts = [ts.hour for ts, _ in series]
        assert starts == [8, 9, 10]

    def test_kwh_for_hour_invalid_samples_raises(self):
        model = ClearSkyModel(_config())
        with pytest.raises(ValueError):
            model.kwh_for_hour(_utc(2026, 6, 21, 12), samples=0)


# ── Module constants sanity ─────────────────────────────────────────────────
class TestConstants:
    def test_solar_constant_in_expected_range(self):
        # Real-world value 1361-1367 W/m²
        assert 1300 < SOLAR_CONSTANT_W_M2 < 1400

    def test_attenuation_and_diffuse_fractions_sane(self):
        assert 0.0 < BEAM_ATTENUATION < 1.0
        assert 0.0 < DIFFUSE_FRACTION < 0.5
