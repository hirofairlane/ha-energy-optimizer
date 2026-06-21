"""Tests for eo.forecasters.quality."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from eo.forecasters.quality import ForecastQualityStats, ForecastQualityTracker


def _utc(hour_offset: int = 0, base: datetime | None = None) -> datetime:
    base = base or datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(hours=hour_offset)


# ── Recording contract ─────────────────────────────────────────────────────
class TestRecording:
    def test_naive_ts_raises(self):
        t = ForecastQualityTracker()
        with pytest.raises(ValueError):
            t.record("s", datetime(2026, 5, 20, 12), 1.0, 1.1)

    def test_nan_observations_silently_dropped(self):
        t = ForecastQualityTracker()
        t.record("s", _utc(), float("nan"), 1.0)
        t.record("s", _utc(), 1.0, float("nan"))
        assert t.stats("s").samples == 0

    def test_series_names_tracked(self):
        t = ForecastQualityTracker()
        t.record("solar.p50", _utc(), 4.0, 4.2)
        t.record("house.p50", _utc(), 1.0, 0.9)
        assert t.series_names() == ["house.p50", "solar.p50"]

    def test_has_series(self):
        t = ForecastQualityTracker()
        assert t.has_series("solar.p50") is False
        t.record("solar.p50", _utc(), 4.0, 4.2)
        assert t.has_series("solar.p50") is True


# ── Stats aggregation ──────────────────────────────────────────────────────
class TestStats:
    def test_empty_series_returns_nan_with_zero_samples(self):
        t = ForecastQualityTracker()
        s = t.stats("never_recorded")
        assert s.samples == 0
        assert math.isnan(s.mae)
        assert math.isnan(s.bias)
        assert s.empirical_coverage is None

    def test_mae_bias_rmse_known_pattern(self):
        t = ForecastQualityTracker()
        # predictions 1.0, 1.0, 1.0; actuals 1.0, 2.0, 0.0
        # errors = predicted - actual = 0, -1, +1
        # mae = (0+1+1)/3 = 0.666
        # bias = (0-1+1)/3 = 0
        # rmse = sqrt((0+1+1)/3) = sqrt(0.666)
        for actual in (1.0, 2.0, 0.0):
            t.record("series", _utc(), 1.0, actual)
        s = t.stats("series")
        assert s.samples == 3
        assert s.mae == pytest.approx(2/3, rel=1e-3)
        assert s.bias == pytest.approx(0.0, abs=1e-9)
        assert s.rmse == pytest.approx(math.sqrt(2/3), rel=1e-3)

    def test_positive_bias_when_over_predicting(self):
        t = ForecastQualityTracker()
        for actual in (1.0, 1.5, 2.0):
            t.record("over", _utc(), actual + 0.5, actual)
        s = t.stats("over")
        assert s.bias == pytest.approx(0.5, rel=1e-3)

    def test_window_hours_respected(self):
        t = ForecastQualityTracker(retention_hours=1000)
        base = _utc()
        t.record("s", base - timedelta(hours=100), 1.0, 0.0)  # large error, old
        t.record("s", base - timedelta(hours=1), 1.0, 1.0)    # zero error, recent
        # Full buffer: MAE = (1 + 0) / 2 = 0.5
        full = t.stats("s", now=base)
        assert full.samples == 2
        assert full.mae == pytest.approx(0.5, rel=1e-3)
        # 24h window: only the recent one survives → MAE = 0
        recent = t.stats("s", window_hours=24, now=base)
        assert recent.samples == 1
        assert recent.mae == pytest.approx(0.0, abs=1e-9)


# ── Quantile calibration ───────────────────────────────────────────────────
class TestCalibration:
    def test_perfectly_calibrated_p50(self):
        t = ForecastQualityTracker()
        # 100 observations: actual uniformly above and below the prediction.
        for i in range(100):
            actual = 1.0 + (i - 50) * 0.01  # range from 0.5 to 1.49
            t.record("solar.p50", _utc(i), 1.0, actual, target_quantile=0.5)
        s = t.stats("solar.p50")
        # Empirical coverage should be close to 0.5 (half ≤ predicted)
        assert s.empirical_coverage == pytest.approx(0.5, abs=0.05)
        assert s.calibration_error is not None
        assert s.calibration_error < 0.1

    def test_over_predicting_p10_has_high_coverage(self):
        # P10 should have ~10 % coverage; if we over-predict, coverage rises.
        t = ForecastQualityTracker()
        for i in range(50):
            actual = 0.5
            t.record("solar.p10", _utc(i), 1.5, actual, target_quantile=0.10)
        s = t.stats("solar.p10")
        # 100 % coverage (all actuals ≤ prediction) → big calibration error.
        assert s.empirical_coverage == pytest.approx(1.0, abs=1e-9)
        assert s.calibration_error == pytest.approx(0.9, abs=1e-9)

    def test_target_quantile_preserved_in_stats(self):
        t = ForecastQualityTracker()
        t.record("solar.p90", _utc(), 4.0, 3.0, target_quantile=0.9)
        s = t.stats("solar.p90")
        assert s.target_quantile == 0.9

    def test_no_target_quantile_means_no_calibration(self):
        t = ForecastQualityTracker()
        t.record("solar.deterministic", _utc(), 4.0, 4.2)
        s = t.stats("solar.deterministic")
        assert s.target_quantile is None
        assert s.empirical_coverage is None
        assert s.calibration_error is None


# ── Retention ──────────────────────────────────────────────────────────────
class TestRetention:
    def test_old_entries_pruned_on_insert(self):
        t = ForecastQualityTracker(retention_hours=24)
        base = _utc()
        # Insert an old entry first, then a new one. Pruning happens against
        # the latest insert's timestamp, so the old one falls outside the 24h
        # window and gets dropped.
        t.record("s", base - timedelta(hours=48), 1.0, 0.0)
        assert t.stats("s").samples == 1
        t.record("s", base, 1.0, 1.0)
        assert t.stats("s").samples == 1  # old one pruned, only new survives

    def test_retention_zero_rejected(self):
        with pytest.raises(ValueError):
            ForecastQualityTracker(retention_hours=0)
        with pytest.raises(ValueError):
            ForecastQualityTracker(retention_hours=-1)


# ── Multiple series ────────────────────────────────────────────────────────
class TestMultipleSeries:
    def test_stats_per_series_independent(self):
        t = ForecastQualityTracker()
        for _ in range(10):
            t.record("solar.p50", _utc(), 4.0, 4.0)   # zero error
            t.record("house.p50", _utc(), 1.0, 2.0)   # 1.0 error consistently
        assert t.stats("solar.p50").mae == pytest.approx(0.0, abs=1e-9)
        assert t.stats("house.p50").mae == pytest.approx(1.0, rel=1e-3)

    def test_stats_all_returns_dict(self):
        t = ForecastQualityTracker()
        t.record("solar.p10", _utc(), 1.0, 0.5, target_quantile=0.1)
        t.record("house.p50", _utc(), 1.0, 1.2)
        result = t.stats_all()
        assert set(result.keys()) == {"solar.p10", "house.p50"}
        assert all(isinstance(v, ForecastQualityStats) for v in result.values())


# ── Persistence ────────────────────────────────────────────────────────────
class TestPersistence:
    def test_round_trip_in_memory(self):
        t = ForecastQualityTracker(retention_hours=72)
        t.record("solar.p50", _utc(), 4.0, 4.2, target_quantile=0.5)
        t.record("house.p50", _utc(1), 1.0, 0.9)
        data = t.to_dict()
        t2 = ForecastQualityTracker.from_dict(data)
        assert t.stats("solar.p50").samples == t2.stats("solar.p50").samples
        assert t.stats("house.p50").samples == t2.stats("house.p50").samples

    def test_save_load_round_trip(self, tmp_path: Path):
        t = ForecastQualityTracker(retention_hours=72)
        t.record("solar.p50", _utc(), 4.0, 4.2, target_quantile=0.5)
        t.record("house.p50", _utc(1), 1.0, 0.9)
        path = tmp_path / "quality.json"
        t.save(path)
        assert path.exists()
        t2 = ForecastQualityTracker.load(path)
        assert t2.has_series("solar.p50")
        assert t2.has_series("house.p50")
        assert t2.stats("solar.p50").samples == 1

    def test_load_nonexistent_returns_empty(self, tmp_path: Path):
        t = ForecastQualityTracker.load(tmp_path / "does_not_exist.json")
        assert t.series_names() == []
        assert t.retention_hours == 168.0  # default

    def test_save_is_atomic_no_partial_on_disk(self, tmp_path: Path):
        """Even if we look during save, the destination file is either the old
        version or the new version, never a half-written one. We check that no
        leftover .tmp files survive after a successful save."""
        t = ForecastQualityTracker()
        t.record("s", _utc(), 1.0, 1.0)
        path = tmp_path / "q.json"
        t.save(path)
        leftover = list(tmp_path.glob("*.tmp"))
        assert leftover == []
        # And the written file parses as valid JSON.
        json.loads(path.read_text(encoding="utf-8"))

    def test_save_creates_parent_dir(self, tmp_path: Path):
        t = ForecastQualityTracker()
        t.record("s", _utc(), 1.0, 1.0)
        deep_path = tmp_path / "nested" / "dirs" / "quality.json"
        t.save(deep_path)
        assert deep_path.exists()

    def test_naive_iso_string_normalised_to_utc(self):
        """JSON written by a buggy producer that omits tzinfo must round-trip
        as UTC, not raise on load."""
        data = {
            "retention_hours": 24.0,
            "series": {
                "s": [
                    {"ts": "2026-05-20T12:00:00", "predicted": 1.0, "actual": 1.1, "target_quantile": None},
                ],
            },
        }
        t = ForecastQualityTracker.from_dict(data)
        assert t.stats("s").samples == 1
