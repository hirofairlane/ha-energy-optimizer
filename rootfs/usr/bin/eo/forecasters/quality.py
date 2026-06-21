"""Forecast quality tracking.

Records (predicted, actual) pairs per forecast series in a rolling window and
emits aggregate metrics — MAE, bias, calibration error — used downstream by:

  * Policy Layer → degraded mode trigger when MAE crosses thresholds
                   (SPEC §1.3 POL6 "Nivel 1").
  * Planner      → margin dynamic adjustment (SPEC §1.1 F5).

The tracker is a thin journal: append on observation, prune by retention,
compute stats on demand. Persistence is JSON-backed and atomic.

Quantile calibration:
    For a P10 series, ideally ~10 % of actuals should fall at or below the
    prediction. Calibration error = |empirical_coverage − target_quantile|.
    Reported per series so the planner can detect when the residual ML
    started to over- or under-cover.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Sentinel for "no recorded observations yet" so callers can distinguish
# from a perfectly-zero MAE on a series with one matching observation.
_NO_STATS_SENTINEL = float("nan")


@dataclass(frozen=True)
class ForecastQualityStats:
    """Aggregate metrics over a window for a single forecast series."""
    series: str
    samples: int
    mae: float          # mean absolute error
    bias: float         # mean(predicted - actual). Positive → over-predicts.
    rmse: float
    p50_actual: float   # median of actual observations (sanity)
    target_quantile: float | None       # for quantile series; None for point series
    empirical_coverage: float | None    # fraction of actuals ≤ predicted
    calibration_error: float | None     # |empirical_coverage − target_quantile|

    def to_dict(self) -> dict:
        d = {
            "series": self.series,
            "samples": self.samples,
            "mae": _round_finite(self.mae, 5),
            "bias": _round_finite(self.bias, 5),
            "rmse": _round_finite(self.rmse, 5),
            "p50_actual": _round_finite(self.p50_actual, 5),
        }
        if self.target_quantile is not None:
            d["target_quantile"] = self.target_quantile
            d["empirical_coverage"] = _round_finite(self.empirical_coverage, 4)
            d["calibration_error"] = _round_finite(self.calibration_error, 4)
        return d


def _round_finite(x: float, n: int) -> float:
    if not math.isfinite(x):
        return x
    return round(x, n)


@dataclass
class _Observation:
    ts: datetime
    predicted: float
    actual: float
    target_quantile: float | None = None  # set for quantile series


class ForecastQualityTracker:
    """Rolling per-series tracker.

    Series names are free-form (e.g. ``"solar.p50"``, ``"house.p90"``,
    ``"solar.deterministic"``). Each series has its own ring buffer keyed by
    ``retention_hours``; older entries are pruned on insert.
    """

    def __init__(self, retention_hours: float = 168.0):
        if retention_hours <= 0:
            raise ValueError("retention_hours must be positive")
        self.retention_hours: float = float(retention_hours)
        self._series: dict[str, deque[_Observation]] = {}

    # ── Recording ────────────────────────────────────────────────────────
    def record(
        self,
        series: str,
        ts: datetime,
        predicted: float,
        actual: float,
        target_quantile: float | None = None,
    ) -> None:
        if ts.tzinfo is None:
            raise ValueError("record requires a tz-aware datetime")
        if math.isnan(predicted) or math.isnan(actual):
            return  # silently drop NaN observations; caller can decide upstream
        obs = _Observation(
            ts=ts.astimezone(timezone.utc),
            predicted=float(predicted),
            actual=float(actual),
            target_quantile=target_quantile,
        )
        bucket = self._series.setdefault(series, deque())
        bucket.append(obs)
        self._prune(series, reference_ts=obs.ts)

    def _prune(self, series: str, reference_ts: datetime | None = None) -> None:
        bucket = self._series.get(series)
        if not bucket:
            return
        now = reference_ts or datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self.retention_hours)
        while bucket and bucket[0].ts < cutoff:
            bucket.popleft()

    # ── Querying ─────────────────────────────────────────────────────────
    def series_names(self) -> list[str]:
        return sorted(self._series.keys())

    def has_series(self, series: str) -> bool:
        return series in self._series and len(self._series[series]) > 0

    def stats(
        self,
        series: str,
        window_hours: float | None = None,
        now: datetime | None = None,
    ) -> ForecastQualityStats:
        """Aggregate stats for ``series`` over the trailing ``window_hours``.

        ``window_hours=None`` uses the full retained buffer.
        ``now`` defaults to current UTC time.
        """
        bucket = self._series.get(series)
        if not bucket:
            return ForecastQualityStats(
                series=series, samples=0,
                mae=_NO_STATS_SENTINEL, bias=_NO_STATS_SENTINEL,
                rmse=_NO_STATS_SENTINEL, p50_actual=_NO_STATS_SENTINEL,
                target_quantile=None, empirical_coverage=None,
                calibration_error=None,
            )

        now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if window_hours is not None:
            cutoff = now_utc - timedelta(hours=window_hours)
            observations: list[_Observation] = [o for o in bucket if o.ts >= cutoff]
        else:
            observations = list(bucket)

        if not observations:
            return ForecastQualityStats(
                series=series, samples=0,
                mae=_NO_STATS_SENTINEL, bias=_NO_STATS_SENTINEL,
                rmse=_NO_STATS_SENTINEL, p50_actual=_NO_STATS_SENTINEL,
                target_quantile=None, empirical_coverage=None,
                calibration_error=None,
            )

        n = len(observations)
        errors = [o.predicted - o.actual for o in observations]
        abs_errors = [abs(e) for e in errors]
        sq_errors = [e * e for e in errors]
        mae = sum(abs_errors) / n
        bias = sum(errors) / n
        rmse = math.sqrt(sum(sq_errors) / n)
        actuals_sorted = sorted(o.actual for o in observations)
        p50_actual = actuals_sorted[n // 2]

        # Calibration only makes sense if the series has a target quantile.
        quantile = observations[0].target_quantile
        if quantile is not None:
            coverage = sum(1 for o in observations if o.actual <= o.predicted) / n
            calibration_error = abs(coverage - quantile)
        else:
            coverage = None
            calibration_error = None

        return ForecastQualityStats(
            series=series, samples=n,
            mae=mae, bias=bias, rmse=rmse, p50_actual=p50_actual,
            target_quantile=quantile,
            empirical_coverage=coverage,
            calibration_error=calibration_error,
        )

    def stats_all(
        self,
        window_hours: float | None = None,
        now: datetime | None = None,
    ) -> dict[str, ForecastQualityStats]:
        return {s: self.stats(s, window_hours=window_hours, now=now) for s in self.series_names()}

    # ── Persistence ──────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "retention_hours": self.retention_hours,
            "series": {
                name: [
                    {
                        "ts": o.ts.isoformat(),
                        "predicted": o.predicted,
                        "actual": o.actual,
                        "target_quantile": o.target_quantile,
                    }
                    for o in bucket
                ]
                for name, bucket in self._series.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ForecastQualityTracker":
        tracker = cls(retention_hours=float(data.get("retention_hours", 168.0)))
        for name, observations in (data.get("series") or {}).items():
            bucket: deque[_Observation] = deque()
            for entry in observations:
                ts = datetime.fromisoformat(entry["ts"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                bucket.append(_Observation(
                    ts=ts,
                    predicted=float(entry["predicted"]),
                    actual=float(entry["actual"]),
                    target_quantile=entry.get("target_quantile"),
                ))
            tracker._series[name] = bucket
        return tracker

    def save(self, path: str | Path) -> None:
        """Atomic write: tmp + fsync + rename (SPEC §1.6 E1)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # NamedTemporaryFile + replace gives us atomic POSIX rename behaviour.
        with tempfile.NamedTemporaryFile(
            "w",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as f:
            json.dump(self.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
            tmp_path = f.name
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: str | Path) -> "ForecastQualityTracker":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
