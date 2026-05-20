"""Forecaster training helpers.

Pure functions that take already-prepared training samples and return fitted
forecaster artefacts. The data-fetch side (InfluxDB / MariaDB / HA recorder)
is intentionally **not** in this module — that connection happens later in
the integration phase, where the addon's existing data-source helpers can
plug in without dragging the whole monolith into the pure forecasting tests.

The split makes the training step trivially unit-testable and reusable for:
  * Nightly retraining triggered by the addon's APScheduler cron.
  * Offline experimentation (e.g. ROI projection pipeline notebook).
  * Backfill jobs that want to retrain on a custom date window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from eo.forecasters.atmospheric_factor import (
    DEFAULT_QUANTILES,
    AtmosphericFactorFeatures,
    AtmosphericFactorModel,
)
from eo.forecasters.house_forecaster import (
    DEFAULT_HOUSE_QUANTILES,
    HouseFeatures,
    HouseForecaster,
)


# Below this many samples we refuse to train — a model fitted on < 50 hours
# of history is just memorising and will mislead the planner.
MIN_TRAIN_SAMPLES: int = 50


@dataclass(frozen=True)
class TrainingReport:
    """Summary of a training run."""
    model_kind: str
    samples_used: int
    samples_dropped: int
    in_sample_mae_p50: float
    in_sample_bias_p50: float

    def to_dict(self) -> dict:
        return {
            "model_kind": self.model_kind,
            "samples_used": self.samples_used,
            "samples_dropped": self.samples_dropped,
            "in_sample_mae_p50": round(self.in_sample_mae_p50, 5),
            "in_sample_bias_p50": round(self.in_sample_bias_p50, 5),
        }


# ── Observation → target helpers ────────────────────────────────────────────
def compute_atmospheric_factor(observed_solar_kwh: float, clear_sky_kwh: float) -> float:
    """Convert an observed solar production hour into its atmospheric factor.

    Returns ``observed / clear_sky`` clipped to ``[0, 1.1]``. Returns 0 when
    ``clear_sky_kwh`` is ≤ 0 (sun below the horizon).
    """
    if clear_sky_kwh <= 0:
        return 0.0
    factor = observed_solar_kwh / clear_sky_kwh
    return max(0.0, min(1.1, factor))


# ── Validation ──────────────────────────────────────────────────────────────
def _clean_samples(
    samples: Sequence[tuple[object, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop NaN / inf rows and return (X, y, n_dropped)."""
    if not samples:
        return np.empty((0, 0)), np.empty((0,)), 0
    n_cols = len(samples[0][1])
    rows: list[np.ndarray] = []
    ys: list[float] = []
    dropped = 0
    for _, feat_arr, y in samples:
        if (
            not isinstance(feat_arr, np.ndarray)
            or feat_arr.shape != (n_cols,)
            or not np.all(np.isfinite(feat_arr))
            or not math.isfinite(y)
        ):
            dropped += 1
            continue
        rows.append(feat_arr.astype(float))
        ys.append(float(y))
    if not rows:
        return np.empty((0, n_cols)), np.empty((0,)), dropped
    return np.vstack(rows), np.asarray(ys, dtype=float), dropped


def _in_sample_p50_metrics(
    predictions_p50: np.ndarray, actuals: np.ndarray
) -> tuple[float, float]:
    errors = predictions_p50 - actuals
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    return mae, bias


# ── Atmospheric factor training ─────────────────────────────────────────────
def train_atmospheric_factor_model(
    samples: Sequence[tuple[object, AtmosphericFactorFeatures, float]],
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    gbr_kwargs: dict | None = None,
    min_samples: int = MIN_TRAIN_SAMPLES,
) -> tuple[AtmosphericFactorModel, TrainingReport]:
    """Train an :class:`AtmosphericFactorModel`.

    ``samples`` is an iterable of ``(timestamp, features, observed_factor)``
    triples. The timestamp is not used by the trainer itself (it lives there
    for the caller's bookkeeping) but is part of the signature so the same
    structure is reused for downstream auditing.
    """
    arr_samples = [(ts, f.to_array(), y) for ts, f, y in samples]
    X, y, dropped = _clean_samples(arr_samples)
    if len(X) < min_samples:
        raise ValueError(
            f"Not enough valid samples to train AtmosphericFactorModel: "
            f"{len(X)} < {min_samples}"
        )

    model = AtmosphericFactorModel(quantiles=quantiles, gbr_kwargs=gbr_kwargs)
    model.fit(X, y)

    in_sample = model.predict(X)["p50"]
    mae, bias = _in_sample_p50_metrics(in_sample, y)
    report = TrainingReport(
        model_kind="atmospheric_factor",
        samples_used=len(X),
        samples_dropped=dropped,
        in_sample_mae_p50=mae,
        in_sample_bias_p50=bias,
    )
    return model, report


# ── House load training ─────────────────────────────────────────────────────
def train_house_forecaster(
    samples: Sequence[tuple[object, HouseFeatures, float]],
    quantiles: tuple[float, ...] = DEFAULT_HOUSE_QUANTILES,
    gbr_kwargs: dict | None = None,
    kwh_cap_per_hour: float | None = None,
    min_samples: int = MIN_TRAIN_SAMPLES,
) -> tuple[HouseForecaster, TrainingReport]:
    """Train a :class:`HouseForecaster`."""
    arr_samples = [(ts, f.to_array(), y) for ts, f, y in samples]
    X, y, dropped = _clean_samples(arr_samples)
    if len(X) < min_samples:
        raise ValueError(
            f"Not enough valid samples to train HouseForecaster: "
            f"{len(X)} < {min_samples}"
        )

    cap_kwargs = {} if kwh_cap_per_hour is None else {"kwh_cap_per_hour": kwh_cap_per_hour}
    model = HouseForecaster(
        quantiles=quantiles, gbr_kwargs=gbr_kwargs, **cap_kwargs
    )
    model.fit(X, y)

    in_sample = model.predict(X)["p50"]
    mae, bias = _in_sample_p50_metrics(in_sample, y)
    report = TrainingReport(
        model_kind="house",
        samples_used=len(X),
        samples_dropped=dropped,
        in_sample_mae_p50=mae,
        in_sample_bias_p50=bias,
    )
    return model, report
