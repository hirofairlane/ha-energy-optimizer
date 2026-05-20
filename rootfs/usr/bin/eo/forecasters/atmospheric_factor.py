"""Atmospheric factor model — ML residual on top of the clear-sky baseline.

Given the deterministic ``ClearSkyModel`` baseline, this model learns the
multiplicative factor in ``[0, 1.1]`` that converts a clear-sky theoretical
output into a realistic atmospheric-aware forecast.

    realistic_kwh = factor × clear_sky_kwh

Targets and features:
    target  : observed_solar_kwh / clear_sky_kwh, clipped to [0, 1.1].
              Factors slightly above 1.0 happen because of reflective ground,
              broken-cloud enhancement, and small calibration errors — clipping
              at 1.1 prevents the residual from absorbing outliers.

    features:
        - hour, weekday, month, day_of_year         (temporal context)
        - weather_condition_int                     (AEMET code → enum)
        - weather_condition_lag1d_int               (yesterday at same hour)
        - t_outdoor, humidity                       (atmospheric proxies)
        - yield_factor_similar_days_mean / _std     (empirical prior from
                                                     same-month days with same
                                                     weather class)

Quantile heads (P10, P50, P90) are three separate ``GradientBoostingRegressor``
instances with ``loss="quantile"``. The ScenarioBuilder downstream stratifies
which quantile to consume based on debt_state — see SPEC §1.2.

The model is **pure**: it makes no I/O, no DB access, no global state. The
caller builds feature vectors with :func:`make_features_for_hour` and supplies
them to ``fit()`` / ``predict()``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


# ── Weather condition encoding ──────────────────────────────────────────────
# Maps Home Assistant weather state strings (and AEMET equivalents) to a small
# ordinal int. Higher numbers mean cloudier / worse for PV.
WEATHER_CONDITION_CODES: dict[str, int] = {
    "clear-night":   0,
    "sunny":         1,
    "windy":         2,
    "partlycloudy":  2,
    "cloudy":        3,
    "fog":           3,
    "hail":          4,
    "snowy":         4,
    "snowy-rainy":   4,
    "rainy":         5,
    "pouring":       5,
    "lightning":     6,
    "lightning-rainy": 6,
    "exceptional":   6,
    "unknown":       3,  # treat unknown as average — safer than optimistic
}


def weather_to_int(condition: str | None) -> int:
    """Map a weather string to its ordinal code; falls back to 'unknown'."""
    if not condition:
        return WEATHER_CONDITION_CODES["unknown"]
    return WEATHER_CONDITION_CODES.get(
        condition.lower(), WEATHER_CONDITION_CODES["unknown"]
    )


# ── Feature vector schema (order matters — used by fit/predict) ────────────
FEATURE_NAMES: tuple[str, ...] = (
    "hour",
    "weekday",
    "month",
    "day_of_year",
    "weather_condition_int",
    "weather_condition_lag1d_int",
    "t_outdoor",
    "humidity",
    "yield_factor_similar_days_mean",
    "yield_factor_similar_days_std",
)


@dataclass(frozen=True)
class AtmosphericFactorFeatures:
    """Per-hour feature vector. The field order must match FEATURE_NAMES."""
    hour: int
    weekday: int
    month: int
    day_of_year: int
    weather_condition_int: int
    weather_condition_lag1d_int: int
    t_outdoor: float
    humidity: float
    yield_factor_similar_days_mean: float
    yield_factor_similar_days_std: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.hour,
                self.weekday,
                self.month,
                self.day_of_year,
                self.weather_condition_int,
                self.weather_condition_lag1d_int,
                self.t_outdoor,
                self.humidity,
                self.yield_factor_similar_days_mean,
                self.yield_factor_similar_days_std,
            ],
            dtype=float,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def make_features_for_hour(
    ts: datetime,
    weather_condition: str | None,
    weather_condition_lag1d: str | None,
    t_outdoor: float,
    humidity: float,
    yield_factor_similar_days: list[float] | None = None,
) -> AtmosphericFactorFeatures:
    """Build the feature vector for a single hour.

    ``yield_factor_similar_days`` is the list of historical observed factors
    on "similar" days (same month + same weather class) — typically pulled by
    the training/prediction pipeline before calling this function. Empty list
    or None falls back to an uninformative prior (mean=0.5, std=0.3) so the
    model still has something to learn against.
    """
    if yield_factor_similar_days:
        mean = statistics.fmean(yield_factor_similar_days)
        std = (
            statistics.pstdev(yield_factor_similar_days)
            if len(yield_factor_similar_days) > 1
            else 0.3
        )
    else:
        mean = 0.5
        std = 0.3

    return AtmosphericFactorFeatures(
        hour=ts.hour,
        weekday=ts.weekday(),
        month=ts.month,
        day_of_year=ts.timetuple().tm_yday,
        weather_condition_int=weather_to_int(weather_condition),
        weather_condition_lag1d_int=weather_to_int(weather_condition_lag1d),
        t_outdoor=float(t_outdoor),
        humidity=float(humidity),
        yield_factor_similar_days_mean=float(mean),
        yield_factor_similar_days_std=float(std),
    )


# ── Quantile model ──────────────────────────────────────────────────────────
DEFAULT_QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)

# Sensible defaults for an addon-sized dataset (months-year of hourly samples).
DEFAULT_GBR_KWARGS: dict = dict(
    n_estimators=120,
    max_depth=4,
    learning_rate=0.05,
    min_samples_leaf=10,
    subsample=0.9,
    random_state=42,
)


class AtmosphericFactorModel:
    """Ensemble of one GBR per quantile, sharing features and labels."""

    def __init__(
        self,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        gbr_kwargs: dict | None = None,
    ):
        if not quantiles:
            raise ValueError("at least one quantile required")
        for q in quantiles:
            if not 0.0 < q < 1.0:
                raise ValueError(f"quantile must be in (0, 1), got {q}")
        self.quantiles: tuple[float, ...] = tuple(sorted(quantiles))
        self._gbr_kwargs: dict = {**DEFAULT_GBR_KWARGS, **(gbr_kwargs or {})}
        self._models: dict[float, GradientBoostingRegressor] = {}

    # ── State queries ────────────────────────────────────────────────────
    @property
    def fitted(self) -> bool:
        return bool(self._models)

    @property
    def quantile_names(self) -> list[str]:
        return [self._qname(q) for q in self.quantiles]

    @staticmethod
    def _qname(q: float) -> str:
        return f"p{int(round(q * 100)):02d}"

    @staticmethod
    def _validate_X(X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {arr.shape}")
        if arr.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"X must have {len(FEATURE_NAMES)} columns "
                f"({', '.join(FEATURE_NAMES)}), got {arr.shape[1]}"
            )
        return arr

    # ── Training ─────────────────────────────────────────────────────────
    def fit(self, X, y, sample_weight=None) -> "AtmosphericFactorModel":
        X_arr = self._validate_X(X)
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 1 or len(y_arr) != len(X_arr):
            raise ValueError("y must be 1-D and same length as X")

        # Target lives in [0, 1.1] by physical bounds — clip outliers.
        y_clipped = np.clip(y_arr, 0.0, 1.1)

        self._models = {}
        for q in self.quantiles:
            gbr = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                **self._gbr_kwargs,
            )
            gbr.fit(X_arr, y_clipped, sample_weight=sample_weight)
            self._models[q] = gbr
        return self

    # ── Prediction ───────────────────────────────────────────────────────
    def predict(self, X) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError(
                "AtmosphericFactorModel.predict() called before fit()"
            )
        X_arr = self._validate_X(X)
        out: dict[str, np.ndarray] = {}
        for q in self.quantiles:
            pred = self._models[q].predict(X_arr)
            out[self._qname(q)] = np.clip(pred, 0.0, 1.1)
        # Enforce quantile ordering: P10 ≤ P50 ≤ P90 element-wise. If the GBRs
        # disagree (rare but possible on tiny datasets), sort within each row.
        if len(self.quantiles) > 1:
            stacked = np.stack(
                [out[self._qname(q)] for q in self.quantiles], axis=1
            )
            stacked.sort(axis=1)
            for i, q in enumerate(self.quantiles):
                out[self._qname(q)] = stacked[:, i]
        return out

    def predict_one(self, features: AtmosphericFactorFeatures) -> dict[str, float]:
        """Convenience: predict for a single feature vector."""
        result = self.predict(features.to_array().reshape(1, -1))
        return {k: float(v[0]) for k, v in result.items()}

    # ── Persistence ──────────────────────────────────────────────────────
    def save(self, path) -> None:
        joblib.dump(
            {
                "quantiles": list(self.quantiles),
                "gbr_kwargs": self._gbr_kwargs,
                "feature_names": list(FEATURE_NAMES),
                "models": self._models,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "AtmosphericFactorModel":
        data = joblib.load(path)
        # Cross-version safety: if features were renamed/reordered, refuse load.
        saved_features = tuple(data.get("feature_names", FEATURE_NAMES))
        if saved_features != FEATURE_NAMES:
            raise RuntimeError(
                "Saved model feature schema does not match the current "
                f"FEATURE_NAMES — refusing to load.\n"
                f"  saved : {saved_features}\n"
                f"  current: {FEATURE_NAMES}"
            )
        m = cls(
            quantiles=tuple(data["quantiles"]),
            gbr_kwargs=data["gbr_kwargs"],
        )
        m._models = data["models"]
        return m
