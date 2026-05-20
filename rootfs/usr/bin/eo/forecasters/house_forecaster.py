"""House load forecaster — quantile GBR over per-hour exogenous features.

Predicts hourly household consumption (kWh) for each target hour ``t``, with
P10/P50/P90 heads. By design the model is **not chained**: features for
predicting hour ``t`` come exclusively from the past (real observations) and
from the schedule input the planner already decided, never from the model's
own earlier predictions.

This is the deliberate counterpoint to the v3.x ``predict_soc`` autoregressive
chain that produced an inflated R²=0.998 by exploiting autocorrelation. Each
horizon hour gets its own independent feature vector; quality degrades
gracefully with horizon distance, instead of collapsing into a self-consistent
fantasy.

Features (10 total, order matters):
    hour, weekday, month, is_weekend                     — calendar
    t_outdoor                                            — forecast at target hour
    t_outdoor_lag24h                                     — observed at (target - 24h)
    house_kwh_yesterday_same_hour                        — observed (target - 24h)
    house_kwh_last_observed                              — last closed hour at predict time
    custom_loads_planned_watts                           — planner schedule input
    deferred_loads_planned_watts                         — planner schedule input

Target: observed ``house_kwh`` for the hour. Clipping rule: clipped to
``[0, kwh_cap_per_hour]`` (default 30 kWh) to absorb single-hour outliers.

Persistence: ``joblib.dump`` with a feature-name schema check. Loading a
model from a previous schema raises rather than silently misalign features.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


HOUSE_FEATURE_NAMES: tuple[str, ...] = (
    "hour",
    "weekday",
    "month",
    "is_weekend",
    "t_outdoor",
    "t_outdoor_lag24h",
    "house_kwh_yesterday_same_hour",
    "house_kwh_last_observed",
    "custom_loads_planned_watts",
    "deferred_loads_planned_watts",
)


# Absolute cap for per-hour household consumption used to clip outlier training
# targets. A normal home rarely passes a few kW continuous; 30 kWh/h is the
# upper bound where we treat the value as a sensor glitch rather than reality.
HOUSE_KWH_PER_HOUR_CAP: float = 30.0


@dataclass(frozen=True)
class HouseFeatures:
    hour: int
    weekday: int
    month: int
    is_weekend: int
    t_outdoor: float
    t_outdoor_lag24h: float
    house_kwh_yesterday_same_hour: float
    house_kwh_last_observed: float
    custom_loads_planned_watts: float
    deferred_loads_planned_watts: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.hour,
                self.weekday,
                self.month,
                self.is_weekend,
                self.t_outdoor,
                self.t_outdoor_lag24h,
                self.house_kwh_yesterday_same_hour,
                self.house_kwh_last_observed,
                self.custom_loads_planned_watts,
                self.deferred_loads_planned_watts,
            ],
            dtype=float,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def make_house_features_for_hour(
    target_ts: datetime,
    t_outdoor: float,
    t_outdoor_lag24h: float,
    house_kwh_yesterday_same_hour: float,
    house_kwh_last_observed: float,
    custom_loads_planned_watts: float = 0.0,
    deferred_loads_planned_watts: float = 0.0,
) -> HouseFeatures:
    """Build feature vector for target hour ``target_ts``.

    All non-temporal inputs must be either observed values (past) or already
    decided schedule inputs — never the model's own predictions, by design.
    """
    weekday = target_ts.weekday()
    return HouseFeatures(
        hour=target_ts.hour,
        weekday=weekday,
        month=target_ts.month,
        is_weekend=1 if weekday >= 5 else 0,
        t_outdoor=float(t_outdoor),
        t_outdoor_lag24h=float(t_outdoor_lag24h),
        house_kwh_yesterday_same_hour=float(house_kwh_yesterday_same_hour),
        house_kwh_last_observed=float(house_kwh_last_observed),
        custom_loads_planned_watts=float(custom_loads_planned_watts),
        deferred_loads_planned_watts=float(deferred_loads_planned_watts),
    )


# ── Quantile model ──────────────────────────────────────────────────────────
DEFAULT_HOUSE_QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)

DEFAULT_HOUSE_GBR_KWARGS: dict = dict(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    min_samples_leaf=10,
    subsample=0.9,
    random_state=42,
)


class HouseForecaster:
    """Standalone household consumption forecaster with quantile heads."""

    def __init__(
        self,
        quantiles: tuple[float, ...] = DEFAULT_HOUSE_QUANTILES,
        gbr_kwargs: dict | None = None,
        kwh_cap_per_hour: float = HOUSE_KWH_PER_HOUR_CAP,
    ):
        if not quantiles:
            raise ValueError("at least one quantile required")
        for q in quantiles:
            if not 0.0 < q < 1.0:
                raise ValueError(f"quantile must be in (0, 1), got {q}")
        if kwh_cap_per_hour <= 0:
            raise ValueError("kwh_cap_per_hour must be positive")
        self.quantiles: tuple[float, ...] = tuple(sorted(quantiles))
        self.kwh_cap_per_hour: float = float(kwh_cap_per_hour)
        self._gbr_kwargs: dict = {**DEFAULT_HOUSE_GBR_KWARGS, **(gbr_kwargs or {})}
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
        if arr.shape[1] != len(HOUSE_FEATURE_NAMES):
            raise ValueError(
                f"X must have {len(HOUSE_FEATURE_NAMES)} columns "
                f"({', '.join(HOUSE_FEATURE_NAMES)}), got {arr.shape[1]}"
            )
        return arr

    # ── Training ─────────────────────────────────────────────────────────
    def fit(self, X, y, sample_weight=None) -> "HouseForecaster":
        X_arr = self._validate_X(X)
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 1 or len(y_arr) != len(X_arr):
            raise ValueError("y must be 1-D and same length as X")

        # House kWh per hour is non-negative and bounded.
        y_clipped = np.clip(y_arr, 0.0, self.kwh_cap_per_hour)

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
                "HouseForecaster.predict() called before fit()"
            )
        X_arr = self._validate_X(X)
        out: dict[str, np.ndarray] = {}
        for q in self.quantiles:
            pred = self._models[q].predict(X_arr)
            out[self._qname(q)] = np.clip(pred, 0.0, self.kwh_cap_per_hour)
        if len(self.quantiles) > 1:
            stacked = np.stack(
                [out[self._qname(q)] for q in self.quantiles], axis=1
            )
            stacked.sort(axis=1)
            for i, q in enumerate(self.quantiles):
                out[self._qname(q)] = stacked[:, i]
        return out

    def predict_one(self, features: HouseFeatures) -> dict[str, float]:
        result = self.predict(features.to_array().reshape(1, -1))
        return {k: float(v[0]) for k, v in result.items()}

    # ── Persistence ──────────────────────────────────────────────────────
    def save(self, path) -> None:
        joblib.dump(
            {
                "quantiles": list(self.quantiles),
                "gbr_kwargs": self._gbr_kwargs,
                "kwh_cap_per_hour": self.kwh_cap_per_hour,
                "feature_names": list(HOUSE_FEATURE_NAMES),
                "models": self._models,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "HouseForecaster":
        data = joblib.load(path)
        saved_features = tuple(data.get("feature_names", HOUSE_FEATURE_NAMES))
        if saved_features != HOUSE_FEATURE_NAMES:
            raise RuntimeError(
                "Saved HouseForecaster feature schema does not match the "
                "current HOUSE_FEATURE_NAMES — refusing to load.\n"
                f"  saved : {saved_features}\n"
                f"  current: {HOUSE_FEATURE_NAMES}"
            )
        m = cls(
            quantiles=tuple(data["quantiles"]),
            gbr_kwargs=data["gbr_kwargs"],
            kwh_cap_per_hour=data.get("kwh_cap_per_hour", HOUSE_KWH_PER_HOUR_CAP),
        )
        m._models = data["models"]
        return m
