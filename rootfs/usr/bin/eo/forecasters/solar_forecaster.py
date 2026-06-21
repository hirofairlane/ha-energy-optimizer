"""Solar production forecaster — clear-sky baseline × ML residual.

Composes:
    ClearSkyModel          → deterministic theoretical kWh per hour
    AtmosphericFactorModel → quantile residual factor in [0, 1.1]

into a single ``predict_hourly`` call that returns hourly P10/P50/P90 kWh
forecasts over a configurable horizon (default 48h per SPEC §1.1 F3).

Design choices:
    * The forecaster does **not** know how to fetch atmospheric features
      (weather conditions, temperature, humidity, prior yield history). The
      caller is responsible for assembling per-hour feature vectors.
    * Clipping rule: ``predicted_kwh = clear_sky_kwh × factor``. If
      ``clear_sky_kwh == 0`` (night), all quantiles collapse to 0 regardless
      of what the ML model says. This is a hard physical constraint and the
      ML residual cannot override it.
    * Quantile ordering is guaranteed (P10 ≤ P50 ≤ P90 per slot) because the
      underlying ``AtmosphericFactorModel.predict()`` enforces it.

The forecaster is a pure stateless composition object: ``predict_hourly()``
has no side effects beyond what the inner models do.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Sequence

import numpy as np

from eo.forecasters.atmospheric_factor import (
    AtmosphericFactorFeatures,
    AtmosphericFactorModel,
)
from eo.forecasters.clear_sky import ClearSkyModel

# Type alias: a function the caller supplies that, given a target hour-start
# timestamp, returns the feature vector to feed to the AtmosphericFactorModel.
# This is the seam where the addon plugs in its weather/history data sources.
FeatureProvider = Callable[[datetime], AtmosphericFactorFeatures]


@dataclass(frozen=True)
class HourForecast:
    """Per-hour quantile forecast."""
    hour_start: datetime
    clear_sky_kwh: float
    p10_kwh: float
    p50_kwh: float
    p90_kwh: float
    factor_p10: float
    factor_p50: float
    factor_p90: float

    def to_dict(self) -> dict:
        return {
            "hour_start": self.hour_start.isoformat(),
            "clear_sky_kwh": self.clear_sky_kwh,
            "p10_kwh": self.p10_kwh,
            "p50_kwh": self.p50_kwh,
            "p90_kwh": self.p90_kwh,
            "factor_p10": self.factor_p10,
            "factor_p50": self.factor_p50,
            "factor_p90": self.factor_p90,
        }


@dataclass
class SolarForecaster:
    clear_sky: ClearSkyModel
    atmospheric: AtmosphericFactorModel

    # ── Prediction ────────────────────────────────────────────────────────
    def predict_hourly(
        self,
        start: datetime,
        horizon_hours: int,
        feature_provider: FeatureProvider,
    ) -> list[HourForecast]:
        """Forecast hourly kWh from ``start`` for ``horizon_hours`` hours."""
        if start.tzinfo is None:
            raise ValueError("predict_hourly requires a timezone-aware start")
        if horizon_hours < 1:
            raise ValueError("horizon_hours must be ≥ 1")
        if not self.atmospheric.fitted:
            raise RuntimeError(
                "AtmosphericFactorModel is not fitted; train it before "
                "calling SolarForecaster.predict_hourly"
            )

        start_aligned = start.replace(minute=0, second=0, microsecond=0)

        # Build the per-hour feature matrix and clear-sky baselines in one pass.
        timestamps: list[datetime] = []
        feature_rows: list[np.ndarray] = []
        clear_kwh: list[float] = []

        for h in range(horizon_hours):
            t = start_aligned + timedelta(hours=h)
            timestamps.append(t)
            clear_kwh.append(self.clear_sky.kwh_for_hour(t))
            feature_rows.append(feature_provider(t).to_array())

        X = np.vstack(feature_rows)
        quantile_preds = self.atmospheric.predict(X)

        # quantile_preds keys are 'p10', 'p50', 'p90' (or whatever quantiles)
        # — we need exactly those three for v5.0.0 (SPEC §1.2 S4).
        for needed in ("p10", "p50", "p90"):
            if needed not in quantile_preds:
                raise RuntimeError(
                    "AtmosphericFactorModel did not produce {needed}; "
                    "ensure the model was fitted with quantiles (0.1, 0.5, 0.9)"
                    .format(needed=needed)
                )

        factors_p10 = quantile_preds["p10"]
        factors_p50 = quantile_preds["p50"]
        factors_p90 = quantile_preds["p90"]

        forecasts: list[HourForecast] = []
        for i, t in enumerate(timestamps):
            cs = clear_kwh[i]
            # Hard physical constraint: at night the ML residual cannot
            # invent production.
            if cs <= 0.0:
                p10 = p50 = p90 = 0.0
            else:
                p10 = float(cs * factors_p10[i])
                p50 = float(cs * factors_p50[i])
                p90 = float(cs * factors_p90[i])
            forecasts.append(HourForecast(
                hour_start=t,
                clear_sky_kwh=float(cs),
                p10_kwh=p10,
                p50_kwh=p50,
                p90_kwh=p90,
                factor_p10=float(factors_p10[i]),
                factor_p50=float(factors_p50[i]),
                factor_p90=float(factors_p90[i]),
            ))
        return forecasts

    # ── Diagnostic helpers ────────────────────────────────────────────────
    def total_kwh(self, forecasts: Sequence[HourForecast]) -> dict[str, float]:
        """Aggregate per-quantile totals over a forecast series."""
        return {
            "p10_kwh": sum(f.p10_kwh for f in forecasts),
            "p50_kwh": sum(f.p50_kwh for f in forecasts),
            "p90_kwh": sum(f.p90_kwh for f in forecasts),
            "clear_sky_kwh": sum(f.clear_sky_kwh for f in forecasts),
        }
