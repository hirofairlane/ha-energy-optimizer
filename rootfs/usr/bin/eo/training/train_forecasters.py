"""Offline backfill + train SolarForecaster + HouseForecaster.

Reads InfluxDB through the monolith's existing helpers
(``ha_history_influx`` / ``cfg`` / ``_WIZARD``), aggregates per-hour observations,
derives atmospheric factor targets from a ``ClearSkyModel``, fits both the
atmospheric_factor and house quantile models, and persists the artefacts under
``/data/forecasters/``.

Designed to run inside the addon container:

    docker exec addon_<slug>_energy_optimizer \\
        python3 -m eo.training.train_forecasters --days 60

This is the implementation of `docs/v5-wiring.md` §2.1. It does NOT touch
``main()`` or the cycle entry point — flipping ``v5_engine_enabled`` still
needs §2.2 (adapter) and §2.3 (entry-point swap).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib

# Force the monolith's import path so we get its _WIZARD / cfg / ha_* helpers
sys.path.insert(0, "/usr/bin")
import energy_optimizer as mono  # noqa: E402
from eo.forecasters.atmospheric_factor import (  # noqa: E402
    AtmosphericFactorFeatures,
    make_features_for_hour,
)
from eo.forecasters.clear_sky import ClearSkyConfig, ClearSkyModel  # noqa: E402
from eo.forecasters.house_forecaster import (  # noqa: E402
    HouseFeatures,
    make_house_features_for_hour,
)
from eo.forecasters.training import (  # noqa: E402
    MIN_TRAIN_SAMPLES,
    compute_atmospheric_factor,
    train_atmospheric_factor_model,
    train_house_forecaster,
)

ARTEFACT_DIR = Path("/data/forecasters")
REPORT_FILE = ARTEFACT_DIR / "training_report.json"

log = logging.getLogger("eo.training")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ── Config resolution ──────────────────────────────────────────────────────
def _resolve_clear_sky_config() -> ClearSkyConfig:
    wiz = mono._WIZARD or {}
    loc = wiz.get("location") or {}
    lat = float(loc.get("latitude") or mono.HOME_LAT)
    lon = float(loc.get("longitude") or mono.HOME_LON)
    # Capacity / azimuth / tilt — wizard fields are optional. Sensible defaults
    # cover the most common Spanish rooftop install (south-facing, latitude tilt).
    capacity_kwp = float(mono.cfg("solar_capacity_kwp", wiz.get("solar_capacity_kwp", 5.0)))
    azimuth = float(mono.cfg("solar_azimuth_deg", wiz.get("solar_azimuth_deg", 180.0)))
    tilt = mono.cfg("solar_tilt_deg", wiz.get("solar_tilt_deg"))
    return ClearSkyConfig(
        latitude=lat,
        longitude=lon,
        capacity_kwp=capacity_kwp,
        tilt_deg=float(tilt) if tilt is not None else None,
        azimuth_deg=azimuth,
    )


# ── Per-hour aggregation ───────────────────────────────────────────────────
def _hourly_series_from_power_rows(rows: list[dict]) -> dict[datetime, float]:
    """Average power (W) per hour from a list of ``{last_changed, state}`` rows.

    Returns a mapping ``utc_hour_start → kWh`` (W avg × 1h / 1000).
    """
    bucket_sum: defaultdict[datetime, float] = defaultdict(float)
    bucket_n: defaultdict[datetime, int] = defaultdict(int)
    for row in rows:
        try:
            ts_iso = row["last_changed"]
            val = float(row["state"])
        except (KeyError, ValueError, TypeError):
            continue
        try:
            ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            continue
        h = ts.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        bucket_sum[h] += val
        bucket_n[h] += 1
    return {
        h: (bucket_sum[h] / bucket_n[h]) / 1000.0  # W avg → kWh per hour
        for h in bucket_sum
        if bucket_n[h] > 0
    }


def _hourly_avg_from_state_rows(rows: list[dict]) -> dict[datetime, float]:
    """Numeric average per hour (no unit conversion) — for temperature etc."""
    bucket_sum: defaultdict[datetime, float] = defaultdict(float)
    bucket_n: defaultdict[datetime, int] = defaultdict(int)
    for row in rows:
        try:
            val = float(row["state"])
            ts = datetime.fromisoformat(row["last_changed"].replace("Z", "+00:00"))
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        h = ts.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        bucket_sum[h] += val
        bucket_n[h] += 1
    return {h: bucket_sum[h] / bucket_n[h] for h in bucket_sum if bucket_n[h] > 0}


def _hourly_mode_from_state_rows(rows: list[dict]) -> dict[datetime, str]:
    """Modal string per hour — for weather condition."""
    bucket: defaultdict[datetime, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for row in rows:
        try:
            ts = datetime.fromisoformat(row["last_changed"].replace("Z", "+00:00"))
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        val = str(row.get("state", "unknown"))
        h = ts.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        bucket[h][val] += 1
    return {h: max(counts.items(), key=lambda kv: kv[1])[0] for h, counts in bucket.items()}


# ── Backfill ──────────────────────────────────────────────────────────────
def backfill(days: int) -> dict[str, dict]:
    """Pull every required series from InfluxDB and aggregate per UTC hour.

    Returns ``{kind: {hour_utc: value}}`` for: solar_kwh, house_kwh,
    t_outdoor, weather.
    """
    wiz = mono._WIZARD or {}

    def fetch(role: str, legacy: str = "", default: str = "") -> list[dict]:
        entity = mono._wiz(role, legacy, default)
        if not entity:
            log.warning(f"  Skipping {role!r}: no entity configured")
            return []
        rows, err = mono.ha_history_influx(entity, days=days)
        if err:
            log.warning(f"  {role} ({entity}): {err}")
        return rows

    solar_rows = fetch("solar_power", "sensor_solar_power", "")
    grid_rows = fetch("grid_power", "sensor_grid_power", "")
    battery_rows = fetch("battery_power", "sensor_battery_power", "")
    t_out_rows = fetch("temp_outdoor", "sensor_temp_outdoor", "")
    weather_rows = fetch("weather", "sensor_weather", "")

    solar = _hourly_series_from_power_rows(solar_rows)
    grid = _hourly_series_from_power_rows(grid_rows)
    battery = _hourly_series_from_power_rows(battery_rows)
    t_outdoor = _hourly_avg_from_state_rows(t_out_rows)
    weather = _hourly_mode_from_state_rows(weather_rows)

    grid_flip = bool(wiz.get("grid_flip"))
    if grid_flip:
        grid = {h: -v for h, v in grid.items()}

    # House consumption derived from the energy balance:
    #     house = solar + battery_discharge − grid_export
    # With the addon convention battery +ve = discharging, grid +ve = exporting,
    # this is just `solar + battery − grid`. Skip hours where any of the three
    # series is missing.
    house: dict[datetime, float] = {}
    for h in solar:
        if h in grid and h in battery:
            v = solar[h] + battery[h] - grid[h]
            if v >= 0:
                house[h] = v
    log.info(
        f"  Backfill: solar={len(solar)} grid={len(grid)} battery={len(battery)} "
        f"t_outdoor={len(t_outdoor)} weather={len(weather)} → house={len(house)}"
    )
    return {
        "solar": solar,
        "house": house,
        "t_outdoor": t_outdoor,
        "weather": weather,
    }


# ── Sample assembly ───────────────────────────────────────────────────────
def _atmospheric_samples(
    aggregates: dict[str, dict],
    clear_sky: ClearSkyModel,
) -> list[tuple[datetime, AtmosphericFactorFeatures, float]]:
    """Build (ts, features, target_atm_factor) triples for the atmospheric model."""
    solar = aggregates["solar"]
    t_outdoor = aggregates["t_outdoor"]
    weather = aggregates["weather"]
    samples: list[tuple[datetime, AtmosphericFactorFeatures, float]] = []
    for h, observed_kwh in solar.items():
        clear_kwh = clear_sky.kwh_for_hour(h)
        if clear_kwh <= 0.05:
            # Night-time or pathologically low — drop, would only inject noise.
            continue
        t_out = t_outdoor.get(h)
        if t_out is None:
            continue
        cond_now = weather.get(h)
        cond_lag = weather.get(h - timedelta(days=1))
        feat = make_features_for_hour(
            ts=h,
            weather_condition=cond_now,
            weather_condition_lag1d=cond_lag,
            t_outdoor=t_out,
            humidity=50.0,  # no humidity series wired today; safe prior
            yield_factor_similar_days=None,
        )
        target = compute_atmospheric_factor(observed_kwh, clear_kwh)
        samples.append((h, feat, target))
    return samples


def _house_samples(
    aggregates: dict[str, dict],
) -> list[tuple[datetime, HouseFeatures, float]]:
    """Build (ts, features, target_kwh) triples for the house model."""
    house = aggregates["house"]
    t_outdoor = aggregates["t_outdoor"]
    samples: list[tuple[datetime, HouseFeatures, float]] = []
    for h, observed_kwh in house.items():
        t_now = t_outdoor.get(h)
        t_lag = t_outdoor.get(h - timedelta(hours=24))
        kwh_yesterday = house.get(h - timedelta(hours=24))
        kwh_last = house.get(h - timedelta(hours=1))
        if any(v is None for v in (t_now, t_lag, kwh_yesterday, kwh_last)):
            continue
        feat = make_house_features_for_hour(
            target_ts=h,
            t_outdoor=t_now,
            t_outdoor_lag24h=t_lag,
            house_kwh_yesterday_same_hour=kwh_yesterday,
            house_kwh_last_observed=kwh_last,
            custom_loads_planned_watts=0.0,
            deferred_loads_planned_watts=0.0,
        )
        samples.append((h, feat, observed_kwh))
    return samples


# ── Entrypoint ────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=60, help="Backfill window in days")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the backfill + sample assembly but do not fit/save",
    )
    args = parser.parse_args(argv)

    mono._load_setup_cache()
    mono._load_wizard_cache()

    cs_cfg = _resolve_clear_sky_config()
    log.info(
        f"  Clear-sky config: lat={cs_cfg.latitude} lon={cs_cfg.longitude} "
        f"kwp={cs_cfg.capacity_kwp} tilt={cs_cfg.effective_tilt_deg()} "
        f"az={cs_cfg.azimuth_deg}"
    )
    cs_model = ClearSkyModel(cs_cfg)

    aggregates = backfill(args.days)

    atm_samples = _atmospheric_samples(aggregates, cs_model)
    house_samples = _house_samples(aggregates)
    log.info(f"  Atmospheric samples: {len(atm_samples)} (min {MIN_TRAIN_SAMPLES})")
    log.info(f"  House samples:       {len(house_samples)} (min {MIN_TRAIN_SAMPLES})")

    if args.dry_run:
        log.info("  --dry-run: skipping fit and save")
        return 0

    ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {"days": args.days, "trained_at": datetime.now(timezone.utc).isoformat()}

    try:
        atm_model, atm_report = train_atmospheric_factor_model(atm_samples)
        joblib.dump(atm_model, ARTEFACT_DIR / "atmospheric.joblib")
        reports["atmospheric"] = atm_report.to_dict()
        log.info(f"  ✓ Atmospheric model saved — {atm_report.to_dict()}")
    except Exception as e:
        log.error(f"  ✗ Atmospheric training failed: {e}")
        reports["atmospheric_error"] = str(e)

    try:
        house_model, house_report = train_house_forecaster(house_samples)
        joblib.dump(house_model, ARTEFACT_DIR / "house.joblib")
        reports["house"] = house_report.to_dict()
        log.info(f"  ✓ House model saved — {house_report.to_dict()}")
    except Exception as e:
        log.error(f"  ✗ House training failed: {e}")
        reports["house_error"] = str(e)

    REPORT_FILE.write_text(json.dumps(reports, indent=2))
    log.info(f"  Report written to {REPORT_FILE}")
    return 0 if "atmospheric_error" not in reports and "house_error" not in reports else 1


if __name__ == "__main__":
    sys.exit(main())
