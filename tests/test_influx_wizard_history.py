"""Tests for _influx_wizard_history epoch→ISO timestamp conversion.

Regression test for the bug where `_influx_query` returns timestamps as
millisecond integers (due to `epoch=ms` param) but `_influx_wizard_history`
passes them raw to `_rows_to_15min_series`, which expects ISO 8601 strings.

The legacy path (`ha_history_influx`) correctly converts:
    ms = int(point[time_i])
    ts = datetime.fromtimestamp(ms / 1000, tz=utc).isoformat()

The wizard path was missing this conversion.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest


# The monolith can't be imported directly (too many side effects at module
# level: Flask app, HA connection, etc.). We extract and test the core logic
# that _influx_wizard_history + _rows_to_15min_series implement.


def _influx_wizard_history_BROKEN(entity: str, days: int, influx_cfg: dict, mock_response: dict) -> list:
    """Reproduces the broken code path (pre-fix)."""
    rows = []
    for series in mock_response.get("results", [{}])[0].get("series", []):
        cols = series.get("columns", [])
        try:
            ti = cols.index("time")
            vi = cols.index("value")
        except ValueError:
            continue
        for point in series.get("values", []):
            ts, val = point[ti], point[vi]
            if val is None:
                continue
            # BUG: stores raw epoch-ms integer as last_changed
            rows.append({"last_changed": ts, "state": str(val)})
    return rows


def _influx_wizard_history_FIXED(entity: str, days: int, influx_cfg: dict, mock_response: dict) -> list:
    """The fixed code path — converts epoch-ms to ISO string."""
    rows = []
    for series in mock_response.get("results", [{}])[0].get("series", []):
        cols = series.get("columns", [])
        try:
            ti = cols.index("time")
            vi = cols.index("value")
        except ValueError:
            continue
        for point in series.get("values", []):
            ts, val = point[ti], point[vi]
            if val is None:
                continue
            # FIX: convert epoch-ms to ISO 8601 string
            if isinstance(ts, (int, float)):
                ts = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            rows.append({"last_changed": ts, "state": str(val)})
    return rows


def _rows_to_15min_series(rows: list, col: str):
    """Minimal reproduction of the downstream consumer (requires ISO strings)."""
    import pandas as pd

    records = []
    for row in rows:
        ts = datetime.fromisoformat(row["last_changed"].replace("Z", "+00:00"))
        val = float(row["state"])
        records.append({"ts": ts.replace(tzinfo=None), "value": val})
    if not records:
        return pd.Series(dtype=float, name=col)
    s = pd.DataFrame(records).set_index("ts")["value"]
    return s.resample("15min").mean().ffill().rename(col)


# ── Test fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def influx_response_epoch_ms():
    """Simulates InfluxDB response with epoch=ms timestamps (integers)."""
    base_ms = int(datetime(2026, 5, 26, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
    return {
        "results": [{
            "series": [{
                "name": "%",
                "columns": ["time", "value"],
                "values": [
                    [base_ms + i * 900_000, 50.0 + i * 0.5]  # every 15 min
                    for i in range(96)  # 24 hours
                ],
            }]
        }]
    }


@pytest.fixture
def influx_cfg():
    return {"host": "192.168.10.131", "port": 8086, "db": "homeassistant",
            "username": "user", "password": "pass"}


# ── Tests ────────────────────────────────────────────────────────────────────

class TestBrokenPath:
    """Demonstrates the bug: epoch-ms integers crash _rows_to_15min_series."""

    def test_broken_returns_integer_timestamps(self, influx_response_epoch_ms, influx_cfg):
        rows = _influx_wizard_history_BROKEN(
            "sensor.battery_state_of_charge", 90, influx_cfg, influx_response_epoch_ms
        )
        assert len(rows) == 96
        # last_changed is an integer, not a string
        assert isinstance(rows[0]["last_changed"], int)

    def test_broken_crashes_rows_to_15min_series(self, influx_response_epoch_ms, influx_cfg):
        rows = _influx_wizard_history_BROKEN(
            "sensor.battery_state_of_charge", 90, influx_cfg, influx_response_epoch_ms
        )
        with pytest.raises(AttributeError, match="'int' object has no attribute 'replace'"):
            _rows_to_15min_series(rows, "value")


class TestFixedPath:
    """Confirms the fix: epoch-ms integers are converted to ISO strings."""

    def test_fixed_returns_iso_timestamps(self, influx_response_epoch_ms, influx_cfg):
        rows = _influx_wizard_history_FIXED(
            "sensor.battery_state_of_charge", 90, influx_cfg, influx_response_epoch_ms
        )
        assert len(rows) == 96
        # last_changed is now an ISO string
        assert isinstance(rows[0]["last_changed"], str)
        # Parseable as ISO datetime
        dt = datetime.fromisoformat(rows[0]["last_changed"])
        assert dt.year == 2026

    def test_fixed_works_with_rows_to_15min_series(self, influx_response_epoch_ms, influx_cfg):
        rows = _influx_wizard_history_FIXED(
            "sensor.battery_state_of_charge", 90, influx_cfg, influx_response_epoch_ms
        )
        series = _rows_to_15min_series(rows, "value")
        assert len(series) >= 90  # 96 points, some may merge in resampling
        assert series.iloc[0] == pytest.approx(50.0)

    def test_fixed_handles_none_values(self, influx_cfg):
        """None values in the series are skipped (not converted)."""
        response = {
            "results": [{"series": [{"columns": ["time", "value"],
                                     "values": [[1716714000000, None],
                                                [1716714900000, 42.0]]}]}]
        }
        rows = _influx_wizard_history_FIXED("sensor.x", 1, influx_cfg, response)
        assert len(rows) == 1
        assert rows[0]["state"] == "42.0"

    def test_fixed_handles_string_timestamps_passthrough(self, influx_cfg):
        """If InfluxDB ever returns RFC3339 strings (no epoch param), still works."""
        response = {
            "results": [{"series": [{"columns": ["time", "value"],
                                     "values": [["2026-05-26T10:00:00Z", 55.0]]}]}]
        }
        rows = _influx_wizard_history_FIXED("sensor.x", 1, influx_cfg, response)
        assert len(rows) == 1
        assert rows[0]["last_changed"] == "2026-05-26T10:00:00Z"


class TestDataQualityFirstTs:
    """Bonus bug: first_ts detection also assumes string timestamps."""

    def test_first_ts_slice_fails_on_integer(self):
        """The data-quality endpoint does `values[0][0][:10]` — fails on int."""
        epoch_ms = 1716714000000
        with pytest.raises(TypeError):
            _ = epoch_ms[:10]  # type: ignore[index]

    def test_first_ts_works_after_conversion(self):
        epoch_ms = 1716714000000
        iso = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat()
        date_str = iso[:10]
        assert date_str == "2024-05-26"
