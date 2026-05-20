"""Aggregated SystemState dataclass.

Construction model: ``SystemState`` is immutable per cycle. At the start of
the cycle the orchestrator builds one from sensors + persisted state +
forecast_quality.json; at the end it builds a fresh one for persistence.

Persistence: JSON-backed with atomic write (tmp + fsync + os.replace),
matching SPEC §1.6 E1 and the ForecastQualityTracker pattern. The
write/read are pure helpers — they take a Path and a SystemState, with no
hidden global state.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from eo.planner.load_quota import DebtState
from eo.policy.antiflap import AntiflapState


@dataclass(frozen=True)
class BatteryState:
    soc_pct: float
    power_w: float                        # +ve charge, -ve discharge
    last_updated: datetime | None

    def to_dict(self) -> dict:
        return {
            "soc_pct": self.soc_pct,
            "power_w": self.power_w,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatteryState":
        ts = data.get("last_updated")
        return cls(
            soc_pct=float(data["soc_pct"]),
            power_w=float(data["power_w"]),
            last_updated=datetime.fromisoformat(ts) if ts else None,
        )


@dataclass(frozen=True)
class ExecutionWorldState:
    """Last confirmed world-state read after ACK. Per-load on/off + ts."""
    loads_on: tuple[str, ...]                       # names of loads believed ON
    last_reconciled_at: datetime | None

    def is_on(self, load: str) -> bool:
        return load in self.loads_on

    def to_dict(self) -> dict:
        return {
            "loads_on": list(self.loads_on),
            "last_reconciled_at": (
                self.last_reconciled_at.isoformat()
                if self.last_reconciled_at else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionWorldState":
        ts = data.get("last_reconciled_at")
        return cls(
            loads_on=tuple(data.get("loads_on") or []),
            last_reconciled_at=datetime.fromisoformat(ts) if ts else None,
        )


@dataclass(frozen=True)
class SystemState:
    """All state the cycle needs, in one frozen dataclass."""
    cycle_ts: datetime                                          # tz-aware
    battery: BatteryState
    forecast_quality: dict                                      # series → ForecastQualityStats.to_dict()
    planner_hash_history: tuple[str, ...]                       # last N hashes
    load_debt: dict[str, str]                                   # load_name → DebtState.value
    antiflap_state: dict[str, AntiflapState]                    # load_name → AntiflapState
    execution_world_state: ExecutionWorldState

    # ── Validation ────────────────────────────────────────────────────────
    def __post_init__(self):
        if self.cycle_ts.tzinfo is None:
            raise ValueError("cycle_ts must be tz-aware")

    # ── Replace helpers ──────────────────────────────────────────────────
    def with_battery(self, battery: BatteryState) -> "SystemState":
        return replace(self, battery=battery)

    def with_world_state(self, world: ExecutionWorldState) -> "SystemState":
        return replace(self, execution_world_state=world)

    def with_planner_history(self, hashes: tuple[str, ...]) -> "SystemState":
        return replace(self, planner_hash_history=hashes)

    # ── Serialisation ────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "cycle_ts": self.cycle_ts.isoformat(),
            "battery": self.battery.to_dict(),
            "forecast_quality": dict(self.forecast_quality),
            "planner_hash_history": list(self.planner_hash_history),
            "load_debt": dict(self.load_debt),
            "antiflap_state": {
                load: {
                    "load": s.load,
                    "last_action": s.last_action,
                    "last_change_ts": s.last_change_ts.isoformat(),
                }
                for load, s in self.antiflap_state.items()
            },
            "execution_world_state": self.execution_world_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SystemState":
        cycle_ts = datetime.fromisoformat(data["cycle_ts"])
        if cycle_ts.tzinfo is None:
            cycle_ts = cycle_ts.replace(tzinfo=timezone.utc)
        antiflap = {}
        for load, raw in (data.get("antiflap_state") or {}).items():
            ts = datetime.fromisoformat(raw["last_change_ts"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            antiflap[load] = AntiflapState(
                load=raw["load"],
                last_action=raw["last_action"],
                last_change_ts=ts,
            )
        return cls(
            cycle_ts=cycle_ts,
            battery=BatteryState.from_dict(data["battery"]),
            forecast_quality=dict(data.get("forecast_quality") or {}),
            planner_hash_history=tuple(data.get("planner_hash_history") or []),
            load_debt=dict(data.get("load_debt") or {}),
            antiflap_state=antiflap,
            execution_world_state=ExecutionWorldState.from_dict(
                data.get("execution_world_state") or {"loads_on": [], "last_reconciled_at": None}
            ),
        )

    # ── Persistence ──────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Atomic write — SPEC §1.6 E1."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as f:
            json.dump(self.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
            tmp = f.name
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "SystemState | None":
        path = Path(path)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
