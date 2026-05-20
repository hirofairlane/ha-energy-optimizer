"""Execution-layer dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class CommandRequest:
    """One HA service-call to emit, derived from a policy-adjusted decision."""
    load: str
    target_action: str        # "on" or "off"
    domain: str               # e.g. "switch"
    service: str              # e.g. "turn_on" / "turn_off"
    entity_id: str
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one CommandRequest."""
    request: CommandRequest
    issued: bool              # we tried to send it
    acknowledged: bool        # HA returned success
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "load": self.request.load,
            "target_action": self.request.target_action,
            "entity_id": self.request.entity_id,
            "domain": self.request.domain,
            "service": self.request.service,
            "issued": self.issued,
            "acknowledged": self.acknowledged,
            "error": self.error,
        }


@dataclass(frozen=True)
class ExecutionPlan:
    """The list of commands we intend to issue this cycle."""
    cycle_ts: datetime
    commands: tuple[CommandRequest, ...]


@dataclass(frozen=True)
class ExecutionResult:
    plan: ExecutionPlan
    results: tuple[CommandResult, ...]

    @property
    def all_acknowledged(self) -> bool:
        return all(r.acknowledged for r in self.results) if self.results else True

    @property
    def loads_now_on(self) -> tuple[str, ...]:
        """Loads we believe are ON after this execution."""
        return tuple(sorted({
            r.request.load
            for r in self.results
            if r.request.target_action == "on" and r.acknowledged
        }))

    def to_dict(self) -> dict:
        return {
            "cycle_ts": self.plan.cycle_ts.isoformat(),
            "commands": [r.to_dict() for r in self.results],
            "all_acknowledged": self.all_acknowledged,
            "loads_now_on": list(self.loads_now_on),
        }
