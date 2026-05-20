"""Setup integrity checker.

Detects misconfiguration in ``wizard_config.json`` where the same ``entity_id``
is assigned to multiple controlling roles. The original motivating bug
(reported by issue #4) was a single ``switch.*`` configured simultaneously as
``sensors.pool_switch`` and as a ``custom_loads[].switch``. Under that setup,
the pool branch issued ``turn_on`` while the custom-load branch issued
``turn_off`` in the very same decision cycle — making the switch flap and
producing a "black magic" feel for the end user.

This module is the first occupant of the new modular package ``eo`` and is the
template for how subsequent v5.0.0 modules will be organised: pure, importable,
testable in isolation, with no side effects beyond what the caller wires up.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable


# ── Role taxonomy ────────────────────────────────────────────────────────────
# "actuable" = the engine writes commands to this entity (turn_on, set_value,
# climate.set_temperature, etc). A duplicate here is what flaps switches.
# "sensor"   = the engine only reads from this entity. A duplicate is suspicious
# but not actively harmful — usually a config copy-paste error.
ACTUABLE_SENSOR_ROLES: frozenset[str] = frozenset({
    "pool_switch",
    "dishwasher_switch",
})


@dataclass(frozen=True)
class RoleAssignment:
    """One occurrence of an entity_id used in a role."""
    role: str             # e.g. "sensors.pool_switch" or "custom_loads[Boiler].switch"
    category: str         # "actuable" or "sensor"


@dataclass
class Conflict:
    entity_id: str
    occurrences: list[RoleAssignment]
    severity: str         # "critical" or "warning"

    @property
    def actuable_count(self) -> int:
        return sum(1 for o in self.occurrences if o.category == "actuable")

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "severity": self.severity,
            "actuable_count": self.actuable_count,
            "occurrences": [asdict(o) for o in self.occurrences],
            "roles": [o.role for o in self.occurrences],
        }


@dataclass
class IntegrityReport:
    ok: bool
    conflicts: list[Conflict]

    @property
    def critical_conflicts(self) -> list[Conflict]:
        return [c for c in self.conflicts if c.severity == "critical"]

    @property
    def warnings(self) -> list[Conflict]:
        return [c for c in self.conflicts if c.severity == "warning"]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "critical_count": len(self.critical_conflicts),
            "warning_count": len(self.warnings),
            "conflicts": [c.to_dict() for c in self.conflicts],
        }

    def to_summary(self) -> str:
        if self.ok and not self.conflicts:
            return "Setup integrity: OK (no entity_id collisions)"
        lines: list[str] = []
        if self.critical_conflicts:
            lines.append(f"Setup integrity: {len(self.critical_conflicts)} CRITICAL conflict(s) — these will cause switches to flap:")
            for c in self.critical_conflicts:
                lines.append(f"  • {c.entity_id} is assigned to {len(c.occurrences)} roles: {', '.join(o.role for o in c.occurrences)}")
        if self.warnings:
            lines.append(f"Setup integrity: {len(self.warnings)} warning(s) — same entity_id used in multiple non-actuable roles:")
            for c in self.warnings:
                lines.append(f"  • {c.entity_id} ↔ {', '.join(o.role for o in c.occurrences)}")
        return "\n".join(lines)


# ── Entity collection ────────────────────────────────────────────────────────
def _collect_role_assignments(wizard: dict) -> dict[str, list[RoleAssignment]]:
    """Walk the wizard config and produce {entity_id: [RoleAssignment, ...]}."""
    occurrences: dict[str, list[RoleAssignment]] = {}

    def _add(entity_id: str | None, role: str, category: str) -> None:
        if not entity_id or not isinstance(entity_id, str):
            return
        occurrences.setdefault(entity_id, []).append(
            RoleAssignment(role=role, category=category)
        )

    sensors = wizard.get("sensors") or {}
    if isinstance(sensors, dict):
        for role, eid in sensors.items():
            category = "actuable" if role in ACTUABLE_SENSOR_ROLES else "sensor"
            _add(eid, f"sensors.{role}", category)

    for i, cl in enumerate(wizard.get("custom_loads") or []):
        if not isinstance(cl, dict):
            continue
        name = cl.get("name") or f"#{i}"
        _add(cl.get("switch"), f"custom_loads[{name}].switch", "actuable")

    for i, zone in enumerate(wizard.get("hvac_zones") or []):
        if not isinstance(zone, dict):
            continue
        name = zone.get("name") or f"zone{i}"
        _add(zone.get("climate"),     f"hvac_zones[{name}].climate",     "actuable")
        _add(zone.get("temp_heat"),   f"hvac_zones[{name}].temp_heat",   "actuable")
        _add(zone.get("temp_cool"),   f"hvac_zones[{name}].temp_cool",   "actuable")
        _add(zone.get("temp_sensor"), f"hvac_zones[{name}].temp_sensor", "sensor")

    for i, sm in enumerate(wizard.get("grid_submeters") or []):
        if not isinstance(sm, dict):
            continue
        name = sm.get("name") or f"submeter{i}"
        _add(sm.get("entity"), f"grid_submeters[{name}].entity", "sensor")

    return occurrences


# ── Severity classification ──────────────────────────────────────────────────
def _classify(occurrences: Iterable[RoleAssignment]) -> str:
    """Two-or-more actuable assignments → critical (causes flapping).
    One actuable + N sensors → critical too (writes to a sensed entity is
    almost always a config mistake and can produce phantom readings).
    Sensor-only duplicates → warning (cosmetic, no runtime damage).
    """
    occ_list = list(occurrences)
    actuable_count = sum(1 for o in occ_list if o.category == "actuable")
    if actuable_count >= 2:
        return "critical"
    if actuable_count == 1 and len(occ_list) > 1:
        return "critical"
    return "warning"


# ── Public API ───────────────────────────────────────────────────────────────
def check(wizard_config: dict) -> IntegrityReport:
    """Run the integrity check on a wizard_config dict.

    The function is pure: it does not read files, talk to HA, or log. The
    caller wires up logging, HA notifications, etc. based on the returned
    report. That separation keeps it trivially unit-testable.
    """
    if not isinstance(wizard_config, dict):
        return IntegrityReport(ok=True, conflicts=[])

    role_map = _collect_role_assignments(wizard_config)

    conflicts: list[Conflict] = []
    for eid, occurrences in role_map.items():
        if len(occurrences) <= 1:
            continue
        severity = _classify(occurrences)
        conflicts.append(
            Conflict(entity_id=eid, occurrences=list(occurrences), severity=severity)
        )

    conflicts.sort(key=lambda c: (0 if c.severity == "critical" else 1, c.entity_id))
    ok = not any(c.severity == "critical" for c in conflicts)
    return IntegrityReport(ok=ok, conflicts=conflicts)
