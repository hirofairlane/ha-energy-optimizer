"""Policy layer.

Takes the planner's ``raw_plan`` and applies operational constraints:
    capacity_budget    — greedy power allocation when loads would saturate
                         the inverter (SPEC §1.5 POL3, Gemini R2 §4.1).
    peak_prohibition   — defence-in-depth: cargas diferidas never run in
                         peak unless allow_peak_on_critical is honoured
                         upstream (SPEC §1.5 POL4).
    antiflap           — Schmitt trigger: blocks ON↔OFF changes that happen
                         within decision_hold_minutes of the last change
                         (SPEC §1.5 POL5).
    degraded_mode      — three-level operational degradation when forecast
                         quality, AEMET, or sensors fail (SPEC §1.5 POL6).

All layers are pure transforms (Plan, state, params) → (Plan, overrides).
No mutation in-place; the new Plan is freshly constructed (SPEC §1.5 POL2,
ChatGPT R1 §3 raw_plan ≠ policy_adjusted_plan).
"""

from eo.policy.types import (  # noqa: F401
    PolicyOverride,
    PolicyPipelineResult,
)
