"""Decision matrix — 11-row priority table (SPEC §1.4 P6).

A pure function ``decide_load_for_slot()`` takes the per-load context for a
specific slot and returns ``LoadDecision`` (ON / OFF + reason + utility_score
+ optional alert). The matrix mirrors the prose in the SPEC verbatim:

  1. debt_state == ok → OFF
  2. Solar surplus NOW + SOC OK → ON
  3. critical + period != peak → ON forced
  4. critical + period == peak → OFF + Telegram alert (unless allow_peak_on_critical)
  5. peak + debt_state != critical → OFF
  6. valley + prob_surplus_tomorrow ≥ required_confidence → OFF (defer)
  7. valley + low prob_surplus + high debt → ON full
  8. valley + low prob_surplus + medium debt → ON min_runtime
  9. valley + low prob_surplus + low debt → OFF
 10. mid + high prob_surplus_next_valley → OFF (wait)
 11. mid + no_valley_before_deadline + high debt → ON

Row 11 implicit: if no valley remains in the window AND debt is dangerously
high, mid beats peak. The caller computes ``no_valley_before_deadline``.

The function does NOT mutate state, does NOT call into the simulator, does
NOT touch policy. It is one piece of a planner iteration step.
"""

from __future__ import annotations

from dataclasses import dataclass

from eo.planner.load_quota import DebtState, LoadQuotaConfig, LoadQuotaState
from eo.planner.utility_score import utility_score


# ── Inputs and outputs ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class SlotContext:
    """Everything the matrix needs to decide one load × one slot."""
    period: str                       # "peak" / "mid" / "valley"
    solar_surplus_now: bool           # PV > house this slot AND SOC headroom
    prob_surplus_tomorrow: float      # [0, 1]
    prob_surplus_next_valley: float   # [0, 1] for mid-period look-ahead
    no_valley_before_deadline: bool   # window closes before any valley


@dataclass(frozen=True)
class LoadDecision:
    action: str            # "on" / "off"
    reason: str
    rule_id: int           # 1-11, matches the spec rows
    utility_score: int
    alert: bool = False    # True when row 4 fires (deadline missed in peak)
    min_runtime_only: bool = False  # True when row 8 fires (medium debt)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "rule_id": self.rule_id,
            "utility_score": self.utility_score,
            "alert": self.alert,
            "min_runtime_only": self.min_runtime_only,
        }


# ── Decision function ──────────────────────────────────────────────────────
def decide_load_for_slot(
    config: LoadQuotaConfig,
    quota_state: LoadQuotaState,
    slot: SlotContext,
) -> LoadDecision:
    score = utility_score(
        quota_state.debt_state,
        slot.period,
        prob_surplus_tomorrow=slot.prob_surplus_tomorrow,
    )

    debt = quota_state.debt_state

    # Row 1
    if debt == DebtState.OK:
        return LoadDecision(
            action="off", rule_id=1,
            reason="debt_state=ok — quota already met this window",
            utility_score=score,
        )

    # Row 2 — opportunistic surplus.
    # Free energy wins over any other consideration except the peak ban and
    # the critical-in-peak rule (which we evaluate next).
    if slot.solar_surplus_now and slot.period != "peak":
        return LoadDecision(
            action="on", rule_id=2,
            reason="solar surplus available now",
            utility_score=score,
        )

    # Row 4 — critical + peak: refuse, alert (unless allow_peak_on_critical).
    if debt in (DebtState.CRITICAL, DebtState.IRREACHABLE) and slot.period == "peak":
        if config.allow_peak_on_critical:
            return LoadDecision(
                action="on", rule_id=4,
                reason="critical debt + peak, allow_peak_on_critical=True",
                utility_score=score, alert=True,
            )
        return LoadDecision(
            action="off", rule_id=4,
            reason=(
                "critical debt but tariff is PEAK — refusing to run. "
                "Quota will not be met this window."
            ),
            utility_score=score, alert=True,
        )

    # Row 3 — critical + non-peak: force ON.
    if debt in (DebtState.CRITICAL, DebtState.IRREACHABLE) and slot.period != "peak":
        return LoadDecision(
            action="on", rule_id=3,
            reason=f"critical debt + period={slot.period} — forcing ON",
            utility_score=score,
        )

    # Row 5 — peak + non-critical: OFF.
    if slot.period == "peak":
        return LoadDecision(
            action="off", rule_id=5,
            reason="period=peak, debt not critical — deferred loads do not run in peak",
            utility_score=score,
        )

    # Valley logic — rows 6-9.
    if slot.period == "valley":
        confidence_threshold = config.required_confidence_pct / 100.0
        # Row 6 — high probability of surplus tomorrow → defer.
        if slot.prob_surplus_tomorrow >= confidence_threshold:
            return LoadDecision(
                action="off", rule_id=6,
                reason=(
                    f"valley + prob_surplus_tomorrow="
                    f"{slot.prob_surplus_tomorrow:.0%} ≥ "
                    f"{config.required_confidence_pct:.0f}% threshold → wait"
                ),
                utility_score=score,
            )
        # Rows 7/8/9 — low probability, decide by debt level.
        if debt == DebtState.HIGH:
            return LoadDecision(
                action="on", rule_id=7,
                reason="valley + low prob_surplus + high debt → ON full",
                utility_score=score,
            )
        if debt == DebtState.MEDIUM:
            return LoadDecision(
                action="on", rule_id=8,
                reason="valley + low prob_surplus + medium debt → ON min_runtime",
                utility_score=score, min_runtime_only=True,
            )
        # Row 9 — low debt: defer further.
        return LoadDecision(
            action="off", rule_id=9,
            reason="valley + low prob_surplus + low debt → defer",
            utility_score=score,
        )

    # Mid period — rows 10-11.
    if slot.period == "mid":
        # Row 10 — if the next valley is likely to bring surplus, wait.
        if slot.prob_surplus_next_valley >= (config.required_confidence_pct / 100.0):
            return LoadDecision(
                action="off", rule_id=10,
                reason=(
                    f"mid + prob_surplus_next_valley="
                    f"{slot.prob_surplus_next_valley:.0%} → wait for valley"
                ),
                utility_score=score,
            )
        # Row 11 — no valley remaining in the window AND debt high → ON.
        if slot.no_valley_before_deadline and debt == DebtState.HIGH:
            return LoadDecision(
                action="on", rule_id=11,
                reason="mid + no valley before deadline + high debt → ON",
                utility_score=score, min_runtime_only=True,
            )
        # Otherwise stay OFF in mid for non-critical loads.
        return LoadDecision(
            action="off", rule_id=10,
            reason="mid + insufficient urgency to run",
            utility_score=score,
        )

    # Unknown period — fail safe.
    return LoadDecision(
        action="off", rule_id=0,
        reason=f"unknown period={slot.period!r} — defaulting to OFF",
        utility_score=score,
    )
