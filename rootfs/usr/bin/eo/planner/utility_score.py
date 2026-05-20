"""Per-load utility score.

A simple ordinal scoring function used to:
  * Rank competing loads in the greedy capacity budget (SPEC §1.5 POL3).
  * Tie-break the decision matrix when multiple rows would fire.
  * Future-proof for the global utility function in v5.1+ / v6.0
    (SPEC §7 — Gemini R3 §4.3 confirmed structure ahead of execution).

In v5.0.0 the score is a small integer weighted sum of three signals:
    * debt_state component (higher debt → higher score)
    * tariff period component (avoiding peak is more valuable)
    * confidence component (lower forecast confidence depresses score)

The function is deliberately monotone and explainable. The full utility
function (cost + comfort + wear + risk + switching) is a v5.1+ topic.
"""

from __future__ import annotations

from eo.planner.load_quota import DebtState


# Debt component (relative priority).
DEBT_WEIGHT: dict[DebtState, int] = {
    DebtState.OK: 0,
    DebtState.LOW: 10,
    DebtState.MEDIUM: 25,
    DebtState.HIGH: 50,
    DebtState.CRITICAL: 80,
    DebtState.IRREACHABLE: 75,   # below critical: it's a lost cause
}


# Period component (preference vs. each period).
# ON in peak is very costly; ON in valley is cheap.
PERIOD_WEIGHT_FOR_RUNNING: dict[str, int] = {
    "peak":   -40,
    "mid":     0,
    "valley":  20,
}


def utility_score(
    debt_state: DebtState,
    period: str,
    prob_surplus_tomorrow: float | None = None,
    *,
    debt_weight: dict[DebtState, int] | None = None,
    period_weight: dict[str, int] | None = None,
) -> int:
    """Return an ordinal utility score for running this load in this slot.

    The score is in roughly the range [-50, 100]. Higher = more urgent /
    more profitable to run.

    Parameters
    ----------
    debt_state
        Current debt classification from compute_debt_state().
    period
        Tariff period at the slot timestamp: "peak" / "mid" / "valley".
    prob_surplus_tomorrow
        If provided, a [0, 1] probability that tomorrow will hit surplus.
        High probability lowers the urgency of running today.
    """
    dw = debt_weight or DEBT_WEIGHT
    pw = period_weight or PERIOD_WEIGHT_FOR_RUNNING
    score = dw.get(debt_state, 0) + pw.get(period, 0)
    if prob_surplus_tomorrow is not None:
        # Strong tomorrow-surplus suggests we can defer → lower score.
        # We do not penalise CRITICAL or IRREACHABLE because deferral is
        # not an option for them.
        if debt_state not in (DebtState.CRITICAL, DebtState.IRREACHABLE):
            score -= int(round(15 * max(0.0, min(1.0, prob_surplus_tomorrow))))
    return score
