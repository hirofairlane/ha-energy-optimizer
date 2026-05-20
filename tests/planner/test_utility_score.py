"""Tests for eo.planner.utility_score."""

from __future__ import annotations

from eo.planner.load_quota import DebtState
from eo.planner.utility_score import (
    DEBT_WEIGHT,
    PERIOD_WEIGHT_FOR_RUNNING,
    utility_score,
)


class TestMonotonicity:
    def test_higher_debt_higher_score(self):
        assert (
            utility_score(DebtState.OK, "valley")
            < utility_score(DebtState.LOW, "valley")
            < utility_score(DebtState.MEDIUM, "valley")
            < utility_score(DebtState.HIGH, "valley")
            < utility_score(DebtState.CRITICAL, "valley")
        )

    def test_valley_beats_peak_for_same_debt(self):
        assert (
            utility_score(DebtState.HIGH, "valley")
            > utility_score(DebtState.HIGH, "peak")
        )

    def test_higher_prob_surplus_lowers_score_for_non_critical(self):
        s_low_prob = utility_score(DebtState.HIGH, "valley", prob_surplus_tomorrow=0.0)
        s_high_prob = utility_score(DebtState.HIGH, "valley", prob_surplus_tomorrow=1.0)
        assert s_low_prob > s_high_prob

    def test_critical_score_does_not_drop_with_high_prob_surplus(self):
        s_low_prob = utility_score(DebtState.CRITICAL, "valley", prob_surplus_tomorrow=0.0)
        s_high_prob = utility_score(DebtState.CRITICAL, "valley", prob_surplus_tomorrow=1.0)
        assert s_high_prob == s_low_prob

    def test_irreachable_score_does_not_drop_with_high_prob_surplus(self):
        s_low = utility_score(DebtState.IRREACHABLE, "valley", prob_surplus_tomorrow=0.0)
        s_high = utility_score(DebtState.IRREACHABLE, "valley", prob_surplus_tomorrow=1.0)
        assert s_high == s_low


class TestBounds:
    def test_ok_in_peak_is_lowest(self):
        s = utility_score(DebtState.OK, "peak")
        assert s == PERIOD_WEIGHT_FOR_RUNNING["peak"]  # debt weight is 0

    def test_critical_in_valley_is_highest(self):
        s_crit = utility_score(DebtState.CRITICAL, "valley")
        assert s_crit == DEBT_WEIGHT[DebtState.CRITICAL] + PERIOD_WEIGHT_FOR_RUNNING["valley"]


class TestCustomWeights:
    def test_custom_weights_override(self):
        custom_debt = {DebtState.HIGH: 1000}
        s = utility_score(DebtState.HIGH, "valley", debt_weight=custom_debt)
        assert s >= 1000

    def test_unknown_period_defaults_to_zero(self):
        s = utility_score(DebtState.HIGH, "rare-period")
        assert s == DEBT_WEIGHT[DebtState.HIGH]
