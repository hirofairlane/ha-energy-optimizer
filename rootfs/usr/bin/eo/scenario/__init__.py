"""Scenario builder — the seam between forecasters and the planner.

The planner never consumes quantiles directly (SPEC §1.2 S1). The
ScenarioBuilder collapses the per-hour P10/P50/P90 quantiles of solar and
house forecasts into a coherent per-slot ``Scenario`` according to the
caller's risk preference, which in turn is mapped from the load's debt
state (SPEC §1.2 S4).

This module:
  * Defines :class:`Scenario`, :class:`RiskTolerance`, and the canonical
    debt→risk mapping (SPEC §1.2 S4).
  * Exposes :func:`build_scenario` which takes two per-hour quantile
    series and returns a Scenario aligned to 15-min slots.
"""

from eo.scenario.scenario_builder import (  # noqa: F401
    QuantileHourForecast,
    RiskTolerance,
    Scenario,
    build_scenario,
    risk_from_debt_state,
)
