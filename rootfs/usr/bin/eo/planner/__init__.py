"""Iterative planner — decides ON/OFF for every deferrable load each cycle.

Inputs:
    * scenario              (from ScenarioBuilder, hourly P-quantile forecasts
                             expanded into 15-min slots by the caller)
    * current SOC           (from sensors / SystemState)
    * load_quotas           (per-load LoadQuotaConfig + observed history)
    * forced_states         (slot-0 policy-driven masks — SPEC §1.4 P5)
    * available_loads       (which loads can be considered this cycle)
    * tariff                (period for each slot)

Outputs:
    * raw_plan              (decisions per load per slot, before policy)
    * convergence trace     (hash history + iteration count)

Submodules:
    load_quota.py     — DebtState computation + data-quality scaling.
    utility_score.py  — per-load score that ranks loads for capacity budget.
    decision_matrix.py — 11-row priority table from SPEC §1.4 P6.
    iterative.py      — convergence loop with forced_states + max_iter + 2-cycle.
"""

from eo.planner.load_quota import (  # noqa: F401
    DebtState,
    LoadQuotaConfig,
    LoadQuotaState,
    compute_debt_state,
)
