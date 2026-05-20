"""Forecasting layer.

Components:
    ClearSkyModel          — deterministic geometry-based PV baseline (no ML).
    AtmosphericFactorModel — ML residual that scales the clear-sky output.
    SolarForecaster        — combines the two with quantile heads (P10/P50/P90).
    HouseForecaster        — standalone load forecaster with quantile heads.

All forecasters are designed to be importable in isolation so they can be
unit-tested without bringing up the full add-on runtime.
"""
