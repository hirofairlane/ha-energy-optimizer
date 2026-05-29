"""Offline / batch training entry points.

Lives alongside the pure forecasters in :mod:`eo.forecasters` so the data-fetch
side (InfluxDB, HA recorder) is concentrated in one place and the rest of the
forecaster package stays unit-testable in isolation.
"""
