"""Deterministic clear-sky PV baseline.

Computes the theoretical PV output an installation would produce on a perfectly
clear day, given only its geometry (latitude, longitude, panel tilt, panel
azimuth, rated capacity). Atmospheric effects (clouds, haze, aerosols) are
deliberately left out — they are the job of :class:`AtmosphericFactorModel`,
a separate ML residual that multiplies this baseline by a factor in [0, 1.1].

The aim here is a baseline accurate enough that the ML residual only has to
learn the atmospheric correction, not also the orbital geometry of the Earth.

Algorithms used:
    * Solar declination & equation of time — Spencer (1971) trigonometric series.
    * Solar elevation & azimuth            — standard spherical-trig formulas.
    * Direct beam attenuation              — single-coefficient Beer-Lambert via Kasten-Young air mass.
    * Tilted-plane irradiance              — incidence angle on a fixed plane + simple isotropic-sky diffuse.

Avoids pulling in ``pvlib`` to keep the Docker image lean. Precision is
sufficient for v5.0.0 (a clear-day curve good to ±5-10 % is fine — the ML
residual learns the rest).

Coordinate conventions used throughout:
    * Latitude    +N / −S, degrees.
    * Longitude   +E / −W, degrees.
    * Azimuth     0° = north, 90° = east, 180° = south, 270° = west.
    * Hour angle  0° at solar noon, positive in the afternoon.
    * Tilt        0° = horizontal, 90° = vertical, measured from the horizontal.
    * Timestamps  always tz-aware (UTC or a real timezone). Naive timestamps
                  raise ``ValueError``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


# Solar constant at top of atmosphere (W/m²).
SOLAR_CONSTANT_W_M2: float = 1367.0

# Simple Beer-Lambert coefficient for direct beam attenuation. Chosen so the
# clear-sky GHI at zenith equals roughly 1000 W/m² in summer at moderate
# latitudes, matching the conventional standard test conditions.
BEAM_ATTENUATION: float = 0.30

# Fraction of direct horizontal that we model as isotropic diffuse on clear
# days. Empirical fudge; under-estimating diffuse is fine since the ML residual
# corrects.
DIFFUSE_FRACTION: float = 0.10


@dataclass(frozen=True)
class ClearSkyConfig:
    """Site + system geometry."""
    latitude: float                                # +N, degrees
    longitude: float                               # +E, degrees
    capacity_kwp: float                            # rated DC capacity (kWp)
    tilt_deg: float | None = None                  # None → use abs(latitude)
    azimuth_deg: float = 180.0                     # default south
    system_efficiency: float = 0.85                # DC→AC + soiling + temperature losses

    def effective_tilt_deg(self) -> float:
        if self.tilt_deg is not None:
            return self.tilt_deg
        return abs(self.latitude)


# ── Spencer (1971) series for declination & equation of time ────────────────
def _fractional_year_rad(ts: datetime) -> float:
    """Day-of-year angle γ in radians (Spencer 1971)."""
    n = ts.timetuple().tm_yday
    # Spencer uses (n − 1) and the fractional part of the day. We include the
    # time-of-day for sub-day precision so 06:00 and 18:00 give different
    # equation-of-time values within a day (small effect, but free to include).
    hours = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    return 2.0 * math.pi * ((n - 1) + (hours - 12.0) / 24.0) / 365.0


def _solar_declination_rad(gamma: float) -> float:
    """Declination angle δ in radians."""
    return (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.001480 * math.sin(3 * gamma)
    )


def _equation_of_time_min(gamma: float) -> float:
    """Equation of time in minutes."""
    return 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )


def _eccentricity_correction(gamma: float) -> float:
    """Earth-Sun distance correction factor for the solar constant."""
    return (
        1.000110
        + 0.034221 * math.cos(gamma)
        + 0.001280 * math.sin(gamma)
        + 0.000719 * math.cos(2 * gamma)
        + 0.000077 * math.sin(2 * gamma)
    )


# ── Solar position ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SolarPosition:
    elevation_deg: float                # > 0 above horizon, ≤ 0 below
    azimuth_deg: float                  # 0 = N, clockwise (E=90, S=180, W=270)
    declination_deg: float
    hour_angle_deg: float
    air_mass: float                     # ∞ when sun below horizon


def solar_position(ts: datetime, latitude: float, longitude: float) -> SolarPosition:
    """Compute sun elevation, azimuth, declination, hour angle, air mass.

    ``ts`` must be timezone-aware (UTC or any concrete zone).
    """
    if ts.tzinfo is None:
        raise ValueError("solar_position requires a timezone-aware datetime")

    ts_utc = ts.astimezone(timezone.utc)
    gamma = _fractional_year_rad(ts_utc)
    declination = _solar_declination_rad(gamma)
    eot_min = _equation_of_time_min(gamma)

    # Solar time at the longitude (in hours).
    utc_hours = ts_utc.hour + ts_utc.minute / 60.0 + ts_utc.second / 3600.0
    solar_time_h = utc_hours + (longitude / 15.0) + eot_min / 60.0
    hour_angle = math.radians(15.0 * (solar_time_h - 12.0))

    lat_rad = math.radians(latitude)
    sin_elev = (
        math.sin(lat_rad) * math.sin(declination)
        + math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle)
    )
    sin_elev = max(-1.0, min(1.0, sin_elev))
    elevation = math.asin(sin_elev)

    # Azimuth (0=N, clockwise). Standard formula with sign correction by hour angle.
    cos_elev = math.cos(elevation)
    if cos_elev < 1e-6:
        azimuth_deg = 180.0  # sun straight up; arbitrary
    else:
        cos_az = (
            math.sin(declination) - math.sin(lat_rad) * sin_elev
        ) / (math.cos(lat_rad) * cos_elev)
        cos_az = max(-1.0, min(1.0, cos_az))
        az = math.acos(cos_az)
        # In the morning (hour_angle < 0) the sun is to the east of the meridian.
        # In the afternoon (hour_angle > 0) it is to the west.
        if hour_angle > 0:
            azimuth_deg = math.degrees(2 * math.pi - az)
        else:
            azimuth_deg = math.degrees(az)

    # Air mass with the Kasten-Young (1989) approximation, valid down to the
    # horizon (and well-defined when the sun is below — we return inf there).
    if elevation <= 0:
        air_mass = float("inf")
    else:
        elev_deg = math.degrees(elevation)
        air_mass = 1.0 / (sin_elev + 0.50572 * (6.07995 + elev_deg) ** -1.6364)

    return SolarPosition(
        elevation_deg=math.degrees(elevation),
        azimuth_deg=azimuth_deg,
        declination_deg=math.degrees(declination),
        hour_angle_deg=math.degrees(hour_angle),
        air_mass=air_mass,
    )


# ── Clear-sky model ─────────────────────────────────────────────────────────
@dataclass
class ClearSkyModel:
    """Geometry-only PV output baseline.

    Reuse a single instance — the constructor stores ``ClearSkyConfig`` and
    nothing else. The model is stateless beyond that and safe to share across
    threads.
    """
    config: ClearSkyConfig

    # ── Irradiance ─────────────────────────────────────────────────────────
    def irradiance_w_m2(self, ts: datetime) -> float:
        """Plane-of-array clear-sky irradiance at ``ts`` (W/m²)."""
        sp = solar_position(ts, self.config.latitude, self.config.longitude)
        if sp.elevation_deg <= 0:
            return 0.0

        gamma = _fractional_year_rad(ts.astimezone(timezone.utc))
        e0 = _eccentricity_correction(gamma)
        i0 = SOLAR_CONSTANT_W_M2 * e0  # top of atmosphere normal incidence

        # Direct normal at the surface — simple Beer-Lambert.
        dni = i0 * math.exp(-BEAM_ATTENUATION * sp.air_mass)

        # GHI (horizontal): direct projected + isotropic diffuse approximation.
        sin_elev = math.sin(math.radians(sp.elevation_deg))
        dni_h = dni * sin_elev
        dhi = DIFFUSE_FRACTION * dni_h
        ghi = dni_h + dhi

        # Project onto tilted plane (incidence-angle method).
        tilt_rad = math.radians(self.config.effective_tilt_deg())
        sun_az_rad = math.radians(sp.azimuth_deg)
        panel_az_rad = math.radians(self.config.azimuth_deg)

        cos_incidence = (
            sin_elev * math.cos(tilt_rad)
            + math.cos(math.radians(sp.elevation_deg))
            * math.sin(tilt_rad)
            * math.cos(sun_az_rad - panel_az_rad)
        )
        cos_incidence = max(0.0, cos_incidence)

        direct_tilt = dni * cos_incidence
        diffuse_tilt = dhi * (1.0 + math.cos(tilt_rad)) / 2.0  # isotropic sky
        # Ground reflection ignored (small for typical installations).

        return max(0.0, direct_tilt + diffuse_tilt)

    # ── Power and energy ──────────────────────────────────────────────────
    def power_w(self, ts: datetime) -> float:
        """Instantaneous AC PV output (W) at ``ts``.

        At standard test conditions (1000 W/m², 25 °C), 1 kWp of rated DC
        capacity outputs 1 kW of DC power. We scale linearly with irradiance
        and apply ``system_efficiency`` to convert DC → AC.
        """
        irr = self.irradiance_w_m2(ts)
        # capacity_kwp rated at 1000 W/m² → power scales linearly with irr/1000.
        dc_w = self.config.capacity_kwp * 1000.0 * (irr / 1000.0)
        return dc_w * self.config.system_efficiency

    def kwh_for_hour(self, hour_start: datetime, samples: int = 4) -> float:
        """Energy produced during the hour starting at ``hour_start`` (kWh).

        Integrates ``power_w`` with the rectangle rule at ``samples`` equally
        spaced points. ``samples = 4`` (every 15 min) is the sweet spot
        between cost and accuracy — sub-hour irradiance changes are smooth
        on a clear day.
        """
        if samples < 1:
            raise ValueError("samples must be >= 1")
        step = timedelta(minutes=60 / samples)
        total_wh = 0.0
        for i in range(samples):
            t = hour_start + step * i + step / 2  # midpoint
            total_wh += self.power_w(t) * (step.total_seconds() / 3600.0)
        return total_wh / 1000.0

    def kwh_series(
        self, start: datetime, end: datetime, samples_per_hour: int = 4
    ) -> list[tuple[datetime, float]]:
        """Series of ``(hour_start, kwh)`` from ``start`` up to but not
        including ``end``. Aligned to whole-hour boundaries.
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("kwh_series requires timezone-aware datetimes")
        # Snap start to the top of its hour.
        cur = start.replace(minute=0, second=0, microsecond=0)
        out: list[tuple[datetime, float]] = []
        while cur < end:
            out.append((cur, self.kwh_for_hour(cur, samples=samples_per_hour)))
            cur = cur + timedelta(hours=1)
        return out
