# Manual season switch — Home Assistant side

Companion design for the v5.0.3 `season_select` role. The addon already
consumes whatever `input_select` you wire into the wizard; this document
covers the Home Assistant configuration that makes the same selector flip
the heating/cooling setpoint contracts used by the rest of the house.

The reference install is Sergio's: aerotermia heating + radiant cooling, with
several `climate` / `number` thermostats today duplicated in
`configuration.yaml` and toggled by hand twice a year (one block commented
out, the other live). The goal is to replace that ritual with a single helper.

## 1. Define the selector

Create the helper from the UI (Settings → Devices & services → Helpers →
Create helper → Dropdown) or in YAML:

```yaml
# input_select.yaml (or inline in configuration.yaml under input_select:)
season:
  name: Estación
  options:
    - summer
    - winter
  initial: winter
  icon: mdi:sun-snowflake-variant
```

The addon also accepts `verano` / `invierno` as labels, so you can localise
the UI without breaking the consumer side.

## 2. Wire the addon

Setup Wizard → Sensors → "Season selector entity" → `input_select.season`.
That's the only change required addon-side; restart the addon to pick it up.

## 3. Replace commented thermostats with templates

The cleanest pattern for "swap thermostat by season" is a single
`generic_thermostat` (or `climate.template`) that resolves its target /
heater / cooler entities from the selector via templates. Worked example for
a single zone:

```yaml
# configuration.yaml
template:
  - sensor:
      - name: "Season heater entity"
        unique_id: season_heater_entity
        state: >
          {% if states('input_select.season') == 'summer' %}
            switch.aerotermia_cool_zone1
          {% else %}
            switch.aerotermia_heat_zone1
          {% endif %}

      - name: "Season target temperature"
        unique_id: season_target_temperature
        state: >
          {% if states('input_select.season') == 'summer' %}
            {{ states('input_number.cool_setpoint_zone1') }}
          {% else %}
            {{ states('input_number.heat_setpoint_zone1') }}
          {% endif %}
        unit_of_measurement: "°C"
```

The previously-duplicated `climate:` blocks collapse to one entry that reads
from these templates instead of being commented in/out.

> For installs where each thermostat references *different* hardware in
> summer vs. winter (e.g. a wall radiator in winter and a floor loop in
> summer), keep both `climate` entries but gate them with
> `availability_template` so only one is active at a time. The selector
> flip then trivially toggles which one HA exposes to the dashboard / the
> addon.

## 4. Optional: react to the flip via automation

If you have side effects that should run *at the moment of the flip*
(turning the radiant-floor pump valve, swapping the aerotermia mode, etc.):

```yaml
automation:
  - alias: "Season → summer (radiant cooling)"
    trigger:
      - platform: state
        entity_id: input_select.season
        to: summer
    action:
      - service: select.select_option
        target: { entity_id: select.aerotermia_mode }
        data: { option: "cooling" }
      - service: switch.turn_on
        target: { entity_id: switch.radiant_floor_cooling_valve }

  - alias: "Season → winter (heating)"
    trigger:
      - platform: state
        entity_id: input_select.season
        to: winter
    action:
      - service: select.select_option
        target: { entity_id: select.aerotermia_mode }
        data: { option: "heating" }
      - service: switch.turn_off
        target: { entity_id: switch.radiant_floor_cooling_valve }
```

## 5. Dashboard

Surface the selector as a card next to the energy panel so the flip is
visible and one-tap:

```yaml
type: entities
title: Estación
entities:
  - entity: input_select.season
  - entity: sensor.season_heater_entity
  - entity: sensor.season_target_temperature
```

## Why a manual selector instead of a month-based or temperature-based rule

- A month rule cannot accommodate freak weather (heatwave in May, cold snap
  in September) without re-deploying the addon config.
- A temperature rule (e.g. "if outdoor 7-day mean > X, summer") is
  adaptive but introduces a phantom mode-flip risk during shoulder seasons
  — particularly bad for a radiant floor that takes a day to swing.
- A user-controlled selector is predictable, debuggable, scoped to a
  single source of truth, and trivially auditable from the energy
  dashboard. Cost: one manual flip in spring and autumn.

The addon's month-based fallback is still available for installs that
don't wire the selector (or whose owners prefer the calendar to decide).
