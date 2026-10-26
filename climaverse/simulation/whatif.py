"""What-if scenario engine.

Perturbs the twin's current state and propagates FIRST-ORDER, transparent impacts.
Every relationship below is an explicit, citable proxy — not a hidden black box and
not invented precision. The point of the PoC is to show the *mechanism* of scenario
analysis on a living twin; the proxies are swappable for calibrated sectoral models.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Scenario:
    name: str
    temp_delta_c: float = 0.0        # uniform temperature shift (deg C)
    rain_pct: float = 0.0            # rainfall change, e.g. -20 => 20% deficit


@dataclass
class ScenarioResult:
    scenario: Scenario
    baseline: dict
    perturbed: dict
    deltas: dict
    notes: list[str]
    def as_dict(self):
        return {"scenario": self.scenario.__dict__, "baseline": self.baseline,
                "perturbed": self.perturbed, "deltas": self.deltas, "notes": self.notes}


def _indices(tmax: np.ndarray, rain: np.ndarray) -> dict:
    """Decision-relevant aggregate indices over the pilot grid."""
    return {
        "mean_tmax_c": round(float(np.nanmean(tmax)), 2),
        "heat_stress_cells": int(np.nansum(tmax > 40.0)),          # Tmax>40C exposure
        "mean_rain_mm": round(float(np.nanmean(rain)), 2),
        "dry_cells": int(np.nansum(rain < 1.0)),                   # near-zero rain
        # Hargreaves-style ET proxy: ET rises ~linearly with temperature.
        "et_proxy_mm": round(float(np.nanmean(0.0023 * (tmax + 17.8) * np.sqrt(np.abs(tmax)))) * 10, 2),
    }


class ScenarioEngine:
    """Operates on the *current twin state* (tmax field, rain field) over the pilot grid."""

    def run(self, tmax: np.ndarray, rain: np.ndarray, scenario: Scenario) -> ScenarioResult:
        base = _indices(tmax, rain)
        tmax_p = tmax + scenario.temp_delta_c
        rain_p = rain * (1.0 + scenario.rain_pct / 100.0)
        pert = _indices(tmax_p, rain_p)
        deltas = {k: round(pert[k] - base[k], 2) for k in base}

        notes = []
        if scenario.temp_delta_c:
            notes.append(f"+{scenario.temp_delta_c} C raises evapotranspiration demand "
                         f"by ~{deltas['et_proxy_mm']} mm and adds {deltas['heat_stress_cells']} "
                         "heat-stress grid-cells (Tmax>40 C).")
        if scenario.rain_pct:
            notes.append(f"{scenario.rain_pct:+.0f}% rainfall shifts mean to "
                         f"{pert['mean_rain_mm']} mm/day and changes dry grid-cells "
                         f"by {deltas['dry_cells']}.")
        notes.append("First-order proxies (Hargreaves ET, threshold exposure counts). "
                     "Swap for calibrated crop/hydrology models in production.")
        return ScenarioResult(scenario, base, pert, deltas, notes)
