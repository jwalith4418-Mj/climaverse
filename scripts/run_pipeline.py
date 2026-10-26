#!/usr/bin/env python3
"""End-to-end CLIMAVERSE twin loop — one command, judge-readable report.

    python scripts/run_pipeline.py [--days 730] [--epochs 12] [--var rain|tmax]

Runs the FULL digital-twin cycle:
  observe -> synchronise (audit-logged) -> train -> validate (walk-forward) ->
  learn (bias recalibration) -> simulate (what-if) -> twin health.
Prints honest numbers; states clearly whether data is REAL or synthetic.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from repo root
import numpy as np

from climaverse.config import RAIN_GRID, TEMP_GRID, PILOT, subset_indices
from climaverse.data import load_variable
from climaverse.twin import TwinState, SyncLedger, synchronise, twin_health
from climaverse.model import Predictor, baselines
from climaverse.validation import walk_forward, BiasCorrector
from climaverse.simulation import ScenarioEngine, Scenario

GRIDS = {"rain": RAIN_GRID, "tmax": TEMP_GRID}


def hr(t): print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--var", default="rain", choices=["rain", "tmax"])
    a = ap.parse_args()
    t0 = time.time()

    hr("1. OBSERVE  — load national IMD-format field")
    obs = load_variable(a.var, n_days=a.days)
    grid = GRIDS[a.var]
    la, lo = subset_indices(grid, PILOT)
    print(f"variable      : {obs.label}")
    print(f"data source   : {'REAL IMD .grd' if obs.real else 'SYNTHETIC (offline stand-in)'}")
    print(f"national grid : {obs.data.shape[1:]}  pilot region: {PILOT.name} {obs.data[:,la,lo].shape[1:]}")

    hr("2. SYNCHRONISE — assimilate every day into the living twin (audit-logged)")
    state = TwinState.empty(grid, PILOT, a.var)
    ledger = SyncLedger()
    for d in range(obs.data.shape[0]):
        synchronise(state, obs.data[d], d, ledger)
    print(f"sync records  : {len(ledger)}   hash-chain valid: {ledger.verify()}")
    print(f"last record   : {json.dumps(ledger.tail(1)[0], default=str)}")

    hr("3-4. PREDICT + VALIDATE — walk-forward, skill vs scientific baselines")
    sub = obs.data[:, la, lo]
    bt = walk_forward(sub, split=0.7, seq_len=7, epochs=a.epochs,
                      log=lambda *m: print("   ", *m))
    d = bt.as_dict()
    print(f"\nbackend             : {d['backend']}")
    print(f"holdout days        : {d['n_days']}")
    print(f"model RMSE          : {d['model_rmse']}")
    print(f"climatology RMSE    : {d['clim_rmse']}   persistence RMSE: {d['persist_rmse']}")
    print(f"SKILL vs climatology: {d['skill_vs_clim']:+.3f}   "
          f"SKILL vs persistence: {d['skill_vs_persist']:+.3f}")
    verdict = "BEATS" if d['skill_vs_clim'] > 0 else "does NOT beat"
    print(f"verdict             : model {verdict} climatology (the bar that matters)")

    hr("5. LEARN — bias recalibration on resolved errors (identity until earned)")
    cut = int(sub.shape[0] * 0.7)
    pred = Predictor(epochs=a.epochs).fit(sub[:cut])
    clim = baselines.day_of_year_climatology(sub[:cut])
    P, O = [], []
    for i in range(max(cut, 7), sub.shape[0]):
        P.append(pred.predict_next(sub[i-7:i], i % 365)); O.append(sub[i])
    bc = BiasCorrector().fit(np.asarray(P), np.asarray(O))
    print(f"resolved pairs : {len(P)*sub.shape[1]*sub.shape[2]}   corrector active: {bc.active}")

    hr("6. SIMULATE — what-if on the current twin state")
    tmax_obs = load_variable("tmax", n_days=a.days)
    la2, lo2 = subset_indices(TEMP_GRID, PILOT)
    tmax_now = tmax_obs.data[-1, la2, lo2]
    rain_now = load_variable("rain", n_days=a.days).data[-1, la, lo]
    res = ScenarioEngine().run(tmax_now, rain_now,
                               Scenario("Monsoon deficit + warming", temp_delta_c=2.0, rain_pct=-20))
    for k in res.deltas:
        print(f"  {k:20s}: {res.baseline[k]:>8} -> {res.perturbed[k]:>8}  (Δ {res.deltas[k]:+})")
    print("  " + res.notes[0])

    hr("7. TWIN HEALTH — one honest number")
    h = twin_health(state, days_since_obs=0, skill_score=max(0.0, d['skill_vs_clim']))
    print(json.dumps(h.as_dict(), indent=2))

    hr("DONE")
    print(f"full loop in {time.time()-t0:.1f}s on CPU. "
          f"{'REAL data.' if obs.real else 'Synthetic data — drop IMD .grd in data/raw/ for real numbers.'}")


if __name__ == "__main__":
    sys.exit(main())
