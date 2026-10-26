# CLIMAVERSE — India's AI Climate Decision Twin

**Bharatiya Antariksh Hackathon 2026 · Problem Statement 05**
*AI-Powered Digital Twin of India's Climate using India's National Data*

> Most teams will build `Satellite → CNN/LSTM → Prediction → Dashboard`.
> That scores well on *prediction* and badly on *Digital Twin* — the category the
> brief names five times. CLIMAVERSE is built as the loop the brief actually asks for:
>
> **observe → synchronise → predict → simulate → compare → learn → re-synchronise**
>
> Prediction is one layer of seven. The twin is the product.

---

## Why this wins the rubric (not just the demo)

| Official evaluation parameter | Where it lives in this repo | Honest status |
|---|---|---|
| Problem Understanding & Clarity | This README + layered architecture | ✅ |
| Data Usage & Pre-processing | `data/imd_grid.py` parses the **exact** IMD `.grd` byte layout (129×135 rain, 31×31 temp) | ✅ real format; synthetic stand-in until you drop real files |
| Model Development & Technical Approach | `model/convlstm.py` — ConvLSTM predicting **tendency over persistence** (anomaly design) | ✅ real PyTorch model |
| Prediction Performance & Validation | `validation/backtest.py` — walk-forward, **skill vs climatology & persistence** | ✅ model beats climatology +10–38% |
| **Digital Twin Concept Implementation** | `twin/` — living state, hash-chained sync ledger, health score, + `validation/recalibrate.py` learn-loop | ✅ the differentiator |
| Visualization & User Interface | `frontend/` spec: Living Twin + Climate Time Machine | ◻ spec'd; build next |
| Innovation & Creativity | Tamper-evident sync ledger, reality-vs-twin learning, honest skill | ✅ |
| Presentation & Communication | `scripts/run_pipeline.py` prints a judge-readable loop in one command | ✅ |

**Two signature demo moments (and only two — focus beats spectacle):**
1. **Living Climate Twin** — India's pilot grid updating cell-by-cell as data is assimilated, each update written to a verifiable hash chain.
2. **Climate Time Machine** — scrub a timeline; apply a what-if (monsoon −20%, +2 °C); watch heat-stress and dry-cell counts respond.

---

## The architecture (strict order — Layers 1–3 are never sacrificed)

```
National IMD/INSAT data
   │
1. Living Climate Digital Twin     twin/state.py · sync.py · ledger.py · health.py
   │   continuously-updated state, freshness, confidence, tamper-evident history
2. AI Prediction Engine            model/convlstm.py · baselines.py · predict.py
   │   ConvLSTM tendency model; climatology + persistence as the bars to beat
3. What-if Scenario Simulation     simulation/whatif.py
   │   transparent first-order impact propagation (ET, heat-stress, dry-spell)
4. Validation + Continuous Learning validation/backtest.py · metrics.py · recalibrate.py
   │   walk-forward skill; isotonic bias correction fit on resolved errors
5. Clean API surface               api/app.py  (FastAPI)
6. Climate Time Machine            frontend/ (spec)
7. Living India Visualization      frontend/ (spec)
```

## Run it

```bash
pip install -r requirements.txt           # torch optional; falls back to baselines
python scripts/run_pipeline.py            # full loop, ~110s CPU, judge-readable report
python scripts/run_pipeline.py --var tmax # temperature instead of rainfall
uvicorn climaverse.api.app:app --reload   # API at http://127.0.0.1:8000/docs
pytest -q                                 # 7 tests incl. model-beats-climatology
```

## Representative results (SYNTHETIC offline data — see honesty note)

| Variable | Model RMSE | Climatology RMSE | Skill vs climatology | Skill vs persistence |
|---|---|---|---|---|
| Rainfall (pilot 0.25°) | ~4.1 mm | ~4.8 mm | **+0.15** | **+0.15** |
| Tmax (pilot 1.0°) | ~0.15 | ~0.23 | **+0.38** | ~0.00 (its floor) |

Rainfall (low persistence) → the ConvLSTM beats *both* baselines. Temperature (high
persistence) → it beats climatology by 38% and *ties* persistence, which is the
mathematical floor of the tendency design. Numbers move when you load real IMD data.

## Honesty notes (read these before you demo)

- **No real data is bundled.** The IMD portal needs manual download. Everything runs
  on a *format-identical synthetic field* clearly flagged `data_is_real: false` in
  every report. Drop real `.grd` files in `data/raw/` (see that folder's README) and
  every number becomes real with zero code changes.
- **The model never hides behind RMSE.** Skill is always reported relative to
  climatology and persistence. If skill ≤ 0 the report says so.
- **The twin never claims skill it hasn't earned.** Health skill is 0 with no
  resolved forecast; the bias corrector is the identity below 60 resolved pairs.
- **What-if impacts are first-order proxies**, labelled as such — a mechanism demo,
  not calibrated sectoral forecasting.

## Scaling to national deployment
The pilot region is a config box (`config.PILOT`). Widen it to all-India and the same
loop runs per-tile; the twin state, ledger, predictor and API are grid-agnostic. INSAT
LST/SST/IMR (MOSDAC `3RIMG_L2B_*`) regrid onto the IMD grid and assimilate identically.
