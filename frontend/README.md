# Frontend — two signature moments (build spec)

The backend is the hard, scored part and it is done. The UI's only job is to make the
**Digital Twin** legible and the demo unforgettable. Do NOT build a generic dashboard
with ten panels — build two moments, polished.

Recommended stack: Next.js + deck.gl (GeoJSON grid layer over a basemap) + a thin
fetch layer to the FastAPI endpoints already implemented in `climaverse/api/app.py`.

## Moment 1 — Living Climate Twin
- Render the pilot grid as coloured cells over a map of peninsular India.
- Poll `GET /twin/state` → colour by value, opacity by `confidence`.
- A "SYNC" control advances `obs_day`; cells flip to fresh colour as data assimilates.
- Side panel: `GET /twin/ledger` → show the hash chain growing, `chain_valid: true`.
  *This is the reality-vs-twin / auditability story judges remember.*
- Corner badge: `GET /health` → Twin Health gauge (freshness / coverage / skill).

## Moment 2 — Climate Time Machine
- A timeline scrubber (day → +7d → season).
- Two sliders: rainfall % and temperature Δ°C → `POST /simulate`.
- Animate the response: heat-stress cells light red, dry cells fade brown, and the
  delta cards (`et_proxy_mm`, `dry_cells`, `heat_stress_cells`) tick in real time.
- "AI narrator" line = `notes[0]` from the simulate response, typed out.

## API contract (already live)
```
GET  /health          -> {status, data_is_real, twin_health{freshness,coverage,skill,overall}}
GET  /twin/state      -> {var, shape, mean, confidence_mean, field[][]}
GET  /twin/ledger?n=5 -> {chain_valid, length, tail[]}
POST /predict {target_doy}              -> {mean_mm, field[][], backend}
POST /simulate {temp_delta_c, rain_pct} -> {baseline, perturbed, deltas, notes[]}
GET  /validate        -> {model_rmse, clim_rmse, skill_vs_clim, skill_vs_persist, ...}
```
Keep the palette scientific and minimal (single accent, dark canvas). Cinematic =
restraint + motion on state change, not neon everywhere.
