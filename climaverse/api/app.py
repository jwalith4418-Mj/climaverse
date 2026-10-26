"""FastAPI surface for the twin. Clean, typed, deployment-ready.

    uvicorn climaverse.api.app:app --reload

State is built once at startup (synthetic or real). Endpoints expose the twin loop:
  GET  /health            service + twin health
  GET  /twin/state        current pilot-grid field + confidence + freshness
  GET  /twin/ledger       tail of the hash-chained sync log (+ chain validity)
  POST /predict           next-day field for the pilot grid
  POST /simulate          what-if scenario on current twin state
  GET  /validate          cached walk-forward skill report
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from ..config import RAIN_GRID, TEMP_GRID, PILOT, subset_indices
from ..data import load_variable
from ..twin import TwinState, SyncLedger, synchronise, twin_health
from ..model import Predictor, baselines
from ..validation import walk_forward
from ..simulation import ScenarioEngine, Scenario

_S: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _boot()
    yield


app = FastAPI(title="CLIMAVERSE — India Climate Decision Twin",
              version="0.1.0", lifespan=lifespan)


def _boot():
    days = 540
    rain = load_variable("rain", n_days=days)
    tmax = load_variable("tmax", n_days=days)
    la, lo = subset_indices(RAIN_GRID, PILOT)
    la2, lo2 = subset_indices(TEMP_GRID, PILOT)
    st = TwinState.empty(RAIN_GRID, PILOT, "rain")
    led = SyncLedger()
    for d in range(days):
        synchronise(st, rain.data[d], d, led)
    pred = Predictor(epochs=6).fit(rain.data[:int(days*0.7), la, lo])
    bt = walk_forward(rain.data[:, la, lo], split=0.7, seq_len=7, epochs=6)
    _S.update(real=rain.real, state=st, ledger=led, pred=pred, backtest=bt,
              rain_now=rain.data[-1, la, lo], tmax_now=tmax.data[-1, la2, lo2],
              rain_seq=rain.data[-7:, la, lo], last_day=days - 1)


@app.get("/health")
def health():
    h = twin_health(_S["state"], 0, max(0.0, _S["backtest"].skill_vs_clim))
    return {"status": "ok", "data_is_real": _S["real"], "twin_health": h.as_dict()}


@app.get("/twin/state")
def twin_state():
    s = _S["state"]
    return {"var": s.var, "shape": list(s.value.shape), "last_obs_day": s.last_obs_day,
            "mean": round(float(np.nanmean(s.value)), 3),
            "confidence_mean": round(float(s.confidence.mean()), 3),
            "field": np.nan_to_num(s.value).round(2).tolist()}


@app.get("/twin/ledger")
def ledger(n: int = 5):
    return {"chain_valid": _S["ledger"].verify(), "length": len(_S["ledger"]),
            "tail": _S["ledger"].tail(n)}


class PredictReq(BaseModel):
    target_doy: int = 210


@app.post("/predict")
def predict(req: PredictReq):
    f = _S["pred"].predict_next(_S["rain_seq"], req.target_doy)
    return {"target_doy": req.target_doy, "backend": _S["pred"].backend,
            "mean_mm": round(float(np.nanmean(f)), 3), "field": f.round(2).tolist()}


class SimReq(BaseModel):
    name: str = "Monsoon deficit + warming"
    temp_delta_c: float = 2.0
    rain_pct: float = -20.0


@app.post("/simulate")
def simulate(req: SimReq):
    res = ScenarioEngine().run(_S["tmax_now"], _S["rain_now"],
                               Scenario(req.name, req.temp_delta_c, req.rain_pct))
    return res.as_dict()


@app.get("/validate")
def validate():
    return _S["backtest"].as_dict()
