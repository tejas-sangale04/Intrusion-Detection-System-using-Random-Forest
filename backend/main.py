"""Phase 7: FastAPI service for the hybrid detector.

Run from the repository root, after Phases 3 and 5 have saved their models:

    uvicorn backend.main:app --port 8000

Interactive documentation is then at http://127.0.0.1:8000/docs.

Design choices
--------------
* **Models load once, at startup.** Loading the forest takes about a second;
  doing it per request would dominate latency. If a model file is missing the
  server refuses to start with a message naming the command that creates it,
  rather than starting and failing on every request.
* **Offline mode only.** Flows arrive as feature vectors over HTTP, e.g.
  rows of a CIC-IDS2017 CSV. Nothing here captures packets, and nothing
  blocks hosts or changes firewall rules: decisions are logged and returned.
* **Scoring runs in a worker thread** (plain ``def`` endpoints), so a slow
  SHAP computation does not stall other requests.
* **Explanations are computed on request and cached.** SHAP takes about a
  third of a second per flow for this forest, which is too slow to do for
  every prediction in a batch but fine for the one alert an analyst opens.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import __version__
from backend.config import Settings
from backend.logging_setup import configure_logging
from backend.schemas import (
    AlertOut,
    AlertsOut,
    BatchIn,
    BatchOut,
    Decision,
    ExplanationOut,
    FlowIn,
    PredictionOut,
)
from backend.services.detector import CLASSIFIER_THRESHOLD, Detector, InvalidFlow
from backend.services.reports import offline_metrics
from backend.services.store import PredictionRecord, PredictionStore

logger = logging.getLogger("backend")

MODE = "offline"  # flows are submitted by a client; no live capture in this phase


class Runtime:
    """Counters about this server process only. Reset on restart."""

    def __init__(self):
        self.started = time.time()
        self.requests = 0
        self.flows_scored = 0
        self.scoring_seconds = 0.0
        self.lock = threading.Lock()  # endpoints run in a thread pool

    def record(self, flows: int, seconds: float) -> None:
        with self.lock:
            self.requests += 1
            self.flows_scored += flows
            self.scoring_seconds += seconds

    def as_dict(self, store: PredictionStore) -> dict:
        return {
            "uptime_seconds": round(time.time() - self.started, 1),
            "prediction_requests": self.requests,
            "flows_scored": self.flows_scored,
            "decision_counts": {d: store.decision_counts.get(d, 0) for d in ("benign", "anomaly", "known_attack")},
            "mean_scoring_ms_per_flow": round(1000 * self.scoring_seconds / self.flows_scored, 3)
            if self.flows_scored else None,
            "predictions_in_memory": len(store),
            "predictions_evicted": store.evicted,
            "note": "Counts of decisions made by this server since it started. They are not accuracy figures: "
                    "submitted flows carry no ground truth.",
        }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        start = time.perf_counter()
        app.state.detector = Detector(settings.models_dir)
        app.state.store = PredictionStore(settings.store_size, settings.alert_log)
        app.state.runtime = Runtime()
        logger.info("models loaded", extra={"seconds": round(time.perf_counter() - start, 2),
                                            "models_dir": str(settings.models_dir), "mode": MODE})
        yield

    app = FastAPI(
        title="Explainable Hybrid Network IDS",
        version=__version__,
        description="Random Forest + Isolation Forest detector for CIC-IDS2017 flow features, with SHAP "
                    "explanations. Offline mode: decisions are returned and logged; nothing is blocked.",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        logger.info("request", extra={"method": request.method, "path": request.url.path,
                                      "status": response.status_code,
                                      "ms": round(1000 * (time.perf_counter() - start), 2)})
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Same shape as InvalidFlow errors; never echo the submitted values back.
        errors = [{"location": ".".join(str(p) for p in e["loc"]), "error": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": "Invalid request body", "errors": errors})

    @app.exception_handler(InvalidFlow)
    async def invalid_flow(request: Request, exc: InvalidFlow):
        return JSONResponse(status_code=422, content={"detail": "Invalid flow features", "errors": exc.problems})

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        logger.exception("unhandled error", extra={"path": request.url.path})
        return JSONResponse(status_code=500, content={"detail": "Internal error; see server log"})

    def score_and_store(request: Request, flows: list[dict], source: str) -> list[PredictionOut]:
        detector: Detector = request.app.state.detector
        store: PredictionStore = request.app.state.store
        runtime: Runtime = request.app.state.runtime
        frame, missing = detector.to_frame(flows)
        start = time.perf_counter()
        scores = detector.score(frame)
        runtime.record(len(frame), time.perf_counter() - start)
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        out = []
        for i in range(len(frame)):
            pid = uuid.uuid4().hex
            record = PredictionRecord(
                prediction_id=pid, created_at=now, source=source, decision=str(scores.decision[i]),
                attack_probability=float(scores.attack_probability[i]),
                anomaly_score=float(scores.anomaly_score[i]), missing_values=missing[i],
                features=frame.iloc[i].to_dict(),
            )
            store.add(record)
            out.append(PredictionOut(**record.summary(), classifier_threshold=CLASSIFIER_THRESHOLD,
                                     anomaly_threshold=detector.anomaly_threshold,
                                     explanation_url=f"/explanations/{pid}"))
        alerts = sum(o.decision != "benign" for o in out)
        logger.info("scored", extra={"flows": len(out), "alerts": alerts, "source": source})
        return out

    @app.get("/health", tags=["service"])
    def health(request: Request) -> dict:
        return {"status": "ok", "version": __version__, "mode": MODE,
                "models_loaded": hasattr(request.app.state, "detector")}

    @app.get("/model/info", tags=["service"])
    def model_info(request: Request) -> dict:
        return {"mode": MODE, **request.app.state.detector.info()}

    @app.post("/predict", response_model=PredictionOut, tags=["detection"])
    def predict(body: FlowIn, request: Request) -> PredictionOut:
        return score_and_store(request, [body.features], body.source)[0]

    @app.post("/predict/batch", response_model=BatchOut, tags=["detection"])
    def predict_batch(body: BatchIn, request: Request) -> BatchOut:
        if len(body.flows) > settings.max_batch:
            raise HTTPException(413, f"At most {settings.max_batch} flows per request (got {len(body.flows)}); "
                                     "split the batch or raise IDS_MAX_BATCH")
        preds = score_and_store(request, body.flows, body.source)
        counts = {d: sum(p.decision == d for p in preds) for d in ("benign", "anomaly", "known_attack")}
        return BatchOut(count=len(preds), decision_counts=counts, predictions=preds)

    @app.get("/alerts", response_model=AlertsOut, tags=["detection"])
    def alerts(request: Request, limit: int = Query(100, ge=1, le=1000),
               decision: Decision | None = Query(None)) -> AlertsOut:
        if decision == "benign":
            raise HTTPException(422, "Alerts are 'anomaly' or 'known_attack' decisions")
        records = request.app.state.store.alerts(limit, decision)
        return AlertsOut(count=len(records), alerts=[AlertOut(**r.summary()) for r in records],
                         note="Recent alerts held in memory by this server, newest first. 'anomaly' means unusual "
                              "compared with benign training traffic, not confirmed malicious.")

    @app.get("/explanations/{prediction_id}", response_model=ExplanationOut, tags=["detection"])
    def explanation(prediction_id: str, request: Request, k: int = Query(10, ge=1, le=60)) -> ExplanationOut:
        record = request.app.state.store.get(prediction_id)
        if record is None:
            raise HTTPException(404, "Unknown prediction id, or it was dropped from memory "
                                     "(the server keeps the most recent IDS_STORE_SIZE predictions)")
        if record.explanation is None:
            detector: Detector = request.app.state.detector
            frame = pd.DataFrame([record.features], columns=detector.input_features)
            start = time.perf_counter()
            record.explanation = detector.explain(frame, k=60)
            logger.info("explained", extra={"ms": round(1000 * (time.perf_counter() - start), 1)})
        e = record.explanation
        scope = ("Explains the Random Forest's attack probability. The Isolation Forest anomaly score is not "
                 "decomposed here." if record.decision != "anomaly" else
                 "This flow was flagged by the Isolation Forest, whose score is not decomposed here. The SHAP "
                 "values show why the Random Forest scored it below 0.5, i.e. why it did not match a known attack.")
        return ExplanationOut(prediction_id=prediction_id, decision=record.decision,
                              attack_probability=e["probability"], base_value=e["base_value"],
                              top_features=e["top_features"][:k], summary=e["summary"], scope=scope)

    @app.get("/metrics", tags=["service"])
    def metrics(request: Request) -> dict:
        return {"offline_evaluation": offline_metrics(settings.reports_dir),
                "runtime": request.app.state.runtime.as_dict(request.app.state.store)}

    return app


app = create_app()
