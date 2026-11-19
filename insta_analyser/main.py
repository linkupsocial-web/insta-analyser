"""
FastAPI service for the Instagram Reel Analyser.

A run takes minutes (crawl, downloads, Whisper), longer than n8n's HTTP node
should wait, so the API works as a background job:

    POST /analyse            starts a run, returns {job_id, status: "queued"} at once
    GET  /analyse/{job_id}   status, the last node that finished, and the result when done
    GET  /analyse/{job_id}/report/{category}   the educational or promotional report as Markdown
    GET  /health             Ollama reachable + models pulled

When a job finishes, both reports are also written to reports/<job_id>/
(educational_report.md/.json, promotional_report.md/.json).

Runs execute one at a time (polite to Instagram, and Whisper uses the whole CPU);
extra jobs wait in the queue. Jobs live in memory, so a restart forgets them.

Run locally (port 8002, so it doesn't clash with the strategist on 8001):
    uv run uvicorn insta_analyser.main:app --port 8010
    uv run uvicorn insta_analyser.main:app --port 8010 --reload

Call it:
    curl -X POST http://localhost:8002/analyse \
      -H "Content-Type: application/json" \
      -d '{"handles": ["digistreet.media", "digidir_"]}'

    curl http://localhost:8002/analyse/<job_id>

In n8n: HTTP Request (POST /analyse) -> Wait 30s -> HTTP Request (GET /analyse/{{job_id}})
-> IF status is "queued" or "running", loop back to Wait.
"""

import asyncio
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from insta_analyser.analyser_graph import run_analysis
from insta_analyser.analyser_models import check_models
from insta_analyser.analyser_schemas import AnalyseRequest, JobStatus
from insta_analyser.report_files import CATEGORIES, render_markdown, write_report_files

MAX_JOBS_KEPT = 50
REPORTS_DIR = Path("reports")

app = FastAPI(
    title="Linkup Instagram Reel Analyser",
    version="0.1.0",
)

JOBS: "OrderedDict[str, JobStatus]" = OrderedDict()
RUN_LOCK = asyncio.Lock()
_TASKS: set = set()  # keeps background tasks referenced until they finish


def _remember(job: JobStatus) -> None:
    JOBS[job.job_id] = job
    while len(JOBS) > MAX_JOBS_KEPT:
        oldest = next(iter(JOBS))
        if JOBS[oldest].status in ("queued", "running"):
            break
        JOBS.pop(oldest)


async def _run_job(job_id: str, request: AnalyseRequest) -> None:
    job = JOBS[job_id]
    async with RUN_LOCK:
        job.status = "running"

        def on_step(node: str) -> None:
            job.step = node

        try:
            job.result = await run_analysis(request, on_step=on_step)
            job.status = "done"
            try:
                write_report_files(job.result, REPORTS_DIR / job_id)
            except OSError as exc:  # the result is still served even if the files can't be written
                job.error = f"report files not written: {exc}"
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {str(exc)[:500]}"


@app.get("/")
def root():
    return {"status": "ok", "service": "Linkup Instagram Reel Analyser"}


@app.get("/health")
def health():
    models = check_models()
    ok = models["ollama_reachable"] and not models["missing"]
    return {"status": "ok" if ok else "degraded", **models}


@app.post("/analyse", response_model=JobStatus, status_code=202)
async def start_analysis(request: AnalyseRequest):
    models = check_models()
    if not models["ollama_reachable"] or models["missing"]:
        raise HTTPException(status_code=503, detail={"message": "models not ready", **models})

    job = JobStatus(job_id=uuid.uuid4().hex, status="queued")
    _remember(job)
    task = asyncio.create_task(_run_job(job.job_id, request))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return job


@app.get("/analyse/{job_id}", response_model=JobStatus)
def get_analysis(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job_id (jobs are kept in memory; a restart clears them)")
    return job


@app.get("/analyse/{job_id}/report/{category}", response_class=PlainTextResponse)
def get_report(job_id: str, category: str):
    if category not in CATEGORIES:
        raise HTTPException(status_code=404, detail=f"category must be one of {list(CATEGORIES)}")
    job = get_analysis(job_id)
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=409, detail=f"job is {job.status}; the report is ready when it is done")
    return PlainTextResponse(render_markdown(job.result, category), media_type="text/markdown; charset=utf-8")
