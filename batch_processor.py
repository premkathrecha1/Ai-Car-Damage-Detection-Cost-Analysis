"""
batch_processor.py
==================
Background batch processing for AutoScan AI.

Usage (called from app.py)
--------------------------
    from batch_processor import submit_batch, get_job_status, JOBS

    job_id = submit_batch(file_list, segment, age, panels)
    # → returns job_id (str UUID)

    status = get_job_status(job_id)
    # → returns dict with progress info

Each job runs in a daemon thread.
Poll /batch/<job_id> every 1–2 s for live progress.

Job status dict schema
----------------------
{
  "job_id":       str,
  "state":        "queued" | "running" | "done" | "error",
  "total":        int,
  "completed":    int,
  "progress_pct": float,
  "results":      list[dict],   # filled as each image completes
  "summary":      dict | None,  # filled when state == "done"
  "error":        str | None,
  "started_at":   str,
  "finished_at":  str | None,
}

Summary dict schema
-------------------
{
  "total":            int,
  "damaged_count":    int,
  "damage_rate":      float (0-1),
  "avg_cost_mid":     float,
  "total_cost_mid":   float,
  "parts_breakdown":  dict[part -> count],
  "damage_breakdown": dict[damage -> count],
  "severity_breakdown": dict[severity -> count],
  "cost_breakdown":   list[{"part": str, "avg_cost": float}],
}
"""

import uuid, threading, datetime, io
from PIL import Image
from inference import predict
from vision_analysis import analyse as vision_analyse

# ── In-memory job store ────────────────────────────────────────────────────────
JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _build_summary(results: list[dict]) -> dict:
    """Aggregate completed job results into a summary dict."""
    total   = len(results)
    damaged = [r for r in results if r.get("is_damaged")]

    parts_breakdown    = {}
    damage_breakdown   = {}
    severity_breakdown = {}
    cost_by_part: dict[str, list[float]] = {}

    for r in results:
        p  = r.get("part",   "unknown")
        d  = r.get("damage", "unknown")
        s  = r.get("severity", "no_damage")
        parts_breakdown[p]    = parts_breakdown.get(p,    0) + 1
        damage_breakdown[d]   = damage_breakdown.get(d,   0) + 1
        severity_breakdown[s] = severity_breakdown.get(s, 0) + 1

        cost = r.get("cost")
        if cost and cost.get("mid"):
            cost_by_part.setdefault(p, []).append(cost["mid"])

    avg_cost_mid   = 0.0
    total_cost_mid = 0.0
    cost_breakdown = []

    if damaged:
        costs = [r["cost"]["mid"] for r in damaged if r.get("cost")]
        if costs:
            avg_cost_mid   = sum(costs) / len(costs)
            total_cost_mid = sum(costs)

    for part, vals in cost_by_part.items():
        cost_breakdown.append({"part": part, "avg_cost": round(sum(vals) / len(vals), 0)})
    cost_breakdown.sort(key=lambda x: x["avg_cost"], reverse=True)

    return {
        "total":               total,
        "damaged_count":       len(damaged),
        "damage_rate":         round(len(damaged) / max(total, 1), 3),
        "avg_cost_mid":        round(avg_cost_mid, 0),
        "total_cost_mid":      round(total_cost_mid, 0),
        "parts_breakdown":     parts_breakdown,
        "damage_breakdown":    damage_breakdown,
        "severity_breakdown":  severity_breakdown,
        "cost_breakdown":      cost_breakdown,
    }


def _worker(job_id: str,
            image_bytes_list: list[bytes],
            segment: str,
            age: int,
            panels: int) -> None:
    """Thread target — processes each image sequentially and updates JOBS."""

    with _JOBS_LOCK:
        JOBS[job_id]["state"]      = "running"
        JOBS[job_id]["started_at"] = _now()

    results = []
    total   = len(image_bytes_list)

    for idx, img_bytes in enumerate(image_bytes_list):
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Run ML inference
            pred = predict(pil_img, segment=segment, age=age, panels=panels)

            # Run vision analysis (lightweight version — skip full annotation)
            vision = vision_analyse(pil_img, max_zones=4)

            item = {
                **pred,
                "vision_score":  vision.get("overall_vision_score", 0),
                "vision_zones":  len(vision.get("vision_zones", [])),
                "file_index":    idx,
                "status":        "ok",
            }
        except Exception as exc:
            item = {
                "file_index": idx,
                "status":     "error",
                "error":      str(exc),
                "is_damaged": False,
                "part":       "unknown",
                "damage":     "unknown",
                "severity":   "no_damage",
                "cost":       None,
            }

        results.append(item)

        with _JOBS_LOCK:
            JOBS[job_id]["completed"]    = idx + 1
            JOBS[job_id]["progress_pct"] = round((idx + 1) / total * 100, 1)
            JOBS[job_id]["results"]      = list(results)   # snapshot

    # Finalise
    summary = _build_summary(results)

    with _JOBS_LOCK:
        JOBS[job_id]["state"]       = "done"
        JOBS[job_id]["summary"]     = summary
        JOBS[job_id]["finished_at"] = _now()


def submit_batch(image_bytes_list: list[bytes],
                 segment: str = "sedan",
                 age: int     = 3,
                 panels: int  = 1) -> str:
    """
    Submit a batch job.

    Parameters
    ----------
    image_bytes_list : list of raw image bytes (read from request.files)
    segment          : vehicle segment string
    age              : vehicle age in years
    panels           : number of panels affected

    Returns
    -------
    job_id : str — use to poll /batch/<job_id>
    """
    job_id = uuid.uuid4().hex

    job = {
        "job_id":       job_id,
        "state":        "queued",
        "total":        len(image_bytes_list),
        "completed":    0,
        "progress_pct": 0.0,
        "results":      [],
        "summary":      None,
        "error":        None,
        "started_at":   _now(),
        "finished_at":  None,
    }

    with _JOBS_LOCK:
        JOBS[job_id] = job

    t = threading.Thread(
        target=_worker,
        args=(job_id, image_bytes_list, segment, age, panels),
        daemon=True,
        name=f"batch-{job_id[:8]}",
    )
    t.start()

    return job_id


def get_job_status(job_id: str) -> dict | None:
    """
    Get current status of a job.

    Returns
    -------
    dict snapshot (thread-safe copy) or None if job_id unknown.
    """
    with _JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        return dict(job)   # shallow copy is fine for JSON serialisation


def list_jobs(n: int = 20) -> list[dict]:
    """Return the n most recent jobs (lightweight — no results list)."""
    with _JOBS_LOCK:
        jobs = list(JOBS.values())
    jobs.sort(key=lambda j: j["started_at"], reverse=True)
    out = []
    for j in jobs[:n]:
        out.append({k: v for k, v in j.items() if k != "results"})
    return out