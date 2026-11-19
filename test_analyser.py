"""
Test the Instagram Reel Analyser end to end.

Three modes:
    uv run python test_analyser.py offline              # no network: saved discovery + saved transcripts,
                                                        # real tev1 + llama3.2 (about 30 s)
    uv run python test_analyser.py live                 # the real graph: crawl, download, Whisper (several minutes)
    uv run python test_analyser.py api                  # through the FastAPI job endpoints; start the server first:
                                                        #   uv run uvicorn insta_analyser.main:app --port 8002

Handles default to digistreet.media and digidir_; pass others after the mode:
    uv run python test_analyser.py live handle1 handle2

Offline mode needs discovery_debug/results.json and discovery_debug/transcripts/ from the old test scripts.

Outputs:
    discovery_debug/analyser_<mode>_result.json      the full API response
    discovery_debug/reports/educational_report.md    readable reports (+ .json with the same data)
    discovery_debug/reports/promotional_report.md
"""

import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from insta_analyser.analyser_schemas import AnalyseRequest, AnalyseResponse
from insta_analyser.report_files import write_report_files

DEBUG_DIR = Path("discovery_debug")
DEFAULT_HANDLES = ["digistreet.media", "digidir_"]
API_URL = "http://localhost:8002"
POLL_EVERY_S = 10


# ============================================================
# OFFLINE: replace discovery and transcription with saved files
# ============================================================

def patch_offline():
    import insta_analyser.analyser_graph as graph
    from insta_analyser.tools.transcription import decide_status

    results_path = DEBUG_DIR / "results.json"
    if not results_path.exists():
        sys.exit(f"{results_path} not found; run test_reel_discovery.py --metadata first, or use live mode.")

    async def saved_discovery(handles):
        data = json.loads(results_path.read_text(encoding="utf-8"))
        for acct in data:
            for r in acct["reels"]:
                r["caption"] = (r.get("metadata") or {}).get("description", "")
        found = [a for a in data if a["account"] in handles]
        missing = set(handles) - {a["account"] for a in found}
        if missing:
            print(f"(no saved discovery for {sorted(missing)}; they are left out)")
        return found

    def saved_transcripts(reels, config=None):
        out = []
        for r in reels:
            fields = {"transcript": "", "transcript_status": "empty", "transcript_notes": [],
                      "transcript_error": "", "language": None, "duration_s": None, "word_count": 0}
            path = DEBUG_DIR / "transcripts" / f"{r['shortcode']}.json"
            if path.exists():
                t = json.loads(path.read_text(encoding="utf-8"))
                notes = ["no_vad_pass" if n == "no_vad_check_for_hallucination" else n for n in t.get("notes") or []]
                fields.update(
                    transcript=t.get("text", ""), language=t.get("language"), duration_s=t.get("duration_s"),
                    word_count=t.get("word_count", 0), transcript_notes=notes,
                    transcript_status=decide_status(t.get("word_count", 0), t.get("words_per_minute"), notes),
                )
            out.append({**r, **fields})
        return out

    graph.discover_accounts = saved_discovery
    graph.transcribe_reels = saved_transcripts


# ============================================================
# RUNNERS
# ============================================================

def run_direct(handles):
    from insta_analyser.analyser_graph import run_analysis

    start = time.time()
    return asyncio.run(run_analysis(
        AnalyseRequest(handles=handles),
        on_step=lambda node: print(f"-- {node} done ({time.time() - start:.0f}s)"),
    ))


def _http(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {url} -> HTTP {e.code}: {e.read().decode()[:500]}")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach {url} ({e.reason}). Start the server: uv run uvicorn insta_analyser.main:app --port 8002")


def run_api(handles):
    print("health:", _http("GET", f"{API_URL}/health"))
    job = _http("POST", f"{API_URL}/analyse", {"handles": handles})
    print(f"job {job['job_id']} {job['status']}")
    start, last = time.time(), None
    while job["status"] in ("queued", "running"):
        time.sleep(POLL_EVERY_S)
        job = _http("GET", f"{API_URL}/analyse/{job['job_id']}")
        if (job["status"], job.get("step")) != last:
            last = (job["status"], job.get("step"))
            print(f"-- {job['status']}  last finished: {job.get('step') or '-'}  ({time.time() - start:.0f}s)")
    if job["status"] == "failed":
        sys.exit(f"job failed: {job['error']}")
    return AnalyseResponse.model_validate(job["result"])


# ============================================================
# OUTPUT
# ============================================================

def show(result: AnalyseResponse):
    print("\n" + "=" * 70)
    print("ACCOUNTS")
    print("=" * 70)
    for a in result.accounts:
        print(f"{a.account:<20} {a.status:<8} found={a.reels_found} ranked={a.reels_ranked} "
              f"educational={a.educational_kept} promotional={a.promotional_kept}")
        for n in a.notes:
            print(f"    note: {n}")

    print("\n" + "=" * 70)
    print("SELECTED REELS")
    print("=" * 70)
    for r in result.selected_reels:
        used = "*" if r.used_in_report else " "
        p = f"{r.label_probability:.2f}" if r.label_probability is not None else "-"
        print(f"{used} {r.account:<18} #{r.rank} {r.shortcode:<13} score={r.score:<6.2f} "
              f"speech={r.transcript_status or '-':<16} {'mixed' if len(r.categories) == 2 else r.label:<12} "
              f"p={p}  {','.join(r.flags)}")
    print("  (* = used in a report)")

    for report in (result.educational_report, result.promotional_report):
        print("\n" + "=" * 70)
        print(f"{report.category.upper()} REPORT  ({report.reels_analysed} reels, "
              f"accounts: {', '.join(report.accounts_covered) or 'none'})")
        print("=" * 70)
        for field in ("themes", "hook_patterns", "cta_patterns", "format_notes", "opportunities_for_linkup"):
            items = getattr(report, field)
            if items:
                print(f"{field}:")
                for item in items:
                    print(f"  - {item}")
        if report.content_structure:
            print(f"content_structure: {report.content_structure}")
        for ex in report.top_examples:
            print(f"example {ex.shortcode} ({ex.account}, score {ex.score}): {ex.why_it_worked}")
        for lim in report.limitations:
            print(f"limitation: {lim}")

    if result.errors:
        print("\nERRORS:")
        for e in result.errors:
            print(f"  - {e}")


def main():
    args = sys.argv[1:]
    if not args or args[0] not in ("offline", "live", "api"):
        sys.exit(__doc__)
    mode, handles = args[0], args[1:] or DEFAULT_HANDLES

    if mode == "offline":
        patch_offline()
    result = run_api(handles) if mode == "api" else run_direct(handles)

    show(result)
    out = DEBUG_DIR / f"analyser_{mode}_result.json"
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    for path in write_report_files(result, DEBUG_DIR / "reports"):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
