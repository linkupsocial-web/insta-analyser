"""
LangGraph pipeline for the Instagram Reel Analyser.

    discover_step -> rank_step -> transcribe_step -> classify_step -+-> report_educational_step -+-> assemble_step -> END
                         |                                          +-> report_promotional_step -+
                         +-- nothing to rank ---------------------------------------------------> assemble_step

Nodes are thin: each reads the state, calls a tool from insta_analyser/tools,
and returns only the keys it writes. The two report nodes run in parallel.

Run it:
    result = await run_analysis(AnalyseRequest(handles=["digistreet.media", "digidir_"]))
"""

import json
import operator
import re
from typing import Annotated, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ValidationError

from insta_analyser.analyser_models import get_classifier_config, get_model, get_transcription_config
from insta_analyser.analyser_prompts import (
    REEL_NOTES_SYSTEM_PROMPT,
    RETRY_PROMPT,
    insights_system_prompt,
    insights_user_prompt,
    reel_notes_user_prompt,
)
from insta_analyser.analyser_schemas import (
    AccountSummary,
    AnalyseRequest,
    AnalyseResponse,
    CategoryReport,
    ReelNotes,
    ReportInsights,
    SelectedReel,
    TopExample,
)
from insta_analyser.tools.classification import classify_reels
from insta_analyser.tools.discovery import discover_accounts
from insta_analyser.tools.ranking import rank_accounts
from insta_analyser.tools.transcription import transcribe_reels

CATEGORIES = ("educational", "promotional")
THIN_DATA_REELS = 3


# ============================================================
# JSON FROM THE LLM (fence stripping + validation + one retry)
# ============================================================

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json(text: str) -> dict:
    text = _FENCE_RE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start:end + 1])


def invoke_json(model, messages: list, schema: type[BaseModel], retries: int = 1) -> BaseModel:
    """Call the LLM and validate its JSON against schema. On failure, show it the error and retry."""
    messages = list(messages)
    for attempt in range(retries + 1):
        reply = model.invoke(messages)
        try:
            return schema.model_validate(parse_json(reply.content))
        except (ValueError, ValidationError) as exc:
            if attempt == retries:
                raise
            messages += [("assistant", reply.content), ("user", RETRY_PROMPT.format(error=str(exc)[:500]))]


# ============================================================
# STATE
# ============================================================

class AnalyserState(TypedDict, total=False):
    # input
    handles: List[str]
    pool_size: int
    report_top_k: int
    exclude_flagged: bool
    age_adjust: bool

    # written by the nodes, in order
    accounts: List[dict]             # discover_step
    rankings: List[dict]             # rank_step: per account {account, status, ranked, stats, notes}
    pool_reels: List[dict]           # rank_step: flat list, every account's top pool_size
    transcribed_reels: List[dict]    # transcribe_step
    classified_reels: List[dict]     # classify_step
    educational_report: CategoryReport
    promotional_report: CategoryReport
    educational_used: List[str]      # shortcodes that fed each report
    promotional_used: List[str]
    result: AnalyseResponse          # assemble_step

    # any node; the reducer merges lists, so the parallel report nodes can both add errors
    errors: Annotated[List[str], operator.add]


# ============================================================
# NODES 1-4: tools
# ============================================================

async def discover_step(state: AnalyserState):
    accounts = await discover_accounts(state["handles"])
    errors = [f"{a['account']}: discovery {a['status']} ({a['note']})"
              for a in accounts if a["status"] in ("blocked", "failed")]
    return {"accounts": accounts, "errors": errors}


def rank_step(state: AnalyserState):
    rankings = rank_accounts(
        state["accounts"],
        pool_size=state.get("pool_size", 8),
        exclude_flagged=state.get("exclude_flagged", False),
        age_adjust=state.get("age_adjust", True),
    )
    pool = [r for acct in rankings for r in acct["ranked"]]
    return {"rankings": rankings, "pool_reels": pool}


def route_after_rank(state: AnalyserState) -> str:
    return "transcribe_step" if state.get("pool_reels") else "assemble_step"


def transcribe_step(state: AnalyserState):
    reels = transcribe_reels(state["pool_reels"], get_transcription_config())
    errors = [f"{r['shortcode']}: {r['transcript_status']} ({r['transcript_error']})"
              for r in reels if r["transcript_error"]]
    return {"transcribed_reels": reels, "errors": errors}


def classify_step(state: AnalyserState):
    reels = classify_reels(state["transcribed_reels"], get_classifier_config())
    errors = [f"{r['shortcode']}: classification failed ({r['classify_error']})"
              for r in reels if r["classify_error"]]
    return {"classified_reels": reels, "errors": errors}


# ============================================================
# NODES 5-6: reports (shared implementation)
# ============================================================

def select_for_report(reels: List[dict], category: str, top_k: int) -> List[dict]:
    """Best top_k reels of this category per account (by rank), then best first overall.

    Mixed reels (categories = both) are picked by both reports.
    """
    picked = []
    for account in dict.fromkeys(r["account"] for r in reels):
        mine = sorted((r for r in reels if r["account"] == account and category in (r.get("categories") or [])),
                      key=lambda r: r["rank"])
        picked += mine[:top_k]
    return sorted(picked, key=lambda r: r["score"], reverse=True)


def _limitations(category: str, picked: List[dict], handles: List[str], notes_failed: int) -> List[str]:
    lim = []
    covered = {r["account"] for r in picked}
    for h in handles:
        if h not in covered:
            lim.append(f"No {category} reels found for {h}.")
    if 0 < len(picked) < THIN_DATA_REELS:
        lim.append(f"Only {len(picked)} {category} reel(s) analysed; treat patterns as early signals.")
    caption_only = sum(1 for r in picked if r.get("transcript_status") != "ok")
    if caption_only:
        lim.append(f"{caption_only} reel(s) had no usable speech and were judged from the caption only.")
    inflated = [r["shortcode"] for r in picked if "possibly_inflated" in r.get("flags", [])]
    if inflated:
        lim.append(f"Possibly inflated views: {', '.join(inflated)}.")
    unverified = [r["shortcode"] for r in picked if "unverified" in r.get("flags", [])]
    if unverified:
        lim.append(f"Likes hidden, views not cross-checked: {', '.join(unverified)}.")
    mixed = [r["shortcode"] for r in picked if len(r.get("categories") or []) == 2]
    if mixed:
        lim.append(f"Both educational and promotional, so also in the other report: {', '.join(mixed)}.")
    if notes_failed:
        lim.append(f"{notes_failed} reel(s) could not be summarised by the LLM and were left out of the patterns.")
    return lim


def _report_node(state: AnalyserState, category: str):
    handles = state["handles"]
    picked = select_for_report(state.get("classified_reels", []), category, state.get("report_top_k", 3))
    key = f"{category}_report"
    used_key = f"{category}_used"

    if not picked:
        return {key: CategoryReport.empty(category, _limitations(category, [], handles, 0)), used_key: []}

    errors = []
    model = get_model(temperature=0.3)

    # step 1: short notes per reel
    notes, why = [], {}
    for r in picked:
        try:
            n = invoke_json(model, [("system", REEL_NOTES_SYSTEM_PROMPT), ("user", reel_notes_user_prompt(r, category))],
                            ReelNotes)
        except Exception as exc:
            errors.append(f"{category} notes for {r['shortcode']} failed: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        why[r["shortcode"]] = n.why_it_worked
        notes.append({
            "account": r["account"],
            "shortcode": r["shortcode"],
            "score": round(r["score"], 2),
            "flags": r.get("flags", []),
            "duration_s": r.get("duration_s"),
            "language": r.get("language"),
            "has_speech": r.get("transcript_status") == "ok",
            "also_in_other_report": len(r.get("categories") or []) == 2,
            **n.model_dump(),
        })

    report = CategoryReport(
        category=category,
        reels_analysed=len(picked),
        accounts_covered=sorted({r["account"] for r in picked}),
        top_examples=[
            TopExample(
                shortcode=r["shortcode"], account=r["account"], url=r["url"], score=round(r["score"], 3),
                views=r.get("grid_views"), flags=r.get("flags", []), why_it_worked=why.get(r["shortcode"], ""),
            )
            for r in picked
        ],
        limitations=_limitations(category, picked, handles, len(picked) - len(notes)),
    )

    # step 2: patterns across the notes
    if notes:
        try:
            insights = invoke_json(
                get_model(temperature=0.4),
                [("system", insights_system_prompt(category)), ("user", insights_user_prompt(category, notes))],
                ReportInsights,
            )
            report = report.model_copy(update=insights.model_dump())
        except Exception as exc:
            errors.append(f"{category} insights failed: {type(exc).__name__}: {str(exc)[:200]}")
            report.limitations.append("Pattern summary failed; only the facts above are available.")

    return {key: report, used_key: [r["shortcode"] for r in picked], "errors": errors}


def report_educational_step(state: AnalyserState):
    return _report_node(state, "educational")


def report_promotional_step(state: AnalyserState):
    return _report_node(state, "promotional")


# ============================================================
# NODE 7: assemble the API response
# ============================================================

def assemble_step(state: AnalyserState):
    reels = state.get("classified_reels") or state.get("pool_reels") or []
    used = set(state.get("educational_used", [])) | set(state.get("promotional_used", []))
    rankings = {r["account"]: r for r in state.get("rankings", [])}

    accounts = []
    for a in state.get("accounts", []):
        mine = [r for r in reels if r.get("account") == a["account"]]
        accounts.append(AccountSummary(
            account=a["account"],
            status=a["status"],
            note=a.get("note", ""),
            reels_found=a.get("reels_found", 0),
            reels_ranked=len(rankings.get(a["account"], {}).get("ranked", [])),
            educational_kept=sum(1 for r in mine if "educational" in (r.get("categories") or [])),
            promotional_kept=sum(1 for r in mine if "promotional" in (r.get("categories") or [])),
            notes=rankings.get(a["account"], {}).get("notes", []),
        ))

    selected = [
        SelectedReel(
            shortcode=r["shortcode"], url=r["url"], account=r["account"], rank=r["rank"],
            score=round(r["score"], 4), score_basis=r["score_basis"], flags=r.get("flags", []),
            views=r.get("grid_views"), likes=r.get("grid_likes"), comments=r.get("grid_comments"),
            engagement_rate=r.get("eng_rate"), age_days=r.get("age_days"),
            caption=r.get("caption") or "", transcript=r.get("transcript") or "",
            transcript_status=r.get("transcript_status"), language=r.get("language"),
            label=r.get("label", "other"), categories=r.get("categories") or [],
            label_probability=r.get("label_probability"),
            has_cta=r.get("has_cta"), hook_score=r.get("hook_score"),
            classify_notes=r.get("classify_notes", []), used_in_report=r["shortcode"] in used,
        )
        for r in reels
    ]

    def report(category):
        return state.get(f"{category}_report") or CategoryReport.empty(
            category, ["No reels could be ranked, so nothing was classified."])

    return {"result": AnalyseResponse(
        accounts=accounts,
        selected_reels=selected,
        educational_report=report("educational"),
        promotional_report=report("promotional"),
        errors=state.get("errors", []),
    )}


# ============================================================
# GRAPH
# ============================================================

def build_graph():
    graph = StateGraph(AnalyserState)

    graph.add_node("discover_step", discover_step)
    graph.add_node("rank_step", rank_step)
    graph.add_node("transcribe_step", transcribe_step)
    graph.add_node("classify_step", classify_step)
    graph.add_node("report_educational_step", report_educational_step)
    graph.add_node("report_promotional_step", report_promotional_step)
    graph.add_node("assemble_step", assemble_step)

    graph.set_entry_point("discover_step")
    graph.add_edge("discover_step", "rank_step")
    graph.add_conditional_edges("rank_step", route_after_rank, ["transcribe_step", "assemble_step"])
    graph.add_edge("transcribe_step", "classify_step")
    graph.add_edge("classify_step", "report_educational_step")
    graph.add_edge("classify_step", "report_promotional_step")
    graph.add_edge(["report_educational_step", "report_promotional_step"], "assemble_step")
    graph.add_edge("assemble_step", END)

    return graph.compile()


pipeline = build_graph()


async def run_analysis(request: AnalyseRequest, on_step=None) -> AnalyseResponse:
    """Run the whole graph. on_step(node_name) is called as each node finishes (for job progress)."""
    state = {**request.model_dump(), "errors": []}
    final = {}
    async for update in pipeline.astream(state, stream_mode="updates"):
        for node, values in update.items():
            final.update(values or {})
            if on_step:
                on_step(node)
    return final["result"]
