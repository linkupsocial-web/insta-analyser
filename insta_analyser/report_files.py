"""
Report files: writes the educational and promotional reports to disk.

For each category, two files:
    <category>_report.md     readable report: patterns, top examples, limitations,
                             and an appendix with each reel's caption and transcript
    <category>_report.json   the same report as data (CategoryReport + the reels it used)

Used by test_analyser.py (into discovery_debug/reports/) and by the API
(into reports/<job_id>/).
"""

import json
from datetime import datetime
from pathlib import Path

from insta_analyser.analyser_schemas import AnalyseResponse, CategoryReport, SelectedReel

CATEGORIES = ("educational", "promotional")
TRANSCRIPT_EXCERPT_CHARS = 600


def _bullets(items: list[str]) -> list[str]:
    return [f"- {i}" for i in items] if items else ["- _None found._"]


def _md_cell(text) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def result_report(result: AnalyseResponse, category: str) -> CategoryReport:
    return result.educational_report if category == "educational" else result.promotional_report


def reels_for(result: AnalyseResponse, category: str) -> list[SelectedReel]:
    """The reels that fed this category's report, best score first."""
    codes = {e.shortcode for e in result_report(result, category).top_examples}
    return sorted((r for r in result.selected_reels if r.shortcode in codes), key=lambda r: r.score, reverse=True)


def render_markdown(result: AnalyseResponse, category: str, generated_at: datetime | None = None) -> str:
    report = result_report(result, category)
    reels = reels_for(result, category)
    when = (generated_at or datetime.now()).strftime("%d %B %Y, %H:%M")
    handles = ", ".join(a.account for a in result.accounts) or "none"

    lines = [
        f"# {category.title()} Report: Competitor Instagram Reels",
        "",
        f"*Generated {when} · Accounts analysed: {handles}*",
        "",
        f"**{report.reels_analysed} reel(s) analysed** from "
        f"{', '.join(report.accounts_covered) or 'no accounts'}.",
        "",
    ]

    if report.reels_analysed == 0:
        lines += ["## Limitations", "", *_bullets(report.limitations), ""]
        return "\n".join(lines)

    lines += [
        "## Themes", "", *_bullets(report.themes), "",
        "## Hook patterns", "", *_bullets(report.hook_patterns), "",
        "## Content structure", "", report.content_structure or "_Not identified._", "",
        "## Calls to action", "", *_bullets(report.cta_patterns), "",
        "## Format notes", "", *_bullets(report.format_notes), "",
        "## Opportunities for Linkup", "", *_bullets(report.opportunities_for_linkup), "",
        "## Top examples", "",
        "| Reel | Account | Score | Views | Flags | Why it worked |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for ex in report.top_examples:
        views = f"{ex.views:,}" if ex.views is not None else "-"
        lines.append(
            f"| [{ex.shortcode}]({ex.url}) | {_md_cell(ex.account)} | {ex.score:.2f} | {views} | "
            f"{_md_cell(', '.join(ex.flags) or '-')} | {_md_cell(ex.why_it_worked)} |"
        )
    lines += ["", "## Limitations", "", *_bullets(report.limitations), ""]

    lines += ["## Appendix: reels used", ""]
    for r in reels:
        mixed = " · also in the other report" if len(r.categories) == 2 else ""
        transcript = " ".join(r.transcript.split())
        if r.transcript_status != "ok":
            transcript = f"_No usable speech ({r.transcript_status})._"
        elif len(transcript) > TRANSCRIPT_EXCERPT_CHARS:
            transcript = transcript[:TRANSCRIPT_EXCERPT_CHARS] + " …"
        lines += [
            f"### [{r.shortcode}]({r.url}) · {r.account} · rank {r.rank}{mixed}",
            "",
            f"Score {r.score:.2f} ({r.score_basis}) · views {r.views if r.views is not None else '-'} · "
            f"language {r.language or '-'} · label probability "
            f"{r.label_probability if r.label_probability is not None else '-'}",
            "",
            f"**Caption:** {' '.join(r.caption.split()) or '_none_'}",
            "",
            f"**Transcript:** {transcript}",
            "",
        ]
    return "\n".join(lines)


def write_report_files(result: AnalyseResponse, out_dir: Path) -> list[Path]:
    """Write <category>_report.md and .json for both categories. Returns the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    written = []
    for category in CATEGORIES:
        md_path = out_dir / f"{category}_report.md"
        md_path.write_text(render_markdown(result, category, now), encoding="utf-8")

        json_path = out_dir / f"{category}_report.json"
        payload = {
            "generated_at": now.isoformat(timespec="seconds"),
            "report": result_report(result, category).model_dump(),
            "reels": [r.model_dump() for r in reels_for(result, category)],
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        written += [md_path, json_path]
    return written
