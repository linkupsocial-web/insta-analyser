"""
Schemas for the Instagram Reel Analyser (discover -> rank -> transcribe -> classify -> reports).

Three groups:
  - API input / output        what n8n sends and receives
  - LLM outputs               what the report nodes ask the LLM for (validated, then retried once on failure)
  - Report                    LLM insights + facts filled in by code

The LLM schemas are deliberately flat (strings and short lists of strings) and
forgiving (a single string becomes a one-item list, long lists are trimmed),
because small local models like llama3.2 get nested JSON wrong more often.
"""

import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Category = Literal["educational", "promotional"]
HANDLE_RE = re.compile(r"^[A-Za-z0-9._]{1,30}$")

MAX_LIST_ITEMS = 5


def _as_short_list(value):
    """LLM tolerance: None -> [], "text" -> ["text"], drop blanks, keep at most MAX_LIST_ITEMS."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, list):
        items = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return items[:MAX_LIST_ITEMS]
    return value


# ============================================================
# API INPUT
# ============================================================

class AnalyseRequest(BaseModel):
    handles: List[str] = Field(..., min_length=1, max_length=3,
                               description="1-3 public competitor accounts: handle, @handle or profile URL")
    pool_size: int = Field(8, ge=1, le=12, description="Reels per account passed from ranking to transcription")
    report_top_k: int = Field(3, ge=1, le=8, description="Reels per category per account used in each report")
    exclude_flagged: bool = Field(False, description="Drop possibly_inflated reels instead of keeping them labelled")
    age_adjust: bool = Field(True, description="Divide views by age^0.5 when every reel has an upload date")

    @field_validator("handles")
    @classmethod
    def normalise_handles(cls, handles):
        out = []
        for h in handles:
            h = h.strip()
            if h.startswith("http"):
                h = h.split("instagram.com/")[-1]
            h = h.lstrip("@").strip("/").split("/")[0].split("?")[0]
            if not HANDLE_RE.match(h):
                raise ValueError(f"not a valid Instagram handle: {h!r}")
            out.append(h)
        if len(set(out)) != len(out):
            raise ValueError("duplicate handles")
        return out


# ============================================================
# LLM OUTPUT 1: per-reel notes (one short call per reel)
# ============================================================

class ReelNotes(BaseModel):
    topic: str = Field(..., description="What the reel is about, in one short phrase")
    hook: str = Field(..., description="How the first seconds grab attention, one sentence")
    structure: str = Field(..., description="The flow of the reel, e.g. 'hook -> 3 tips -> CTA'")
    cta: str = Field("", description="What viewers are asked to do; empty if nothing")
    why_it_worked: str = Field(..., description="One sentence on why this reel performed well")

    @field_validator("topic", "hook", "structure", "cta", "why_it_worked", mode="before")
    @classmethod
    def flatten(cls, v):
        if isinstance(v, list):
            v = "; ".join(str(x) for x in v if x)
        return "" if v is None else str(v).strip()


# ============================================================
# LLM OUTPUT 2: category insights (one call over the per-reel notes)
# ============================================================

class ReportInsights(BaseModel):
    themes: List[str] = Field(..., min_length=1, description="Topics the competitors teach or sell")
    hook_patterns: List[str] = Field(..., min_length=1, description="Recurring ways the reels open")
    content_structure: str = Field(..., description="The typical flow of these reels")
    cta_patterns: List[str] = Field(default_factory=list, description="What viewers are asked to do")
    format_notes: List[str] = Field(default_factory=list, description="Length, Hindi/English mix, on-screen text use")
    opportunities_for_linkup: List[str] = Field(..., min_length=1,
                                                description="Gaps or angles Linkup Web, Social, Legal or Finserv could use")

    @field_validator("themes", "hook_patterns", "cta_patterns", "format_notes", "opportunities_for_linkup",
                     mode="before")
    @classmethod
    def short_list(cls, v):
        return _as_short_list(v)

    @field_validator("content_structure", mode="before")
    @classmethod
    def flatten(cls, v):
        if isinstance(v, list):
            v = " -> ".join(str(x) for x in v if x)
        return "" if v is None else str(v).strip()


# ============================================================
# REPORT (code facts + LLM insights)
# ============================================================

class TopExample(BaseModel):
    shortcode: str
    account: str
    url: str
    score: float
    views: Optional[int] = None
    flags: List[str] = Field(default_factory=list)
    why_it_worked: str = ""


class CategoryReport(BaseModel):
    category: Category
    # filled by code
    reels_analysed: int = 0
    accounts_covered: List[str] = Field(default_factory=list)
    top_examples: List[TopExample] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    # filled by the LLM (empty when there were no reels)
    themes: List[str] = Field(default_factory=list)
    hook_patterns: List[str] = Field(default_factory=list)
    content_structure: str = ""
    cta_patterns: List[str] = Field(default_factory=list)
    format_notes: List[str] = Field(default_factory=list)
    opportunities_for_linkup: List[str] = Field(default_factory=list)

    @classmethod
    def empty(cls, category: str, limitations: List[str]) -> "CategoryReport":
        """Report for a category with no reels: same shape, no invented patterns."""
        return cls(category=category, limitations=limitations)


# ============================================================
# API OUTPUT
# ============================================================

class AccountSummary(BaseModel):
    account: str
    status: str                     # ok / partial / blocked / failed
    note: str = ""
    reels_found: int = 0
    reels_ranked: int = 0
    educational_kept: int = 0
    promotional_kept: int = 0
    notes: List[str] = Field(default_factory=list)


class SelectedReel(BaseModel):
    """One reel that went through transcription and classification."""
    shortcode: str
    url: str
    account: str
    rank: int
    score: float
    score_basis: str
    flags: List[str] = Field(default_factory=list)
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    engagement_rate: Optional[float] = None
    age_days: Optional[int] = None
    caption: str = ""
    transcript: str = ""
    transcript_status: Optional[str] = None
    language: Optional[str] = None
    label: str = "other"
    categories: List[str] = Field(default_factory=list)  # reports it feeds; both for a mixed reel
    label_probability: Optional[float] = None
    has_cta: Optional[float] = None
    hook_score: Optional[float] = None
    classify_notes: List[str] = Field(default_factory=list)
    used_in_report: bool = False


class AnalyseResponse(BaseModel):
    accounts: List[AccountSummary]
    selected_reels: List[SelectedReel]
    educational_report: CategoryReport
    promotional_report: CategoryReport
    errors: List[str] = Field(default_factory=list)


# ============================================================
# BACKGROUND JOB (the run takes minutes, so n8n polls)
# ============================================================

class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    step: Optional[str] = None      # the last node that finished (None until the first one does)
    result: Optional[AnalyseResponse] = None
    error: Optional[str] = None
