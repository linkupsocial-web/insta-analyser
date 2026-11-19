"""
Classification tool: transcribed Reels -> educational / promotional / other.

Pure functions, no LangGraph imports. Used by the graph's classify_step.
Uses a local System One decision model (tev1) through Ollama: it returns
probabilities, not text, so there is nothing to parse.

Setup:
    ollama pull tev1:0.8b          (Ollama >= 0.35)
    uv add ollama

Main entry point:
    reels = classify_reels(transcribed_reels, ClassificationConfig())

State sent per reel (a dict, so captions and transcripts stay data, never instructions):
    caption            full caption, hashtags included
    transcript         only when transcript_status is "ok", otherwise ""
    transcript_status  so the model knows when it is judging from the caption alone
    duration_s

Fields added to each reel:
    label                  "educational" / "promotional" / "other" (the stronger one for a mixed reel)
    categories             reports the reel goes into: [], [one], or both for a mixed reel
    kept                   True when categories is not empty
    label_probability      probability of the chosen category
    category_probabilities {educational, promotional, other}
    fits_marketing         probability the reel is marketing content at all (companion check)
    has_cta                probability the reel asks viewers to act (follow, DM, book, buy...)
    hook_score             0-3 (Weak, Okay, Strong, Irresistible), probability weighted
    classify_notes         e.g. "caption_only", "mixed", "low_probability", "not_marketing", "order_unstable"
    classify_error         set if the model call failed (the reel is then labelled "other")

Pruning rule (in code, not in the model), in this order:
    1. fits_marketing < min_fits                                  -> other
    2. educational + promotional >= mixed_total (0.8) and the
       smaller of the two >= mixed_min (0.35)                     -> mixed: both reports
    3. top category is educational or promotional and its
       probability >= min_probability (0.6)                       -> that one report
    4. anything else                                              -> other

Manual check on saved discovery + transcript data:
    uv run python -m insta_analyser.tools.classification
"""

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "tev1:0.8b"
MIN_PROBABILITY = 0.6
MIN_FITS = 0.5
# A reel that both teaches and sells: most probability on the two marketing
# categories, and a real share on each. It goes into both reports.
MIXED_TOTAL = 0.8
MIXED_MIN = 0.35
MAX_TRANSCRIPT_CHARS = 4000
MAX_CAPTION_CHARS = 2200  # Instagram's own caption limit

KEEP_LABELS = ("educational", "promotional")

CATEGORY_INSTRUCTIONS = (
    "What is the main purpose of this Instagram reel from a marketing agency or creator? "
    "Judge from the caption and, when present, the spoken transcript."
)
CATEGORIES = {
    "educational": (
        "Teaches something: marketing, branding, business or social media tips, how-tos, "
        "explanations, mistakes to avoid, or lessons from a case study."
    ),
    "promotional": (
        "Sells or advertises the creator's own agency, service, offer, course or client results, "
        "or mainly asks viewers to buy, book, enquire or contact them."
    ),
    "other": (
        "Memes, skits, comedy, trends, festival greetings, culture, music, office fun or personal "
        "content with no lesson and no offer."
    ),
}

QUESTIONS = {
    "category": {"type": "choice", "instructions": CATEGORY_INSTRUCTIONS, "criteria": CATEGORIES},
    "fits_marketing": {
        "type": "noul",
        "instructions": "Is this reel educational or promotional marketing content?",
        "criteria": {
            "true": "It teaches something useful or promotes a business, service or offer.",
            "false": "It is entertainment, a meme, a greeting or personal content.",
        },
    },
    "has_cta": {
        "type": "noul",
        "instructions": "Does the reel ask viewers to take an action?",
        "criteria": {
            "true": "It asks viewers to follow, comment, save, share, DM, call, book, visit a link or buy.",
            "false": "It asks viewers to do nothing.",
        },
    },
    "hook": {
        "type": "score",
        "instructions": "How strongly does the opening line make someone stop scrolling?",
        "criteria": ["Weak", "Okay", "Strong", "Irresistible"],
    },
}


@dataclass
class ClassificationConfig:
    model: str = DEFAULT_MODEL
    host: str | None = None              # None = OLLAMA_HOST or http://localhost:11434
    min_probability: float = MIN_PROBABILITY
    min_fits: float = MIN_FITS
    mixed_total: float = MIXED_TOTAL
    mixed_min: float = MIXED_MIN
    stability_runs: int = 0              # >0: re-ask with shuffled options and flag unstable labels


# ------------------------------------------------------------------- state ---

def reel_caption(reel: dict) -> str:
    cap = reel.get("caption")
    if cap is None:
        cap = (reel.get("metadata") or {}).get("description") or ""
    return cap[:MAX_CAPTION_CHARS]


def build_state(reel: dict) -> dict:
    """What the model sees for one reel. Metrics are left out on purpose: this is about content."""
    status = reel.get("transcript_status") or "missing"
    transcript = (reel.get("transcript") or "") if status == "ok" else ""
    return {
        "caption": reel_caption(reel),
        "transcript": transcript[:MAX_TRANSCRIPT_CHARS],
        "transcript_status": status,
        "duration_s": reel.get("duration_s"),
    }


# ---------------------------------------------------------------- decision ---

def decide(probs: dict, fits: float, config: ClassificationConfig) -> tuple[str, float, list[str], list[str]]:
    """Apply the pruning rule. Returns (label, label_probability, categories, notes).

    categories = the reports this reel goes into:
      []                              other / pruned
      ["educational"] or ["promotional"]
      ["educational", "promotional"]  mixed: it teaches AND sells, so it feeds both reports
    label is the stronger of the two for a mixed reel.
    """
    edu, promo = probs.get("educational", 0.0), probs.get("promotional", 0.0)
    marketing = edu + promo
    top = max(probs, key=probs.get)
    notes = []

    if fits < config.min_fits:
        return "other", probs[top], [], ["not_marketing"]

    # mixed: most of the probability is on marketing and it is split between the two
    if marketing >= config.mixed_total and min(edu, promo) >= config.mixed_min:
        label = "educational" if edu >= promo else "promotional"
        return label, probs[label], ["educational", "promotional"], ["mixed"]

    if top in KEEP_LABELS and probs[top] >= config.min_probability:
        return top, probs[top], [top], notes

    if top in KEEP_LABELS:
        notes.append("low_probability")
    return "other", probs[top], [], notes


def _client(config: ClassificationConfig):
    import ollama

    return ollama.Client(host=config.host) if config.host else ollama.Client()


def _ask(client, model: str, state: dict, questions: dict):
    return client.systemone(model=model, state=state, questions=questions).answers


def _shuffled_category_question() -> dict:
    items = list(CATEGORIES.items())
    random.shuffle(items)
    return {"category": {**QUESTIONS["category"], "criteria": dict(items)}}


def classify_reel(reel: dict, config: ClassificationConfig, client=None) -> dict:
    """Classify one reel. Returns only the classification fields. Never raises."""
    out = {
        "label": "other",
        "categories": [],
        "kept": False,
        "label_probability": None,
        "category_probabilities": {},
        "fits_marketing": None,
        "has_cta": None,
        "hook_score": None,
        "classify_notes": [],
        "classify_error": "",
    }
    state = build_state(reel)
    if not state["caption"].strip() and not state["transcript"].strip():
        out["classify_notes"].append("no_content")
        return out
    if not state["transcript"]:
        out["classify_notes"].append("caption_only")

    client = client or _client(config)
    try:
        a = _ask(client, config.model, state, QUESTIONS)
    except Exception as exc:
        out["classify_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return out

    probs = {k: round(v, 4) for k, v in a["category"].probabilities.items()}
    fits = a["fits_marketing"].noul
    label, p, categories, notes = decide(probs, fits, config)
    out.update({
        "label": label,
        "categories": categories,
        "kept": bool(categories),
        "label_probability": round(p, 4),
        "category_probabilities": probs,
        "fits_marketing": round(fits, 4),
        "has_cta": round(a["has_cta"].noul, 4),
        "hook_score": round(a["hook"].score, 3),
    })
    out["classify_notes"].extend(notes)

    # optional: the answer should not depend on the order the options are listed in
    if config.stability_runs:
        top = max(probs, key=probs.get)
        try:
            picks = {max(r.probabilities, key=r.probabilities.get) for r in (
                _ask(client, config.model, state, _shuffled_category_question())["category"]
                for _ in range(config.stability_runs)
            )}
        except Exception as exc:
            out["classify_notes"].append(f"stability_check_failed: {type(exc).__name__}")
        else:
            if picks != {top}:
                out["classify_notes"].append("order_unstable")
    return out


def classify_reels(reels: list[dict], config: ClassificationConfig | None = None, log=print) -> list[dict]:
    """Classify every reel. Returns new reel dicts with the classification fields added."""
    config = config or ClassificationConfig()
    client = _client(config)
    results = []
    for i, reel in enumerate(reels, 1):
        fields = classify_reel(reel, config, client)
        results.append({**reel, **fields})
        if log:
            probs = fields["category_probabilities"]
            pstr = " ".join(f"{k[:4]}={v:.2f}" for k, v in probs.items()) if probs else "-"
            log(f"[{i}/{len(reels)}] {reel.get('account', '')} {reel['shortcode']}: {fields['label']} "
                f"({pstr}, fits={fields['fits_marketing']})"
                + (f" [{', '.join(fields['classify_notes'])}]" if fields["classify_notes"] else "")
                + (f" ERROR {fields['classify_error']}" if fields["classify_error"] else ""))
    return results


def split_by_label(reels: list[dict]) -> dict[str, list[dict]]:
    """Group classified reels: {"educational": [...], "promotional": [...], "other": [...]}.

    A mixed reel appears in both the educational and the promotional group.
    """
    groups = {"educational": [], "promotional": [], "other": []}
    for r in reels:
        cats = r.get("categories") or []
        for c in cats:
            groups[c].append(r)
        if not cats:
            groups["other"].append(r)
    return groups


# --------------------------------------------------------------- manual run ---

def _saved_reels(debug_dir: Path) -> list[dict]:
    """Non-pinned reels from discovery_debug/results.json, joined with saved transcripts if any."""
    reels = []
    for acct in json.loads((debug_dir / "results.json").read_text(encoding="utf-8")):
        for r in acct["reels"]:
            if r.get("pinned"):
                continue
            r = {**r, "account": acct["account"]}
            tpath = debug_dir / "transcripts" / f"{r['shortcode']}.json"
            if tpath.exists():
                from insta_analyser.tools.transcription import decide_status

                t = json.loads(tpath.read_text(encoding="utf-8"))
                # files from the old transcribe_reels.py: re-derive the status with the new rules
                notes = ["no_vad_pass" if n == "no_vad_check_for_hallucination" else n
                         for n in t.get("notes") or []]
                r["transcript"] = t.get("text", "")
                r["transcript_status"] = decide_status(t.get("word_count", 0), t.get("words_per_minute"), notes)
                r["duration_s"] = t.get("duration_s")
            reels.append(r)
    return reels


if __name__ == "__main__":
    debug = Path("discovery_debug")
    runs = int(sys.argv[sys.argv.index("--stability") + 1]) if "--stability" in sys.argv else 0
    classified = classify_reels(_saved_reels(debug), ClassificationConfig(stability_runs=runs))
    groups = split_by_label(classified)
    print("\n" + ", ".join(f"{k}: {len(v)}" for k, v in groups.items()))
