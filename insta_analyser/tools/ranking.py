"""
Ranking tool: picks each account's best-performing Reels. No network, no LLM.

Pure functions, no LangGraph imports. Used by the graph's rank_step.
Input = the account dicts returned by tools.discovery.discover_accounts().

Rules:
 1. Drop pinned reels.
 2. If an upload date is known, drop reels older than RECENCY_DAYS.
    If not, keep the reel and mark recency as "grid position only".
 3. Per account, compute median views, median interactions (likes + comments)
    and median engagement rate ((likes + comments) / views).
 4. Flag "possibly_inflated" when a reel's engagement rate is below
    FLAG_RATIO x the account's median rate. Flag "unverified" when likes are hidden.
 5. Score, always relative to the account's own median:
      normal / unverified reels -> views / median views
      flagged reels             -> interactions / median interactions
    If every reel in the account has an upload date, views and interactions are
    first divided by max(age_days, MIN_EFFECTIVE_AGE_DAYS) ** AGE_EXPONENT, so
    older reels don't win just by being online longer.
 6. Drop reels under MIN_VIEWS, sort by score (engagement rate breaks ties),
    keep the top pool_size (default POOL_SIZE = 8). This pool goes on to
    transcription and classification; pruning happens there.

Flags detect suspicious reels. They cannot prove views were paid or boosted.

Manual check on saved discovery output:
    uv run python -m insta_analyser.tools.ranking discovery_debug/results.json
"""

import json
import statistics
import sys
from datetime import datetime, timedelta

POOL_SIZE = 8
MIN_VIEWS = 300
FLAG_RATIO = 1 / 3
RECENCY_DAYS = 90
MIN_REELS_TO_RANK = 3
RANKABLE_STATUSES = ("ok", "partial")

# Age adjustment. exponent 0.5 = damped: a reel 4x older needs 2x the views to tie.
# Reels younger than MIN_EFFECTIVE_AGE_DAYS are treated as that age, so a reel
# posted yesterday can't win on one lucky day.
AGE_EXPONENT = 0.5
MIN_EFFECTIVE_AGE_DAYS = 3

# Working fields removed from the output so downstream state stays small.
_INTERNAL_FIELDS = ("adj_views", "adj_interactions", "grid_text")


def _parse_date(reel: dict):
    raw = (reel.get("metadata") or {}).get("upload_date")
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw), "%Y%m%d")
    except ValueError:
        return None


def age_factor(age_days: int) -> float:
    """Divisor that puts reels of different ages on a comparable footing."""
    return max(age_days, MIN_EFFECTIVE_AGE_DAYS) ** AGE_EXPONENT


def rank_account(
    reels: list[dict],
    pool_size: int = POOL_SIZE,
    exclude_flagged: bool = False,
    now: datetime | None = None,
    age_adjust: bool = True,
) -> dict:
    """Rank one account's reels. Returns {"ranked": [...], "stats": {...}, "notes": [...]}."""
    now = now or datetime.now()
    notes = []

    # 1. pinned
    pool = [dict(r) for r in reels if not r.get("pinned", False)]

    # 2. recency
    kept = []
    for r in pool:
        d = _parse_date(r)
        if d is None:
            r["recency"] = "grid position only"
            r["age_days"] = None
            kept.append(r)
        elif now - d <= timedelta(days=RECENCY_DAYS):
            r["age_days"] = (now - d).days
            r["recency"] = f"{r['age_days']}d old"
            kept.append(r)
        else:
            notes.append(f"dropped {r['shortcode']}: older than {RECENCY_DAYS} days")
    pool = [r for r in kept if r.get("grid_views")]

    if len(pool) < MIN_REELS_TO_RANK:
        return {"ranked": [], "stats": {}, "notes": notes + ["too few reels to rank"]}

    # 2b. age adjustment: all-or-nothing per account, so dated and undated
    # reels are never scored on different scales
    undated = sum(1 for r in pool if r["age_days"] is None)
    use_age = age_adjust and undated == 0
    if age_adjust and undated:
        notes.append(f"age adjustment skipped: {undated} reel(s) have no upload date, so raw views were used")
    for r in pool:
        r["age_adjusted"] = use_age
        r["adj_views"] = r["grid_views"] / age_factor(r["age_days"]) if use_age else r["grid_views"]

    # 3. account medians (engagement rate stays raw: it's a ratio, so age cancels out)
    median_views = statistics.median(r["grid_views"] for r in pool)
    median_adj_views = statistics.median(r["adj_views"] for r in pool)
    with_likes = [r for r in pool if r.get("grid_likes") is not None]
    for r in with_likes:
        r["interactions"] = r["grid_likes"] + (r.get("grid_comments") or 0)
        r["eng_rate"] = r["interactions"] / r["grid_views"]
        r["adj_interactions"] = r["interactions"] / age_factor(r["age_days"]) if use_age else r["interactions"]
    median_rate = statistics.median(r["eng_rate"] for r in with_likes) if with_likes else None
    median_adj_inter = statistics.median(r["adj_interactions"] for r in with_likes) if with_likes else None

    # 4 + 5. flags and scores
    for r in pool:
        r["flags"] = []
        if r.get("grid_likes") is None:
            r["flags"].append("unverified")
            r["score"] = r["adj_views"] / median_adj_views
            r["score_basis"] = "views"
            continue
        if median_rate and r["eng_rate"] < median_rate * FLAG_RATIO:
            r["flags"].append("possibly_inflated")
            r["score"] = (r["adj_interactions"] / median_adj_inter) if median_adj_inter else 0.0
            r["score_basis"] = "interactions"
        else:
            r["score"] = r["adj_views"] / median_adj_views
            r["score_basis"] = "views"

    # 6. floor, optional exclusion, sort, pool
    eligible = [r for r in pool if r["grid_views"] >= MIN_VIEWS]
    if exclude_flagged:
        eligible = [r for r in eligible if "possibly_inflated" not in r["flags"]]
    eligible.sort(key=lambda r: (r["score"], r.get("eng_rate") or 0), reverse=True)

    ranked = []
    for i, r in enumerate(eligible[:pool_size], 1):
        for k in _INTERNAL_FIELDS:
            r.pop(k, None)
        r["rank"] = i
        ranked.append(r)

    return {
        "ranked": ranked,
        "stats": {
            "median_views": median_views,
            "median_eng_rate": median_rate,
            "reels_considered": len(pool),
            "age_adjusted": use_age,
        },
        "notes": notes,
    }


def rank_accounts(
    accounts: list[dict],
    pool_size: int = POOL_SIZE,
    exclude_flagged: bool = False,
    now: datetime | None = None,
    age_adjust: bool = True,
) -> list[dict]:
    """Rank every discovered account.

    Returns one dict per account: {account, status, ranked, stats, notes}.
    Each ranked reel also carries its "account" name, so later steps can work
    on a flat list of reels. Blocked or failed accounts are kept with an
    empty ranking and a note, so the run continues.
    """
    results = []
    for acct in accounts:
        name = acct["account"]
        if acct.get("status") not in RANKABLE_STATUSES:
            results.append({
                "account": name,
                "status": acct.get("status"),
                "ranked": [],
                "stats": {},
                "notes": [f"skipped: discovery status {acct.get('status')} ({acct.get('note') or 'no note'})"],
            })
            continue
        res = rank_account(acct.get("reels") or [], pool_size, exclude_flagged, now, age_adjust)
        for r in res["ranked"]:
            r["account"] = name
        results.append({"account": name, "status": acct["status"], **res})
    return results


# --------------------------------------------------------------- manual run ---

def _print_results(results: list[dict]) -> None:
    for res in results:
        print(f"\n=== {res['account']} ===  status={res['status']}")
        s = res["stats"]
        if s:
            rate = f"{s['median_eng_rate']:.2%}" if s["median_eng_rate"] else "n/a"
            scoring = "age-adjusted" if s["age_adjusted"] else "raw views"
            print(f"considered: {s['reels_considered']}  median views: {s['median_views']:.0f}  "
                  f"median engagement: {rate}  scoring: {scoring}")
        for n in res["notes"]:
            print(f"note: {n}")
        for r in res["ranked"]:
            rate = f"{r['eng_rate']:.2%}" if r.get("eng_rate") is not None else "n/a"
            print(f"  {r['rank']}. {r['shortcode']:<13} views={r['grid_views']:<7} eng={rate:<6} "
                  f"score={r['score']:.2f} ({r['score_basis']})  flags={','.join(r['flags']) or '-'}  "
                  f"recency={r['recency']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m insta_analyser.tools.ranking <discovery json> [--pool N] [--exclude-flagged]")
    data = json.loads(open(sys.argv[1], encoding="utf-8").read())
    pool = int(sys.argv[sys.argv.index("--pool") + 1]) if "--pool" in sys.argv else POOL_SIZE
    _print_results(rank_accounts(data, pool_size=pool, exclude_flagged="--exclude-flagged" in sys.argv))
