"""
Discovery tool: logged-out Instagram Reel discovery with Crawl4AI + yt-dlp metadata.

Pure functions, no LangGraph imports. Used by the graph's discover_step.

For each handle it:
  1. loads instagram.com/<handle>/reels/ logged out and parses the grid
     (shortcode, pinned, views, likes, comments),
  2. decides an account status: ok / partial / blocked / failed,
  3. fetches yt-dlp metadata (no download) for every non-pinned reel:
     upload date, duration and the FULL caption.

Main entry point:
    accounts = await discover_accounts(["digistreet.media", "digidir_"])

Each account dict:
    account, profile_url, status, http_status, login_wall_detected,
    reels_found, reels_with_views, note, reels[]

Each reel dict (same field names the ranking tool reads):
    shortcode, url, pinned, grid_views, grid_likes, grid_comments, grid_text,
    caption, metadata{upload_date, view_count, like_count, comment_count, duration, description}

Manual check (saves discovery_debug/discovery_tool.json):
    uv run python -m insta_analyser.tools.discovery digistreet.media digidir_
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

MAX_ACCOUNTS = 3
PAGE_TIMEOUT_MS = 60_000
OVERALL_TIMEOUT_S = 90
DELAY_BETWEEN_ACCOUNTS_S = 8
DELAY_BETWEEN_METADATA_S = 2
MAX_REELS_KEPT = 24
# Login-prompt text appears on every logged-out page, even when the full grid
# loads. If at least this many reels with view counts came back, the grid is
# treated as complete and the account is "ok" despite the markers.
FULL_GRID_MIN_REELS = 9

SHORTCODE_RE = re.compile(r"/(?:[A-Za-z0-9._]+/)?(?:reel|reels|p)/([A-Za-z0-9_-]{5,})/?")
ANCHOR_RE = re.compile(
    r'<a\b[^>]*href=["\']([^"\']*/(?:reel|p)/[A-Za-z0-9_-]{5,}/?[^"\']*)["\'][^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")
VIEW_RE = re.compile(r"View Count Icon\s*([\d.,]+\s*[KkMmBb]?)")
NUM_RE = re.compile(r"\d[\d.,]*\s*[KkMmBb]?")

LOGIN_WALL_MARKERS = [
    "log in to instagram",
    "login to instagram",
    "sign up to see photos",
    "log in to see",
    "/accounts/login",
    "create an account or log in",
]

# Scrolls a few times to load more of the grid. Instagram may ignore this when logged out.
SCROLL_JS = """
for (let i = 0; i < 4; i++) {
    window.scrollTo(0, document.body.scrollHeight);
    await new Promise(r => setTimeout(r, 1500));
}
"""


# ------------------------------------------------------------------ parsing ---

def normalise_handle(arg: str) -> tuple[str, str]:
    """Accept a handle, @handle or profile URL. Returns (handle, reels_tab_url)."""
    arg = arg.strip()
    if arg.startswith("http"):
        path = urlparse(arg).path.strip("/")
        handle = path.split("/")[0] if path else arg
    else:
        handle = arg.lstrip("@").strip("/")
    return handle, f"https://www.instagram.com/{handle}/reels/"


def parse_count(s: str):
    """'1,954' -> 1954, '15.2K' -> 15200, '1.2M' -> 1200000. None if unparseable."""
    s = s.strip().replace(",", "").replace(" ", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KkMmBb]?)", s)
    if not m:
        return None
    mult = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(m.group(2).lower(), 1)
    return int(float(m.group(1)) * mult)


def extract_reels(html: str) -> list[dict]:
    """Parse reel tiles from the profile grid HTML, newest first, de-duplicated."""
    seen = {}
    for href, inner in ANCHOR_RE.findall(html):
        m = SHORTCODE_RE.search(href)
        if not m:
            continue
        code = m.group(1)
        if code in seen:
            continue
        text = re.sub(r"\s+", " ", TAG_RE.sub(" ", inner)).strip()

        pinned = "Pinned post icon" in text
        vm = VIEW_RE.search(text)
        views = parse_count(vm.group(1)) if vm else None

        # likes and comments are the two numbers before the view icon; hidden likes = None
        before = text.split("Pinned post icon")[0].split("View Count Icon")[0]
        nums = NUM_RE.findall(before)
        likes = parse_count(nums[0]) if len(nums) >= 2 else None
        comments = parse_count(nums[1]) if len(nums) >= 2 else None

        seen[code] = {
            "shortcode": code,
            "url": f"https://www.instagram.com/reel/{code}/",
            "pinned": pinned,
            "grid_views": views,
            "grid_likes": likes,
            "grid_comments": comments,
            "grid_text": text[:80],
        }
    return list(seen.values())[:MAX_REELS_KEPT]


def detect_login_wall(html: str, markdown: str) -> bool:
    blob = (html[:200_000] + " " + markdown[:20_000]).lower()
    return any(marker in blob for marker in LOGIN_WALL_MARKERS)


def account_status(reels: list[dict], login_wall: bool) -> tuple[str, str]:
    """Decide (status, note) from what the grid returned."""
    with_views = sum(1 for r in reels if r["grid_views"] is not None)
    if not reels:
        return ("blocked" if login_wall else "failed",
                "no reel links found; check the saved .html to see what Instagram returned")
    if with_views >= FULL_GRID_MIN_REELS:
        note = (f"login prompt text present, but {with_views} reels with views loaded; treated as a full grid"
                if login_wall else "")
        return "ok", note
    if login_wall:
        return "partial", "reels found but a login prompt was also detected; grid may be truncated"
    return "ok", "" if with_views else "reels found but no view counts visible on the grid"


# ----------------------------------------------------------------- metadata ---

def fetch_metadata(reel_url: str) -> dict:
    """yt-dlp metadata only, no download. Blocking: call through asyncio.to_thread."""
    import yt_dlp

    opts = {"skip_download": True, "quiet": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(reel_url, download=False)
        return {
            "upload_date": info.get("upload_date"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "comment_count": info.get("comment_count"),
            "duration": info.get("duration"),
            "description": info.get("description") or "",  # full caption, needed for classification
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}


# ---------------------------------------------------------------- discovery ---

async def discover_account(crawler: AsyncWebCrawler, arg: str, debug_dir: Path | None = None) -> dict:
    """Crawl one profile's Reels tab. Never raises: problems end up in status and note."""
    handle, url = normalise_handle(arg)
    out = {
        "account": handle,
        "profile_url": url,
        "status": "failed",
        "http_status": None,
        "login_wall_detected": False,
        "reels_found": 0,
        "reels_with_views": 0,
        "note": "",
        "reels": [],
    }

    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=PAGE_TIMEOUT_MS,
        js_code=[SCROLL_JS],
        delay_before_return_html=3.0,
    )
    try:
        result = await asyncio.wait_for(crawler.arun(url=url, config=config), timeout=OVERALL_TIMEOUT_S)
    except asyncio.TimeoutError:
        out["note"] = f"timed out after {OVERALL_TIMEOUT_S}s"
        return out
    except Exception as exc:
        out["note"] = f"crawl error: {type(exc).__name__}: {exc}"
        return out

    out["http_status"] = getattr(result, "status_code", None)
    if not result.success:
        out["note"] = f"crawl unsuccessful: {getattr(result, 'error_message', '') or 'no message'}"
        return out

    html = result.html or ""
    md = getattr(result, "markdown", None)
    markdown = (getattr(md, "raw_markdown", None) or str(md)) if md is not None else ""

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / f"{handle}.html").write_text(html, encoding="utf-8")
        (debug_dir / f"{handle}.md").write_text(markdown, encoding="utf-8")

    reels = extract_reels(html)
    out["reels"] = reels
    out["reels_found"] = len(reels)
    out["reels_with_views"] = sum(1 for r in reels if r["grid_views"] is not None)
    out["login_wall_detected"] = detect_login_wall(html, markdown)
    out["status"], out["note"] = account_status(reels, out["login_wall_detected"])
    return out


async def add_metadata(account: dict) -> None:
    """Attach yt-dlp metadata and the full caption to every non-pinned reel, in place."""
    for r in account["reels"]:
        if r["pinned"]:
            continue
        meta = await asyncio.to_thread(fetch_metadata, r["url"])
        r["metadata"] = meta
        r["caption"] = meta.get("description", "")
        await asyncio.sleep(DELAY_BETWEEN_METADATA_S)


async def discover_accounts(
    handles: list[str],
    with_metadata: bool = True,
    debug_dir: Path | None = None,
) -> list[dict]:
    """Discover 1-MAX_ACCOUNTS accounts in sequence, with polite delays between them."""
    if not handles:
        raise ValueError("at least one handle is required")
    if len(handles) > MAX_ACCOUNTS:
        raise ValueError(f"at most {MAX_ACCOUNTS} accounts per run")

    accounts = []
    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        for i, handle in enumerate(handles):
            acct = await discover_account(crawler, handle, debug_dir)
            if with_metadata and acct["status"] in ("ok", "partial"):
                await add_metadata(acct)
            accounts.append(acct)
            if i < len(handles) - 1:
                await asyncio.sleep(DELAY_BETWEEN_ACCOUNTS_S)
    return accounts


# --------------------------------------------------------------- manual run ---

def _print_summary(accounts: list[dict]) -> None:
    for a in accounts:
        print(f"\n=== {a['account']} ===  status={a['status']}  reels={a['reels_found']}  "
              f"with views={a['reels_with_views']}")
        if a["note"]:
            print(f"note: {a['note']}")
        for r in a["reels"]:
            meta = r.get("metadata", {})
            cap = " ".join((r.get("caption") or "").split())[:60]
            print(f"  - {r['shortcode']:<13} views={r['grid_views']}  likes={r['grid_likes']}  "
                  f"pinned={r['pinned']}  date={meta.get('upload_date')}  caption={cap!r}")


if __name__ == "__main__":
    handles = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not handles:
        sys.exit("usage: python -m insta_analyser.tools.discovery handle1 [handle2 handle3] [--no-metadata]")
    debug = Path("discovery_debug")
    result = asyncio.run(discover_accounts(handles, "--no-metadata" not in sys.argv, debug))
    _print_summary(result)
    (debug / "discovery_tool.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {debug / 'discovery_tool.json'}")
