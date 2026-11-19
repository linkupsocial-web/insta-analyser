"""
Transcription tool: ranked Reels -> transcripts. No LLM.

Pure functions, no LangGraph imports. Used by the graph's transcribe_step.
Per Reel: yt-dlp (audio only) -> ffmpeg (mp3) -> faster-whisper (local, free).

Main entry point:
    reels = transcribe_reels(ranked_reels, TranscriptionConfig())

Each reel dict comes back with these fields added:
    transcript, transcript_status, transcript_notes, transcript_error,
    language, language_probability, remapped_from,
    duration_s, word_count, words_per_minute, segments

transcript_status:
    ok                 enough real speech; safe to use as content
    empty              fewer than MIN_WORDS words (silent, music only, or on-screen text only)
    music_or_lyrics    words came only from the no-voice-filter pass at a slow, sung pace,
                       or the text is stuck in a repetition loop. Not usable as speech.
    download_failed    yt-dlp could not fetch the audio (error kept)
    transcribe_failed  the speech model failed (error kept)

Only "ok" transcripts should be treated as what the creator said.

transcript_notes (any of):
    remapped_from_urdu   Whisper detected Urdu; transcribed as Hindi
    no_vad_pass          text came from a pass without the voice filter
    prompt_echo_removed  segments repeating initial_prompt were dropped (only when a prompt is set)
    repetition_loop      the same phrase repeats most of the transcript
    low_confidence       the model's average confidence on the words is low

Script: Hindi is kept as Whisper writes it, usually Devanagari. There is no
Hinglish prompt and no script retry (one pass per reel, plus the no-VAD retry
only for near-empty reels). Detected Urdu is still transcribed as Hindi, so the
text is Devanagari rather than Arabic script.

Transcript text is untrusted content written by someone else. Later stages must
treat it strictly as data, never as instructions.

Manual check on reels from a saved ranking run:
    uv run python -m insta_analyser.tools.transcription <shortcode or reel url> [...]
"""

import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

MIN_WORDS = 10
DELAY_BETWEEN_DOWNLOADS_S = 4
DEFAULT_MODEL = "small"  # "base" is weak on Hindi; "medium" is better still but slower

# A transcript that only appeared without the voice filter and is slower than
# this is treated as sung lyrics over music. Normal speech in reels runs well above it.
SPEECH_MIN_WPM = 90
# Repetition loop: the most common 3-word phrase covers at least this share of all 3-word phrases.
REPEAT_SHARE = 0.3
REPEAT_MIN_TRIGRAMS = 8
# Average token log-probability below this is flagged as low confidence.
LOW_CONFIDENCE_LOGPROB = -1.0

# Whisper often labels Hindi speech as Urdu and writes it in Arabic script.
LANGUAGE_REMAP = {"ur": "hi"}
WORD_RE = re.compile(r"[\w']+", re.UNICODE)

USABLE_STATUSES = ("ok",)


@dataclass
class TranscriptionConfig:
    model_name: str = DEFAULT_MODEL
    language: str | None = None          # None = auto-detect
    remap_urdu: bool = True              # transcribe detected Urdu as Hindi (Devanagari, not Arabic script)
    initial_prompt: str | None = None    # optional Whisper prompt; off by default
    vad_retry: bool = True               # retry near-empty reels without the voice filter
    audio_dir: Path = Path("discovery_debug/audio")
    keep_audio: bool = False
    cookies_browser: str | None = None   # e.g. "chrome", only if logged-out downloads fail
    cache_dir: Path | None = None        # reuse finished transcripts saved here (handy while testing)
    delay_s: float = DELAY_BETWEEN_DOWNLOADS_S


# ------------------------------------------------------------- text checks ---

def _words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def is_prompt_echo(segment_text: str, prompt: str) -> bool:
    """True when a segment mostly repeats the prompt (Whisper does this over music).

    Needs at least 3 words, at least 75% of which appear in the prompt, so a short
    real line that happens to share a word ("doston", "brand") is kept.
    """
    seg = _words(segment_text)
    if len(seg) < 3:
        return False
    vocab = set(_words(prompt))
    return sum(w in vocab for w in seg) / len(seg) >= 0.75


def has_repetition_loop(text: str) -> bool:
    """True when one 3-word phrase makes up a large share of the transcript."""
    w = _words(text)
    trigrams = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
    if len(trigrams) < REPEAT_MIN_TRIGRAMS:
        return False
    top = max(trigrams.count(t) for t in set(trigrams))
    return top / len(trigrams) >= REPEAT_SHARE


def _text(segs: list[dict]) -> str:
    return " ".join(s["text"] for s in segs if s["text"])


def _word_count(segs: list[dict]) -> int:
    return len(_text(segs).split())


def decide_status(word_count: int, wpm: float | None, notes: list[str]) -> str:
    """Status for a finished transcription, from its size, pace and notes."""
    if word_count < MIN_WORDS:
        return "empty"
    if "repetition_loop" in notes:
        return "music_or_lyrics"
    if "no_vad_pass" in notes and (wpm is None or wpm < SPEECH_MIN_WPM):
        return "music_or_lyrics"
    return "ok"


# ------------------------------------------------------------------ steps ---

def download_audio(url: str, shortcode: str, audio_dir: Path, cookies_browser: str | None = None):
    """Download one Reel's audio as mp3. Returns (path, info) with upload_date and duration."""
    import yt_dlp

    audio_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(audio_dir / f"{shortcode}.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
        "no_warnings": True,
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    path = audio_dir / f"{shortcode}.mp3"
    if not path.exists():
        raise FileNotFoundError(f"expected audio file not found: {path}")
    return path, {"upload_date": info.get("upload_date"), "duration": info.get("duration")}


@lru_cache(maxsize=2)
def load_model(model_name: str):
    """Load a Whisper model once per process (the first run downloads it)."""
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, compute_type="int8")


def _decode(model, path, language, prompt=None, vad=True, condition=True):
    """One full Whisper pass. Returns (segments, info, echo_removed)."""
    segments, info = model.transcribe(
        str(path),
        language=language,
        vad_filter=vad,
        initial_prompt=prompt,
        condition_on_previous_text=condition,
    )
    segs = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
         "avg_logprob": s.avg_logprob}
        for s in segments  # generator: transcription actually happens here
    ]
    echo_removed = False
    if prompt:
        kept = [s for s in segs if not is_prompt_echo(s["text"], prompt)]
        echo_removed = len(kept) < len(segs)
        segs = kept
    # drop back-to-back duplicate segments (a common Whisper loop)
    deduped = []
    for s in segs:
        if not deduped or s["text"] != deduped[-1]["text"]:
            deduped.append(s)
    return deduped, info, echo_removed


def transcribe_audio(model, path: Path, config: TranscriptionConfig) -> dict:
    """Run language detection, the main pass and (for near-empty reels) the no-VAD retry."""
    notes = []
    language = config.language
    detected_p = None
    remapped_from = None
    prompt = config.initial_prompt

    # 1. language (detection only: the segment generator is never consumed here)
    if language is None:
        _, info = model.transcribe(str(path), vad_filter=True)
        language, detected_p = info.language, info.language_probability
        if config.remap_urdu and language in LANGUAGE_REMAP:
            remapped_from, language = language, LANGUAGE_REMAP[language]
            notes.append("remapped_from_urdu")

    # 2. main pass (Hindi stays in whatever script Whisper writes, usually Devanagari)
    segs, info, echo = _decode(model, path, language, prompt=prompt, vad=True)
    if detected_p is None:
        detected_p = info.language_probability

    # 3. near-empty: the voice filter may have cut speech that sits under music
    if config.vad_retry and _word_count(segs) < MIN_WORDS:
        retry, _, retry_echo = _decode(model, path, language, prompt=prompt, vad=False)
        if _word_count(retry) > _word_count(segs):
            segs, echo = retry, retry_echo
            notes.append("no_vad_pass")

    if echo:
        notes.append("prompt_echo_removed")
    text = _text(segs)
    if has_repetition_loop(text):
        notes.append("repetition_loop")
    if segs and sum(s["avg_logprob"] for s in segs) / len(segs) < LOW_CONFIDENCE_LOGPROB:
        notes.append("low_confidence")

    return {
        "text": text,
        "segments": [{k: s[k] for k in ("start", "end", "text")} for s in segs],
        "language": language,
        "language_probability": round(detected_p, 3) if detected_p is not None else None,
        "duration": info.duration,
        "remapped_from": remapped_from,
        "notes": notes,
    }


# --------------------------------------------------------------- per reel ---

def _empty_fields() -> dict:
    return {
        "transcript": "",
        "transcript_status": None,
        "transcript_notes": [],
        "transcript_error": "",
        "language": None,
        "language_probability": None,
        "remapped_from": None,
        "duration_s": None,
        "word_count": 0,
        "words_per_minute": None,
        "segments": [],
    }


def _load_cached(cache_dir: Path | None, shortcode: str) -> dict | None:
    if not cache_dir:
        return None
    path = cache_dir / f"{shortcode}.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # only finished results are reused; failures are retried
    return saved if saved.get("transcript_status") in ("ok", "empty", "music_or_lyrics") else None


def _save_cached(cache_dir: Path | None, shortcode: str, fields: dict) -> None:
    if not cache_dir:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{shortcode}.json").write_text(json.dumps(fields, indent=2, ensure_ascii=False), encoding="utf-8")


def transcribe_reel(reel: dict, config: TranscriptionConfig) -> dict:
    """Download + transcribe one Reel. Returns only the transcript fields. Never raises."""
    code = reel["shortcode"]
    cached = _load_cached(config.cache_dir, code)
    if cached is not None:
        return cached

    out = _empty_fields()
    url = reel.get("url") or f"https://www.instagram.com/reel/{code}/"

    try:
        audio_path, info = download_audio(url, code, config.audio_dir, config.cookies_browser)
    except Exception as exc:
        out["transcript_status"] = "download_failed"
        out["transcript_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return out

    try:
        tr = transcribe_audio(load_model(config.model_name), audio_path, config)
    except Exception as exc:
        out["transcript_status"] = "transcribe_failed"
        out["transcript_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return out
    finally:
        if not config.keep_audio:
            audio_path.unlink(missing_ok=True)

    words = len(tr["text"].split())
    duration = tr["duration"] or info.get("duration")
    wpm = round(words / (duration / 60), 1) if duration else None
    out.update({
        "transcript": tr["text"],
        "transcript_status": decide_status(words, wpm, tr["notes"]),
        "transcript_notes": tr["notes"],
        "language": tr["language"],
        "language_probability": tr["language_probability"],
        "remapped_from": tr["remapped_from"],
        "duration_s": round(duration, 1) if duration else None,
        "word_count": words,
        "words_per_minute": wpm,
        "segments": tr["segments"],
    })
    _save_cached(config.cache_dir, code, out)
    return out


def transcribe_reels(reels: list[dict], config: TranscriptionConfig | None = None, log=print) -> list[dict]:
    """Transcribe a list of reels in order. Returns new reel dicts with the transcript fields added."""
    config = config or TranscriptionConfig()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install it with: brew install ffmpeg")

    results = []
    for i, reel in enumerate(reels, 1):
        cached = _load_cached(config.cache_dir, reel["shortcode"]) is not None
        fields = transcribe_reel(reel, config)
        results.append({**reel, **fields})
        if log:
            log(f"[{i}/{len(reels)}] {reel.get('account', '')} {reel['shortcode']}: "
                f"{fields['transcript_status']} ({fields['word_count']} words)"
                + (f" [{', '.join(fields['transcript_notes'])}]" if fields["transcript_notes"] else "")
                + (" (cached)" if cached else ""))
        if i < len(reels) and not cached:
            time.sleep(config.delay_s)
    return results


# --------------------------------------------------------------- manual run ---

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("usage: python -m insta_analyser.tools.transcription <shortcode or reel url> [...] [--model small]")
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else DEFAULT_MODEL
    reels = []
    for a in args:
        code = a.rstrip("/").split("/")[-1]
        reels.append({"shortcode": code, "url": f"https://www.instagram.com/reel/{code}/"})
    cfg = TranscriptionConfig(model_name=model, cache_dir=Path("discovery_debug/transcripts_tool"))
    for r in transcribe_reels(reels, cfg):
        print(f"\n{r['shortcode']}  {r['transcript_status']}  lang={r['language']}  wpm={r['words_per_minute']}")
        print("  " + (r["transcript"] or "(no text)"))
