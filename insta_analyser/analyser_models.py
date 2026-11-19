"""
Model configuration for the Instagram Reel Analyser. TESTING PHASE: everything local.

Three models, each set from environment variables (or a .env file) so they can be
swapped without touching the graph:

  Reports LLM   ANALYSER_LLM_PROVIDER  ollama (default) | groq
                ANALYSER_LLM_MODEL     default llama3.2:latest on Ollama, qwen/qwen3.8-27b on Groq
                GROQ_API_KEY           needed only for groq
  Classifier    ANALYSER_CLASSIFIER    default tev1:0.8b  (System One decision model in Ollama)
  Whisper       ANALYSER_WHISPER       default small
                ANALYSER_TRANSCRIPT_CACHE  optional folder that reuses finished transcripts
  Ollama        OLLAMA_HOST            default http://localhost:11434
                                       (from Docker later: http://host.docker.internal:11434)

llama3.2 is for testing only; the final report model is still to be chosen.

IMPORTANT: before the first run
    ollama pull llama3.2
    ollama pull tev1:0.8b
or the first request hangs waiting on a pull. check_models() reports what is missing.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from insta_analyser.tools.classification import ClassificationConfig
from insta_analyser.tools.transcription import TranscriptionConfig

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_PROVIDER = os.getenv("ANALYSER_LLM_PROVIDER", "ollama").lower()
LLM_MODEL = os.getenv(
    "ANALYSER_LLM_MODEL",
    "qwen/qwen3.8-27b" if LLM_PROVIDER == "groq" else "llama3.2:latest",
)
CLASSIFIER_MODEL = os.getenv("ANALYSER_CLASSIFIER", "tev1:0.8b")
WHISPER_MODEL = os.getenv("ANALYSER_WHISPER", "small")
TRANSCRIPT_CACHE = os.getenv("ANALYSER_TRANSCRIPT_CACHE")


# ============================================================
# REPORTS LLM
# ============================================================

def get_model(temperature: float = 0.3, json_mode: bool = True):
    """Chat model for the report nodes. json_mode forces a JSON object reply."""
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq  # only needed when switching to Groq

        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        return ChatGroq(
            model=LLM_MODEL,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
            model_kwargs=kwargs,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=LLM_MODEL,
        temperature=temperature,
        base_url=OLLAMA_HOST,
        format="json" if json_mode else None,
        num_ctx=8192,  # room for one category's reel notes; Ollama's default is smaller
    )


# ============================================================
# CLASSIFIER + WHISPER
# ============================================================

def get_classifier_config(stability_runs: int = 0) -> ClassificationConfig:
    return ClassificationConfig(model=CLASSIFIER_MODEL, host=OLLAMA_HOST, stability_runs=stability_runs)


def get_transcription_config() -> TranscriptionConfig:
    return TranscriptionConfig(
        model_name=WHISPER_MODEL,
        cache_dir=Path(TRANSCRIPT_CACHE) if TRANSCRIPT_CACHE else None,
    )


# ============================================================
# HEALTH CHECK
# ============================================================

def check_models() -> dict:
    """Which configured Ollama models are pulled. Used before a run and by a /health endpoint later."""
    import ollama

    needed = [CLASSIFIER_MODEL] + ([LLM_MODEL] if LLM_PROVIDER == "ollama" else [])
    try:
        pulled = {m.model for m in ollama.Client(host=OLLAMA_HOST).list().models}
    except Exception as exc:
        return {"ollama_reachable": False, "error": f"{type(exc).__name__}: {exc}", "missing": needed}

    def has(name):
        return name in pulled or (":" not in name and f"{name}:latest" in pulled)

    return {
        "ollama_reachable": True,
        "llm": f"{LLM_PROVIDER}:{LLM_MODEL}",
        "classifier": CLASSIFIER_MODEL,
        "whisper": WHISPER_MODEL,
        "missing": [m for m in needed if not has(m)],
    }
