"""LLM-based document relevance judgment with disk caching."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path("llm_cache")

INDIVIDUAL_PROMPT = """Review this document for responsiveness to the following request:

REQUEST: {query}

DOCUMENT:
{doc_text}

Respond with JSON only: {{"judgment": "RELEVANT" or "NOT RELEVANT", "confidence": 0.0-1.0}}"""

BATCH_PROMPT = """Review each document for responsiveness to the following request:

REQUEST: {query}

{documents}

Respond with JSON only — an array of objects, one per document, in the same order:
[{{"doc_id": "...", "judgment": "RELEVANT" or "NOT RELEVANT", "confidence": 0.0-1.0}}, ...]"""


@dataclass
class JudgmentResult:
    doc_id: str
    judgment: str
    confidence: float

    @property
    def is_relevant(self) -> bool:
        return self.judgment.upper() == "RELEVANT"


def _cache_key(model: str, doc_text: str, query: str, paradigm: str) -> str:
    content = f"{model}|{doc_text}|{query}|{paradigm}"
    return hashlib.sha256(content.encode()).hexdigest()[:24]


def _load_cache(cache_dir: Path, key: str) -> list[JudgmentResult] | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return [JudgmentResult(**r) for r in data]


def _save_cache(cache_dir: Path, key: str, results: list[JudgmentResult]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    data = [{"doc_id": r.doc_id, "judgment": r.judgment, "confidence": r.confidence} for r in results]
    with open(path, "w") as f:
        json.dump(data, f)


def _get_anthropic_client():
    import anthropic
    return anthropic.Anthropic()


def _get_openai_client():
    import openai
    return openai.OpenAI()


def _call_anthropic(prompt: str, model: str) -> str:
    client = _get_anthropic_client()
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_openai(prompt: str, model: str) -> str:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _call_together(prompt: str, model: str) -> str:
    client = _get_openai_client()
    client.base_url = "https://api.together.xyz/v1"
    client.api_key = os.environ.get("TOGETHER_API_KEY", "")
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


MODEL_BACKENDS = {
    "claude-sonnet-4": ("anthropic", "claude-sonnet-4-20250514"),
    "gpt-4o": ("openai", "gpt-4o"),
    "llama-3.1-70b": ("together", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
    "qwen-2.5-72b": ("together", "Qwen/Qwen2.5-72B-Instruct-Turbo"),
}


def _call_llm(prompt: str, model: str) -> str:
    if model not in MODEL_BACKENDS:
        raise ValueError(f"Unknown model: {model}. Supported: {list(MODEL_BACKENDS.keys())}")
    backend, model_id = MODEL_BACKENDS[model]
    if backend == "anthropic":
        return _call_anthropic(prompt, model_id)
    elif backend == "openai":
        return _call_openai(prompt, model_id)
    elif backend == "together":
        return _call_together(prompt, model_id)
    raise ValueError(f"Unknown backend: {backend}")


def _parse_json(text: str) -> dict | list:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def judge_individual(
    doc_id: str,
    doc_text: str,
    query: str,
    model: str,
    cache_dir: Path | None = None,
) -> JudgmentResult:
    cache_dir = cache_dir or CACHE_DIR
    key = _cache_key(model, doc_text, query, "individual")
    cached = _load_cache(cache_dir, key)
    if cached:
        result = cached[0]
        result.doc_id = doc_id
        return result

    prompt = INDIVIDUAL_PROMPT.format(query=query, doc_text=doc_text)
    response = _call_llm(prompt, model)
    parsed = _parse_json(response)
    result = JudgmentResult(
        doc_id=doc_id,
        judgment=parsed["judgment"],
        confidence=float(parsed["confidence"]),
    )
    _save_cache(cache_dir, key, [result])
    return result


def judge_batch(
    doc_ids: list[str],
    doc_texts: list[str],
    query: str,
    model: str,
    cache_dir: Path | None = None,
) -> list[JudgmentResult]:
    cache_dir = cache_dir or CACHE_DIR
    combined_text = "\n\n".join(
        f"--- DOCUMENT {doc_id} ---\n{text}" for doc_id, text in zip(doc_ids, doc_texts)
    )
    key = _cache_key(model, combined_text, query, "batch")
    cached = _load_cache(cache_dir, key)
    if cached:
        for i, r in enumerate(cached):
            r.doc_id = doc_ids[i]
        return cached

    prompt = BATCH_PROMPT.format(query=query, documents=combined_text)
    response = _call_llm(prompt, model)
    parsed = _parse_json(response)
    results = []
    for i, entry in enumerate(parsed):
        results.append(JudgmentResult(
            doc_id=doc_ids[i],
            judgment=entry["judgment"],
            confidence=float(entry["confidence"]),
        ))
    _save_cache(cache_dir, key, results)
    return results
