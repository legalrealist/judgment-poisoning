"""LLM-based document relevance judgment with disk caching."""

import hashlib
import json
import os
import time
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
    import os
    return anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        auth_token="unused",
    )


def _get_openai_client():
    import openai
    return openai.OpenAI()


def _call_with_retry(fn, max_retries=5):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Rate limited after {max_retries} retries")


def _call_anthropic(prompt: str, model: str) -> str:
    client = _get_anthropic_client()
    def _do():
        message = client.messages.create(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    return _call_with_retry(_do)


def _call_openai(prompt: str, model: str) -> str:
    client = _get_openai_client()
    def _do():
        response = client.chat.completions.create(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    return _call_with_retry(_do)


def _call_together(prompt: str, model: str) -> str:
    client = _get_openai_client()
    client.base_url = "https://api.together.xyz/v1"
    client.api_key = os.environ.get("TOGETHER_API_KEY", "")
    def _do():
        response = client.chat.completions.create(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    return _call_with_retry(_do)


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


BATCH_CHUNK_SIZE = 15


def _judge_batch_chunk(
    doc_ids: list[str],
    doc_texts: list[str],
    query: str,
    model: str,
) -> list[JudgmentResult]:
    combined_text = "\n\n".join(
        f"--- DOCUMENT {doc_id} ---\n{text}" for doc_id, text in zip(doc_ids, doc_texts)
    )
    prompt = BATCH_PROMPT.format(query=query, documents=combined_text)
    response = _call_llm(prompt, model)
    parsed = _parse_json(response)
    response_map = {}
    for entry in parsed:
        did = entry.get("doc_id", "")
        response_map[did] = entry
    results = []
    for doc_id in doc_ids:
        if doc_id in response_map:
            entry = response_map[doc_id]
            results.append(JudgmentResult(
                doc_id=doc_id,
                judgment=entry["judgment"],
                confidence=float(entry["confidence"]),
            ))
        else:
            results.append(JudgmentResult(
                doc_id=doc_id, judgment="NOT RELEVANT", confidence=0.0,
            ))
    return results


def judge_batch(
    doc_ids: list[str],
    doc_texts: list[str],
    query: str,
    model: str,
    cache_dir: Path | None = None,
    no_chunk: bool = False,
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

    if no_chunk:
        all_results = _judge_batch_chunk(doc_ids, doc_texts, query, model)
    else:
        all_results = []
        for start in range(0, len(doc_ids), BATCH_CHUNK_SIZE):
            end = min(start + BATCH_CHUNK_SIZE, len(doc_ids))
            chunk_results = _judge_batch_chunk(
                doc_ids[start:end], doc_texts[start:end], query, model,
            )
            all_results.extend(chunk_results)

    _save_cache(cache_dir, key, all_results)
    return all_results
