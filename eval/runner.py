from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from eval.dataset import GoldenDataset, GoldenSample


@dataclass
class EvalSample:
    query: str
    ground_truth: str
    answer: str
    contexts: list[str]
    strategy: str
    latency_ms: float


async def _query_one(
    client: httpx.AsyncClient,
    sample: GoldenSample,
    strategy: str,
    top_k: int,
    base_url: str,
    api_key: str,
) -> EvalSample:
    payload: dict[str, object] = {
        "query": sample.query,
        "strategy": strategy,
        "top_k": top_k,
        "debug": False,
    }
    if sample.doc_ids:
        payload["filters"] = {"doc_id": sample.doc_ids[0]}

    response = await client.post(
        f"{base_url.rstrip('/')}/query/",
        json=payload,
        headers={"X-API-Key": api_key},
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()

    contexts = [src.get("chunk_id", "") for src in data.get("sources", [])]

    return EvalSample(
        query=sample.query,
        ground_truth=sample.ground_truth,
        answer=data.get("answer", ""),
        contexts=contexts,
        strategy=strategy,
        latency_ms=data.get("latency_ms", 0.0),
    )


async def _enrich_contexts(
    client: httpx.AsyncClient,
    sample: GoldenSample,
    strategy: str,
    top_k: int,
    base_url: str,
    api_key: str,
) -> list[str]:
    # /query doesn't return chunk content, so we fetch it separately via /search
    payload: dict[str, object] = {
        "query": sample.query,
        "strategy": strategy,
        "top_k": top_k,
        "debug": False,
    }
    if sample.doc_ids:
        payload["filters"] = {"doc_id": sample.doc_ids[0]}

    resp = await client.post(
        f"{base_url.rstrip('/')}/search/",
        json=payload,
        headers={"X-API-Key": api_key},
        timeout=60.0,
    )
    resp.raise_for_status()
    return [r["content"] for r in resp.json().get("results", [])]


async def run_eval(
    dataset: GoldenDataset,
    strategy: str,
    base_url: str,
    api_key: str,
    top_k: int = 5,
    concurrency: int = 3,
) -> list[EvalSample]:
    sem = asyncio.Semaphore(concurrency)
    results: list[EvalSample] = []

    async with httpx.AsyncClient() as client:

        async def _run_one(sample: GoldenSample) -> EvalSample:
            async with sem:
                eval_sample = await _query_one(client, sample, strategy, top_k, base_url, api_key)
                contexts = await _enrich_contexts(
                    client, sample, strategy, top_k, base_url, api_key
                )
                eval_sample.contexts = contexts
                return eval_sample

        gathered = await asyncio.gather(*[_run_one(s) for s in dataset.samples])
        results.extend(gathered)

    return results
