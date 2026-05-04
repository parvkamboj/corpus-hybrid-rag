import asyncio
import json
from typing import Any

import structlog

from app.query.llm import LLMClient
from app.retrieval.base import MetadataFilter, SearchOutput, SearchResult

logger = structlog.get_logger(__name__)

_SYSTEM = "You are a query decomposition assistant. Respond only with valid JSON arrays of strings."

_USER = """\
Break the following complex question into 2–4 simpler, self-contained sub-questions.
Each sub-question must be understandable on its own — no pronouns like "it" or "this" \
without a referent.
Return a JSON array of strings and nothing else.

Question: {query}"""


class QueryDecomposer:
    """Splits a complex query into sub-queries, retrieves for each, then deduplicates."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def decompose(self, query: str) -> list[str]:
        raw = await self._llm.complete(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER.format(query=query)},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        try:
            parsed = json.loads(raw.strip())
            if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                # cap at 4 sub-queries — more than that and latency gets bad
                sub_queries = [q.strip() for q in parsed if q.strip()][:4]
                logger.info("query decomposed", count=len(sub_queries))
                return sub_queries or [query]
        except (json.JSONDecodeError, ValueError):
            logger.warning("decomposition parse failed, using original query", raw=raw[:200])
        return [query]

    async def retrieve_and_merge(
        self,
        query: str,
        retriever: Any,
        top_k: int,
        filters: MetadataFilter | None,
        debug: bool,
    ) -> SearchOutput:
        sub_queries = await self.decompose(query)

        outputs: list[SearchOutput] = await asyncio.gather(
            *[retriever.search(q, top_k, filters, debug) for q in sub_queries]
        )

        # deduplicate by chunk_id, keep highest score
        best: dict[str, SearchResult] = {}
        for out in outputs:
            for result in out.results:
                if result.chunk_id not in best or result.score > best[result.chunk_id].score:
                    best[result.chunk_id] = result

        merged = sorted(best.values(), key=lambda r: r.score, reverse=True)[:top_k]
        for i, r in enumerate(merged):
            r.rank = i

        dbg: dict[str, Any] = {}
        if debug:
            dbg["sub_queries"] = sub_queries
            dbg["per_sub_query"] = [
                {"sub_query": sub_queries[i], "result_count": len(out.results)}
                for i, out in enumerate(outputs)
            ]

        return SearchOutput(results=merged, debug=dbg)
