from collections.abc import AsyncGenerator
from typing import Any

import structlog

from app.query.llm import LLMClient
from app.retrieval.base import SearchResult

logger = structlog.get_logger(__name__)

# prompt went through a few iterations — this version keeps citations tight
# and stops the model from making stuff up
_SYSTEM = """\
You are a knowledgeable assistant. Answer the question using ONLY the provided context.
- Be concise and precise.
- Cite sources inline using bracket notation: [1], [2], etc., matching the numbered chunks below.
- If the context does not contain enough information, say exactly: \
"I don't have enough information in the provided documents to answer this question."
- Do not fabricate facts or draw on knowledge outside the context."""

_USER = """\
Context:
{context}

Question: {query}

Answer:"""

_MAX_TOKENS = 1024
_NO_INFO = (
    "I don't have enough information in the provided documents to answer this question."
)


def _build_context(results: list[SearchResult]) -> str:
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        pages = ", ".join(str(p) for p in r.page_numbers)
        header = f" — {r.section_header}" if r.section_header else ""
        lines.append(f"[{i}] {r.filename} (p. {pages}){header}\n{r.content}")
    return "\n\n".join(lines)


def _build_messages(query: str, results: list[SearchResult]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(context=_build_context(results), query=query)},
    ]


class AnswerSynthesizer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def synthesize(self, query: str, results: list[SearchResult]) -> str:
        if not results:
            return _NO_INFO
        answer = await self._llm.complete(
            messages=_build_messages(query, results),
            temperature=0.2,
            max_tokens=_MAX_TOKENS,
        )
        logger.info("answer synthesized", query_len=len(query), result_count=len(results))
        return answer.strip()

    async def stream(
        self, query: str, results: list[SearchResult]
    ) -> AsyncGenerator[str, None]:
        if not results:
            yield _NO_INFO
            return

        import litellm

        messages = _build_messages(query, results)
        response = await litellm.acompletion(
            model=self._llm._model,
            messages=messages,
            temperature=0.2,
            max_tokens=_MAX_TOKENS,
            api_key=self._llm._api_key,
            stream=True,
        )

        async for chunk in response:
            delta: Any = chunk.choices[0].delta
            token: str = getattr(delta, "content", None) or ""
            if token:
                yield token
