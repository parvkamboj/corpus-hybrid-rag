import structlog

from app.query.llm import LLMClient

logger = structlog.get_logger(__name__)

_SYSTEM = (
    "You are a conversation assistant. "
    "Given the conversation history and a follow-up message, rewrite the follow-up as a "
    "complete, self-contained question that makes sense without the conversation history. "
    "Preserve the user's intent exactly. "
    "Return only the rewritten question — no explanation, no preamble."
)

_USER = """\
Conversation history:
{history}

Follow-up: {query}

Standalone question:"""


class ConversationQueryRewriter:
    """Rewrites a follow-up question into a standalone query using conversation history."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def rewrite(self, query: str, history: list[dict[str, str]]) -> str:
        if not history:
            return query

        history_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-6:]
        )

        rewritten = await self._llm.complete(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER.format(history=history_text, query=query)},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        rewritten = rewritten.strip()
        if not rewritten:
            logger.warning("rewrite returned empty, using original query")
            return query
        logger.info("query rewritten", original=query[:80], rewritten=rewritten[:80])
        return rewritten
