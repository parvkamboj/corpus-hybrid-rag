import structlog

from app.query.llm import LLMClient

logger = structlog.get_logger(__name__)

# prompt tuned to avoid the model saying "The document would state..." type phrasing
_SYSTEM = (
    "You are a document retrieval assistant. "
    "Write a concise passage (2–4 sentences) that would directly answer the question below. "
    "Write as if you are the document — do not say 'The document states'. "
    "Use specific, concrete language."
)

_USER = "Question: {query}\n\nPassage:"


class HyDETransformer:
    """Generates a hypothetical passage and uses it as the retrieval query.

    HyDE (Hypothetical Document Embeddings) works by asking the LLM to hallucinate
    an answer, then using that as the dense query instead of the raw question.
    Embeds better because it's in the same style as the actual documents.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def transform(self, query: str) -> str:
        passage = await self._llm.complete(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER.format(query=query)},
            ],
            temperature=0.7,  # some variation is fine here, we want diverse hypotheticals
            max_tokens=200,
        )
        passage = passage.strip()
        if not passage:
            logger.warning("hyde returned empty passage, using original query")
            return query
        logger.info("hyde passage generated", original_len=len(query), passage_len=len(passage))
        return passage
