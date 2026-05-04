import structlog

from app.core.config import Settings

logger = structlog.get_logger(__name__)


class LLMClient:
    """Thin async wrapper around LiteLLM — keeps the rest of the codebase provider-agnostic."""

    def __init__(self, settings: Settings) -> None:
        self._model = settings.llm_model
        self._api_key = settings.openai_api_key

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        import litellm

        response = await litellm.acompletion(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self._api_key,
        )
        content: str = response.choices[0].message.content or ""
        logger.debug("llm complete", model=self._model, tokens=response.usage.total_tokens)
        return content
