import re
from dataclasses import dataclass
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)

Intent = Literal["lookup", "analysis", "summarization"]

# NOTE: this is purely keyword-based for now. a small classifier model would be
# better long-term but overkill for a POC. revisit if routing accuracy is poor.

_SUMMARIZATION_TOKENS = {
    "summarize", "summary", "summarise", "summarisation", "overview",
    "outline", "recap", "brief", "digest", "abstract",
}

_ANALYSIS_TOKENS = {
    "compare", "contrast", "analyze", "analyse", "analysis",
    "explain", "why", "how does", "how do", "difference", "differences",
    "relationship", "impact", "effect", "effects", "evaluate", "assess",
    "distinguish", "pros", "cons", "advantages", "disadvantages",
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def classify(query: str) -> Intent:
    lower = query.lower()
    words = _tokens(lower)

    if words & _SUMMARIZATION_TOKENS:
        return "summarization"

    if words & _ANALYSIS_TOKENS:
        return "analysis"

    # catch multi-word phrases that tokenization misses
    for phrase in ("how does", "how do", "what is the difference", "what are the differences"):
        if phrase in lower:
            return "analysis"

    return "lookup"


@dataclass(frozen=True)
class RouteConfig:
    strategy: Literal["dense", "sparse", "hybrid", "hybrid_rerank"]
    top_k: int


# top_k values chosen based on rough testing — summarization needs more context
# TODO: make these configurable via settings
_ROUTES: dict[Intent, RouteConfig] = {
    "lookup": RouteConfig(strategy="hybrid_rerank", top_k=5),
    "analysis": RouteConfig(strategy="hybrid_rerank", top_k=15),
    "summarization": RouteConfig(strategy="dense", top_k=20),
}


class QueryRouter:
    def route(self, query: str) -> tuple[Intent, RouteConfig]:
        intent = classify(query)
        config = _ROUTES[intent]
        logger.info("query classified", intent=intent, strategy=config.strategy, top_k=config.top_k)
        return intent, config
