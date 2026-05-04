from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from eval.runner import EvalSample


@dataclass
class MetricScores:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    avg_latency_ms: float
    sample_count: int

    def mean_score(self) -> float:
        return (
            self.faithfulness
            + self.answer_relevancy
            + self.context_precision
            + self.context_recall
        ) / 4


def compute_ragas(
    samples: list[EvalSample],
    llm_model: str = "gpt-4o-mini",
    openai_api_key: str = "",
) -> MetricScores:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    key: str | None = openai_api_key or None
    llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model, api_key=key))
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(openai_api_key=key)
    )

    ragas_samples: list[SingleTurnSample] = [
        SingleTurnSample(
            user_input=s.query,
            response=s.answer,
            retrieved_contexts=s.contexts,
            reference=s.ground_truth,
        )
        for s in samples
        if s.contexts
    ]

    if not ragas_samples:
        return MetricScores(
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_precision=0.0,
            context_recall=0.0,
            avg_latency_ms=_avg_latency(samples),
            sample_count=len(samples),
        )

    from ragas import MultiTurnSample

    dataset = EvaluationDataset(
        samples=cast("list[SingleTurnSample | MultiTurnSample]", ragas_samples)
    )
    raw_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    from ragas import EvaluationResult

    result: EvaluationResult = raw_result
    df = result.to_pandas()

    def _col(name: str) -> float:
        return float(df[name].mean()) if name in df.columns else 0.0

    return MetricScores(
        faithfulness=_col("faithfulness"),
        answer_relevancy=_col("answer_relevancy"),
        context_precision=_col("context_precision"),
        context_recall=_col("context_recall"),
        avg_latency_ms=_avg_latency(samples),
        sample_count=len(samples),
    )


def _avg_latency(samples: list[EvalSample]) -> float:
    if not samples:
        return 0.0
    return sum(s.latency_ms for s in samples) / len(samples)
