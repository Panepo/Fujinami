from __future__ import annotations

import asyncio
import inspect
import json
import os
from pathlib import Path
from statistics import mean

import pytest
from openai import AsyncOpenAI

ragas = pytest.importorskip("ragas")

from ragas import EvaluationDataset
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    Faithfulness,
)

from .ragas_helpers import collect_rag_samples, load_questions


def _emit_report(pytestconfig: pytest.Config, rows, summary) -> None:
    # Use pytest terminal reporter so output is visible even when capture is enabled.
    terminal_reporter = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    rows_json = json.dumps(public_rows, ensure_ascii=True, indent=2)
    summary_json = json.dumps(summary, ensure_ascii=True, indent=2)
    lines = [
        "",
        "RAGAS rows:",
        rows_json,
        "RAGAS summary:",
        summary_json,
    ]
    if terminal_reporter is not None:
        for line in lines:
            for subline in line.splitlines() or [""]:
                terminal_reporter.write_line(subline)
        return

    # Fallback for non-standard runners where terminalreporter is unavailable.
    for line in lines:
        print(line)


def _resolve_collection_name(pytestconfig: pytest.Config) -> str:
    collection_name = str(pytestconfig.getoption("collection")).strip()
    return collection_name or "S510AD"


def _ollama_openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


_MAX_CONTEXT_CHARS = 1_200   # per retrieved chunk
_MAX_CONTEXTS = 3            # max chunks passed to a metric
_MAX_RESPONSE_CHARS = 1_200  # cap answer length sent to metrics


async def _score(metric, sample) -> float:
    contexts = sample.retrieved_contexts or []
    truncated_contexts = [c[:_MAX_CONTEXT_CHARS] for c in contexts[:_MAX_CONTEXTS]]
    available = {
        "user_input": sample.user_input,
        "response": (sample.response or "")[:_MAX_RESPONSE_CHARS],
        "retrieved_contexts": truncated_contexts,
        "reference": sample.reference,
    }
    inputs = {
        k: v for k, v in available.items()
        if k in inspect.signature(metric.ascore).parameters
    }
    result = await metric.ascore(**inputs)
    return result.value


async def _evaluate_collection_metrics(samples, metrics) -> list[dict[str, object]]:
    tasks = [
        _score(metric, sample)
        for sample in samples
        for metric in metrics
    ]
    flat = await asyncio.gather(*tasks, return_exceptions=True)

    rows: list[dict[str, object]] = []
    n = len(metrics)
    for i, sample in enumerate(samples):
        row: dict[str, object] = {"user_input": sample.user_input, "reference": sample.reference}
        for j, metric in enumerate(metrics):
            result = flat[i * n + j]
            if isinstance(result, BaseException):
                row[metric.name] = None
                row[f"_{metric.name}_error"] = repr(result)
            else:
                row[metric.name] = result
        rows.append(row)
    return rows


def test_ragas_evaluation(pytestconfig: pytest.Config) -> None:
    collection_name = _resolve_collection_name(pytestconfig)
    questions_dir = Path(__file__).parent / "questions"
    questions = load_questions(collection_name, questions_dir)

    assert questions, (
        f"No question files found for collection '{collection_name}' "
        f"under {questions_dir}."
    )

    samples = asyncio.run(collect_rag_samples(questions, collection_name))
    dataset = EvaluationDataset(samples=samples)

    ragas_model = os.environ["RAGAS_MODEL"]
    embedding_model = os.environ["EMBEDDING_MODEL"]
    ollama_chat_url = os.environ["OLLAMA_CHAT_URL"]
    openai_client = AsyncOpenAI(
        base_url=_ollama_openai_base_url(ollama_chat_url),
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
    )

    llm = llm_factory(
        ragas_model,
        provider="openai",
        client=openai_client,
        max_tokens=262144,
    )
    embeddings = embedding_factory(
        "openai",
        model=embedding_model,
        client=openai_client,
    )

    metrics = [
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        Faithfulness(llm=llm),
        ContextPrecisionWithoutReference(llm=llm),
    ]

    rows = asyncio.run(_evaluate_collection_metrics(dataset.samples, metrics))

    # Warn about any scoring failures so they are visible in output.
    for row in rows:
        for metric in metrics:
            val = row.get(f"_{metric.name}_error")
            if val is not None:
                import warnings
                warnings.warn(
                    f"Metric '{metric.name}' failed for '{row['user_input'][:60]}': {val}",
                    stacklevel=2,
                )

    summary = {}
    for metric in metrics:
        valid = [
            float(row[metric.name])
            for row in rows
            if isinstance(row[metric.name], (int, float))
        ]
        if not valid:
            summary[metric.name] = None
        else:
            summary[metric.name] = mean(valid)

    _emit_report(pytestconfig, rows, summary)
