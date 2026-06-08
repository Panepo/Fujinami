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
    rows_json = json.dumps(rows, ensure_ascii=True, indent=2)
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


async def _score(metric, sample) -> float:
    available = {
        "user_input": sample.user_input,
        "response": sample.response,
        "retrieved_contexts": sample.retrieved_contexts,
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
        row = {"user_input": sample.user_input, "reference": sample.reference}
        for j, metric in enumerate(metrics):
            result = flat[i * n + j]
            row[metric.name] = None if isinstance(result, BaseException) else result
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

    chat_model = os.environ["CHAT_MODEL"]
    embedding_model = os.environ["EMBEDDING_MODEL"]
    ollama_chat_url = os.environ["OLLAMA_CHAT_URL"]
    openai_client = AsyncOpenAI(
        base_url=_ollama_openai_base_url(ollama_chat_url),
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
    )

    llm = llm_factory(
        chat_model,
        provider="openai",
        client=openai_client,
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
    summary = {
        metric.name: mean(
            float(row[metric.name])
            for row in rows
            if isinstance(row[metric.name], int | float)
        )
        for metric in metrics
    }

    _emit_report(pytestconfig, rows, summary)
