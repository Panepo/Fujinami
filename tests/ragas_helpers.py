from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag_service import RagService

if TYPE_CHECKING:
    from ragas import SingleTurnSample


def load_questions(collection: str, questions_dir: Path) -> list[dict[str, str]]:
    """Load and flatten question files for one collection."""
    import json

    records: list[dict[str, str]] = []
    pattern = f"{collection}_*.json"
    for path in sorted(questions_dir.glob(pattern)):
        with path.open("r", encoding="utf-8") as file_obj:
            items = json.load(file_obj)

        for item in items:
            question = str(item.get("user_input", "")).strip()
            ground_truth = str(item.get("reference", "")).strip()
            category = str(item.get("\u5b50\u5206\u985e", "")).strip()
            if not question:
                continue
            records.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "category": category,
                }
            )
    return records


async def collect_rag_samples(
    questions: list[dict[str, str]],
    collection: str,
) -> list[SingleTurnSample]:
    """Run local RAG calls and convert results to RAGAS samples (parallel)."""
    import asyncio
    from ragas import SingleTurnSample

    svc = RagService(collection_name=collection)

    async def _fetch(row: dict) -> SingleTurnSample:
        user_input = row["question"]
        reference = row.get("ground_truth", "")
        response, raw_results = await asyncio.gather(
            svc.hybrid_search(user_input),
            svc._raw_vector_results(user_input),
        )
        contexts = [str(r.get("text", "")) for r in raw_results if r.get("text")]
        return SingleTurnSample(
            user_input=user_input,
            response=response,
            retrieved_contexts=contexts,
            reference=reference,
        )

    return list(await asyncio.gather(*[_fetch(row) for row in questions]))
