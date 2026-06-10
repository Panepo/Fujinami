from __future__ import annotations

import os
import json
from pathlib import Path
from urllib import request

from document_loader import DocumentLoader


def test_ollama_vlm_smoke_for_image() -> None:
    """Verify index-time Ollama VLM can describe an image."""
    image_path = Path(os.environ.get("VLM_SMOKE_IMAGE", "data/test/IMG_8196.jpeg"))
    assert image_path.exists(), f"Smoke image not found: {image_path}"

    ollama_index_url = os.environ.get("OLLAMA_INDEX_URL", "").strip()
    vlm_model = os.environ.get("VLM_MODEL", "").strip()

    assert ollama_index_url, "OLLAMA_INDEX_URL is required for VLM smoke test"
    assert vlm_model, "VLM_MODEL is required for VLM smoke test"

    with request.urlopen(f"{ollama_index_url.rstrip('/')}/api/tags", timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    model_names = [m.get("name", "") for m in payload.get("models", []) if isinstance(m, dict)]
    assert any(name == vlm_model or name.startswith(f"{vlm_model}:") for name in model_names), (
        f"VLM model '{vlm_model}' not found on {ollama_index_url}. Available models: {model_names}"
    )

    loader = DocumentLoader(
        ollama_base_url=ollama_index_url,
        vlm_model=vlm_model,
        request_timeout=float(os.environ.get("VLM_TIMEOUT", "180")),
    )

    summary = loader._summarize_image(str(image_path)).strip()
    assert summary, (
        "VLM returned empty output. "
        f"OLLAMA_INDEX_URL={ollama_index_url}, VLM_MODEL={vlm_model}, IMAGE={image_path}"
    )
