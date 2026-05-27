"""Delta detection and manifest helpers for incremental indexing."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from document_loader import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> dict[str, str]:
    """Load ``file_manifest.json`` → ``{filename: sha256_hex}``."""
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read manifest, treating as empty: %s", exc)
        return {}


def save_manifest(
    documents_dir: Path,
    manifest_path: Path,
    exclude: set[str] | None = None,
) -> None:
    """Write fresh ``file_manifest.json`` with SHA-256 hashes for all on-disk files.

    Parameters
    ----------
    documents_dir:
        Directory containing source documents.
    manifest_path:
        Destination path for the manifest JSON file.
    exclude:
        Filenames to omit (e.g. files that failed to load). Omitted files
        appear as *new* on the next run and will be retried.
    """
    exclude = exclude or set()
    manifest: dict[str, str] = {}
    for file_path in documents_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            if file_path.name not in exclude:
                manifest[file_path.name] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Manifest saved with %d entries", len(manifest))


def compute_delta(
    documents_dir: Path,
    stored_manifest: dict[str, str],
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Compare on-disk files against *stored_manifest* using SHA-256 content hashes.

    Returns
    -------
    (new_files, modified_files, deleted_files, unchanged_files)
    """
    on_disk: dict[str, str] = {}
    for file_path in documents_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            on_disk[file_path.name] = file_hash

    on_disk_names = set(on_disk)
    stored_names = set(stored_manifest)

    new_files = on_disk_names - stored_names
    deleted_files = stored_names - on_disk_names
    modified_files: set[str] = set()
    unchanged_files: set[str] = set()

    for name in on_disk_names & stored_names:
        if on_disk[name] != stored_manifest[name]:
            modified_files.add(name)
        else:
            unchanged_files.add(name)

    return new_files, modified_files, deleted_files, unchanged_files


def load_index_flags(ragdata_dir: Path) -> dict:
    """Return ``{vector_indexed, graph_indexed}`` for a collection."""
    flags_path = ragdata_dir / "index_flags.json"
    if not flags_path.exists():
        return {"vector_indexed": False, "graph_indexed": False}
    try:
        return json.loads(flags_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"vector_indexed": False, "graph_indexed": False}


def save_index_flags(
    ragdata_dir: Path,
    *,
    vector_indexed: Optional[bool] = None,
    graph_indexed: Optional[bool] = None,
) -> None:
    """Update and persist ``index_flags.json``."""
    flags = load_index_flags(ragdata_dir)
    if vector_indexed is not None:
        flags["vector_indexed"] = vector_indexed
    if graph_indexed is not None:
        flags["graph_indexed"] = graph_indexed
    ragdata_dir.mkdir(parents=True, exist_ok=True)
    (ragdata_dir / "index_flags.json").write_text(
        json.dumps(flags, indent=2), encoding="utf-8"
    )
    logger.info("Index flags saved: %s", flags)
