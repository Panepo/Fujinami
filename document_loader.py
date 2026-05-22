"""
DocumentLoader — converts documents to markdown using Docling's DocumentConverter.
Used by RagService before GraphRAG indexing.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".md", ".markdown", ".adoc", ".asciidoc",
    ".tex", ".html", ".htm", ".xhtml",
    ".csv", ".txt", ".text", ".vtt",
    # Images
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
    # Audio (docling[asr] extra)
    ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac",
    # Video (docling[asr] extra + ffmpeg)
    ".mp4", ".avi", ".mov",
}


class DocumentLoader:
    """
    Loads text and image content from local documents using Docling's DocumentConverter.

    Docling handles format detection, OCR, table extraction, and VLM-based
    picture description. The output is exported as Markdown to preserve
    structure (headings, tables) for richer GraphRAG extraction.

    Parameters
    ----------
    ollama_base_url:
        Base URL of the Ollama server (e.g. ``"http://10.168.3.58:8088"``).
    vlm_model:
        Ollama VLM model name to use for image description.
    request_timeout:
        HTTP timeout in seconds for VLM calls.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://10.168.3.58:8088",
        vlm_model: str = "llava:7b",
        request_timeout: float = 60.0,
    ) -> None:
        self._base_url = ollama_base_url.rstrip("/")
        self._vlm_model = vlm_model
        self._timeout = request_timeout
        self._converter = self._build_converter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, file_path: str | Path) -> str:
        """Convert *file_path* using Docling and return extracted markdown text."""
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")
        result = self._converter.convert(str(path))
        return result.document.export_to_markdown()

    def load_directory(self, directory: str | Path, files_filter: set[str] | None = None) -> dict[str, str]:
        """Walk *directory*, call ``load()`` on each supported file.

        Parameters
        ----------
        directory:
            Root directory to walk recursively.
        files_filter:
            Optional set of basenames (e.g. ``{"guide.pdf", "notes.md"}``).
            When provided, only files whose ``name`` is in this set are loaded.
            Pass ``None`` (default) to load all supported files.

        Returns
        -------
        dict[str, str]
            Mapping of ``filename → extracted text``.
        """
        directory = Path(directory)
        results: dict[str, str] = {}

        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                if files_filter is not None and file_path.name not in files_filter:
                    continue
                try:
                    results[file_path.name] = self.load(file_path)
                    logger.info("Loaded %s", file_path.name)
                except Exception as exc:
                    logger.warning("Failed to load %s: %s", file_path.name, exc)

        return results

    # ------------------------------------------------------------------
    # Converter builder
    # ------------------------------------------------------------------

    def _build_converter(self):
        """Build a Docling DocumentConverter with VLM-based picture description."""
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            PictureDescriptionApiOptions,
        )
        from docling.document_converter import (
            DocumentConverter,
            ImageFormatOption,
            PdfFormatOption,
        )

        pic_desc_opts = PictureDescriptionApiOptions(
            url=f"{self._base_url}/v1/chat/completions",
            timeout=self._timeout,
            params={"model": self._vlm_model},
            prompt=(
                "Describe this image in detail. "
                "If it is a diagram, flowchart, UML diagram, or chart, "
                "list all components and relationships exhaustively. "
                "If it contains text, extract it fully."
            ),
        )

        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_picture_description = True
        pdf_opts.picture_description_options = pic_desc_opts
        pdf_opts.enable_remote_services = True

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pdf_opts),
            }
        )
