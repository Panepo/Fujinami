"""
DocumentLoader — converts PDF/Word documents to plain text with inline
VLM-based image descriptions. Used by RagService before GraphRAG indexing.
"""
from __future__ import annotations

import base64
import io
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diagram-type prompt templates
# ---------------------------------------------------------------------------

_CLASSIFIER_PROMPT = (
    "Is this image a photo, chart, flowchart, UML diagram, table, or other diagram? "
    "Reply with exactly one word from: photo, chart, flowchart, uml, table, diagram."
)

_PROMPT_TEMPLATES: dict[str, str] = {
    "flowchart": (
        "You are analysing a flowchart. The surrounding document text is:\n\n{surrounding_text}\n\n"
        "List every node and every labeled edge in this flowchart as:\n"
        "  Node A → [edge label] → Node B\n"
        "One relationship per line. Be exhaustive."
    ),
    "uml": (
        "You are analysing a UML diagram. The surrounding document text is:\n\n{surrounding_text}\n\n"
        "List every class/actor/component and every labeled relationship as:\n"
        "  Entity A → [relationship label] → Entity B\n"
        "One relationship per line. Be exhaustive."
    ),
    "chart": (
        "You are analysing a chart. The surrounding document text is:\n\n{surrounding_text}\n\n"
        "Describe the chart type, axes labels, data series names, and the key values or trends shown."
    ),
    "table": (
        "You are analysing a table image. The surrounding document text is:\n\n{surrounding_text}\n\n"
        "Extract the full table as pipe-delimited rows (header row first). "
        "Example:\n  Column A | Column B | Column C\n  value1 | value2 | value3"
    ),
    "photo": (
        "You are analysing a photo. The surrounding document text is:\n\n{surrounding_text}\n\n"
        "Provide a concise factual description of what the photo shows, "
        "focusing on elements relevant to the document topic."
    ),
    "diagram": (
        "You are analysing a diagram. The surrounding document text is:\n\n{surrounding_text}\n\n"
        "Describe every component and relationship shown in the diagram. "
        "List relationships as: Component A → [relationship] → Component B where applicable."
    ),
}

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".doc"}


class DocumentLoader:
    """
    Loads text and image content from local documents (PDF, DOCX, DOC, TXT, MD).

    Images embedded in documents are described by an Ollama VLM, with
    descriptions spliced inline at the image's original position so that
    GraphRAG can extract relationships from diagram content.

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, file_path: str | Path) -> str:
        """Dispatch to the correct parser by extension and return extracted text."""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="replace")
        elif ext == ".pdf":
            return self._load_pdf(path)
        elif ext == ".docx":
            return self._load_docx(path)
        elif ext == ".doc":
            return self._load_doc(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

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
    # Format-specific loaders
    # ------------------------------------------------------------------

    def _load_pdf(self, file_path: Path) -> str:
        """Extract text + inline image descriptions from a PDF."""
        import pypdf  # noqa: PLC0415
        import fitz   # pymupdf  # noqa: PLC0415

        text_parts: list[str] = []
        reader = pypdf.PdfReader(str(file_path))
        fitz_doc = fitz.open(str(file_path))

        try:
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text() or ""
                fitz_page = fitz_doc[page_num]

                # Check if this is a scanned (image-only) page
                if not page_text.strip():
                    # Render full page as image for VLM-based OCR
                    pix = fitz_page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    desc = self._describe_image(img_bytes, "")
                    if desc:
                        text_parts.append(f"[PAGE {page_num + 1} SCAN]\n{desc}")
                    continue

                # Extract embedded images and describe each one
                image_descs: list[str] = []
                for img_ref in fitz_page.get_images(full=True):
                    xref = img_ref[0]
                    try:
                        base_image = fitz_doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        surrounding = page_text[:500]
                        desc = self._describe_image(img_bytes, surrounding)
                        if desc:
                            image_descs.append(desc)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Could not extract image on page %d: %s", page_num + 1, exc)

                combined = page_text
                if image_descs:
                    combined += "\n" + "\n".join(image_descs)
                text_parts.append(combined)
        finally:
            fitz_doc.close()

        return "\n\n".join(text_parts)

    def _load_docx(self, file_path: Path) -> str:
        """Extract text + inline image descriptions from a .docx file."""
        from docx import Document  # noqa: PLC0415
        from docx.oxml.ns import qn  # noqa: PLC0415

        doc = Document(str(file_path))
        result_parts: list[str] = []

        for para in doc.paragraphs:
            para_text = para.text
            inline_descs: list[str] = []

            for run in para.runs:
                # Search for <w:drawing> elements inside the run XML
                for drawing in run._element.iter(qn("w:drawing")):
                    for blip in drawing.iter(qn("a:blip")):
                        r_embed = blip.get(qn("r:embed"))
                        if r_embed:
                            rel = doc.part.rels.get(r_embed)
                            if rel is not None and hasattr(rel.target_part, "blob"):
                                try:
                                    img_bytes: bytes = rel.target_part.blob
                                    desc = self._describe_image(img_bytes, para_text[:500])
                                    if desc:
                                        inline_descs.append(desc)
                                except Exception as exc:  # noqa: BLE001
                                    logger.warning("Could not describe image in %s: %s", file_path.name, exc)

            combined = para_text
            if inline_descs:
                combined += "\n" + "\n".join(inline_descs)
            result_parts.append(combined)

        return "\n".join(result_parts)

    def _load_doc(self, file_path: Path) -> str:
        """Convert legacy .doc to .docx via COM automation, then delegate."""
        if sys.platform != "win32":
            raise NotImplementedError(
                "Legacy .doc loading requires Windows with Microsoft Word installed. "
                f"Skipping {file_path.name}."
            )

        import win32com.client  # noqa: PLC0415

        abs_path = str(file_path.resolve())
        docx_path = str(file_path.with_suffix("_converted.docx").resolve())

        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        try:
            doc = word.Documents.Open(abs_path)
            doc.SaveAs2(docx_path, FileFormat=16)  # 16 = wdFormatXMLDocument (.docx)
            doc.Close(False)
        finally:
            word.Quit()

        return self._load_docx(Path(docx_path))

    # ------------------------------------------------------------------
    # VLM image description helpers
    # ------------------------------------------------------------------

    def _detect_diagram_type(self, image_bytes: bytes) -> str:
        """Ask the VLM to classify the image type with a single-word answer."""
        try:
            b64 = base64.b64encode(image_bytes).decode("ascii")
            payload = {
                "model": self._vlm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": _CLASSIFIER_PROMPT,
                        "images": [b64],
                    }
                ],
                "stream": False,
            }
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip().lower()
            # Normalise to known tags
            for tag in ("flowchart", "uml", "chart", "table", "photo", "diagram"):
                if tag in raw:
                    return tag
            return "diagram"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Diagram type detection failed: %s", exc)
            return "diagram"

    def _build_vlm_prompt(self, diagram_type: str, surrounding_text: str) -> str:
        """Return a tailored VLM prompt based on the detected diagram type."""
        template = _PROMPT_TEMPLATES.get(diagram_type, _PROMPT_TEMPLATES["diagram"])
        return template.format(surrounding_text=surrounding_text or "(no surrounding text)")

    def _describe_image(self, image_bytes: bytes, surrounding_text: str) -> str:
        """
        Describe *image_bytes* using the configured Ollama VLM.

        Returns a tagged string such as::

            [DIAGRAM: Node A → depends on → Node B ...]
            [IMAGE DESCRIPTION: bar chart showing ...]

        On any failure logs a warning and returns an empty string so that
        document indexing can continue.
        """
        try:
            diagram_type = self._detect_diagram_type(image_bytes)
            prompt = self._build_vlm_prompt(diagram_type, surrounding_text)
            b64 = base64.b64encode(image_bytes).decode("ascii")

            payload = {
                "model": self._vlm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [b64],
                    }
                ],
                "stream": False,
            }
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"].strip()

            if diagram_type in ("flowchart", "uml", "diagram"):
                return f"[DIAGRAM: {content}]"
            return f"[IMAGE DESCRIPTION: {content}]"

        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM image description failed (best-effort): %s", exc)
            return ""
