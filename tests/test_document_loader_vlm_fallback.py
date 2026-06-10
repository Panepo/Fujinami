from __future__ import annotations

import base64
import io
from pathlib import Path
import sys

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from document_loader import DocumentLoader


def _png_data_uri(width: int = 10, height: int = 10) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 255, 255)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def test_save_picture_from_pages_fallback_crops_bbox(tmp_path: Path) -> None:
    loader = DocumentLoader()

    pic_item = {
        "prov": [
            {
                "page_no": 1,
                "bbox": {
                    "l": 2,
                    "t": 8,
                    "r": 8,
                    "b": 2,
                    "coord_origin": "BOTTOMLEFT",
                },
            }
        ],
        "image": None,
    }
    pages = {
        "1": {
            "image": {
                "uri": _png_data_uri(10, 10),
            }
        }
    }

    out = loader._save_picture_from_json(pic_item, tmp_path, "e0", pages=pages)

    assert out is not None
    saved = Path(out)
    assert saved.exists()

    with Image.open(saved) as im:
        # Expect bbox crop, not full page fallback.
        assert im.width > 0 and im.height > 0
        assert im.width < 10 and im.height < 10


def test_stage4_keeps_picture_chunk_when_vision_missing() -> None:
    loader = DocumentLoader()
    elements = [
        {
            "id": "e0",
            "type": "picture",
            "text": "",
            "page": 3,
            "section_path": [],
        }
    ]

    chunks = loader._stage4_chunk(elements, vision_map={}, doc_stem="doc")

    assert len(chunks) == 1
    assert chunks[0]["preliminary_type"] == "picture"
    assert "Image on page 3" in chunks[0]["chunk_text_original"]


def test_stage1_image_fallback_from_pages_when_body_empty(tmp_path: Path) -> None:
    loader = DocumentLoader()

    empty_docling_json = {
        "body": {"children": []},
        "texts": [],
        "tables": [],
        "pictures": [],
        "groups": [],
        "pages": {
            "1": {
                "image": {
                    "uri": _png_data_uri(12, 12),
                }
            }
        },
    }

    loader._convert_file = lambda _path, _format: empty_docling_json  # type: ignore[method-assign]

    image_path = tmp_path / "single.jpeg"
    image_path.write_bytes(b"placeholder")

    elements, pic_info = loader._stage1_parse(image_path, tmp_path)

    assert len(elements) == 1
    assert elements[0]["type"] == "picture"
    assert elements[0]["page"] == 1
    assert len(pic_info) == 1
    assert Path(pic_info[0]["png_path"]).exists()
