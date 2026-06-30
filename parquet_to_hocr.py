#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "pyarrow",
#   "Pillow",
# ]
# ///
"""Convert OCR data from parquet files to hOCR format.

Each row produces one .html (hOCR) file. Files are grouped into
subdirectories named after the row's source label — resolved from the
HuggingFace ClassLabel names embedded in the parquet schema metadata
(e.g. "Early History of A and M", "First Five", "Fragments of Early
History") — with a per-folder sequential page number.

Usage:
    python parquet_to_hocr.py [--input DIR] [--output DIR]

Defaults:
    --input   data_downloaded/data
    --output  hocr_output
"""

import argparse
import html
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def load_label_names(parquet_files):
    """Read ClassLabel names from HF dataset metadata embedded in the schema."""
    schema = pq.ParquetFile(parquet_files[0]).schema_arrow
    meta = schema.metadata or {}
    hf_meta = meta.get(b"huggingface")
    if not hf_meta:
        return None
    info = json.loads(hf_meta)
    label_feature = info.get("info", {}).get("features", {}).get("label", {})
    return label_feature.get("names")


HOCR_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
  <meta name="ocr-system" content="dots.mocr rednote-hilab/dots.mocr" />
  <meta name="ocr-capabilities" content="ocr_page ocr_block ocr_line ocr_word" />
  <title>{title}</title>
</head>
<body>
{pages}
</body>
</html>
"""


def bbox_attr(x1, y1, x2, y2):
    return f"bbox {x1} {y1} {x2} {y2}"


def block_to_hocr(block_idx, block, page_w, page_h):
    x1, y1, x2, y2 = block["bbox"]
    category = html.escape(block.get("category", "Text"))
    text = block.get("text", "")

    # hOCR block (ocr_carea) wrapping a single line and its words
    lines = [
        f'  <div class="ocr_carea" id="block_{block_idx}" title="'
        f'{bbox_attr(x1, y1, x2, y2)}; x_source_category {category}">',
        f'    <span class="ocr_line" id="line_{block_idx}" title="{bbox_attr(x1, y1, x2, y2)}">',
    ]

    words = text.split()
    if words:
        # Distribute word boxes evenly across the block width
        n = len(words)
        block_w = max(x2 - x1, 1)
        word_w = block_w // n
        for i, word in enumerate(words):
            wx1 = x1 + i * word_w
            wx2 = x1 + (i + 1) * word_w if i < n - 1 else x2
            lines.append(
                f'      <span class="ocrx_word" id="word_{block_idx}_{i}" '
                f'title="{bbox_attr(wx1, y1, wx2, y2)}">'
                f"{html.escape(word)}</span>"
            )
    else:
        lines.append(f'      <span class="ocrx_word" id="word_{block_idx}_0" '
                     f'title="{bbox_attr(x1, y1, x2, y2)}"></span>')

    lines += ["    </span>", "  </div>"]
    return "\n".join(lines)


def row_to_hocr(row, page_id, label):
    try:
        blocks = json.loads(row["markdown"])
    except json.JSONDecodeError:
        # Truncated JSON — keep whatever complete objects parsed successfully
        raw = row["markdown"].strip()
        # Try to recover by closing the array after the last complete object
        last_close = raw.rfind("}]")
        if last_close != -1:
            raw = raw[: last_close + 2]
        else:
            last_close = raw.rfind("},")
            raw = (raw[: last_close + 1] + "]") if last_close != -1 else "[]"
        try:
            blocks = json.loads(raw)
            print(f"  Warning: truncated JSON for label={label}, recovered {len(blocks)} blocks")
        except json.JSONDecodeError:
            print(f"  Warning: unrecoverable JSON for label={label}, skipping blocks")
            blocks = []

    img_bytes = row["image"]["bytes"]
    img = Image.open(io.BytesIO(img_bytes))
    page_w, page_h = img.size

    page_lines = [
        f'<div class="ocr_page" id="page_{page_id}" title="'
        f'image "{page_id}"; label {label}; {bbox_attr(0, 0, page_w, page_h)}">'
    ]
    for i, block in enumerate(blocks):
        page_lines.append(block_to_hocr(i, block, page_w, page_h))
    page_lines.append("</div>")

    return HOCR_TEMPLATE.format(
        title=f"Page {label}",
        pages="\n".join(page_lines),
    )


def convert(input_dir: Path, output_dir: Path):
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No .parquet files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    label_names = load_label_names(parquet_files)
    if label_names:
        print(f"Resolved label names: {label_names}")
    else:
        print("Warning: no ClassLabel names found in schema metadata; using raw label values")

    page_counters = {}  # folder name -> next page number
    total = 0
    for pf in parquet_files:
        print(f"Processing {pf.name} ...")
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            label = row["label"]
            folder = label_names[label] if label_names else str(label)
            folder_slug = slugify(folder)

            page_counters.setdefault(folder_slug, 0)
            page_counters[folder_slug] += 1
            page_num = page_counters[folder_slug]

            page_id = f"{folder_slug}_p{page_num:04d}"
            hocr = row_to_hocr(row, page_id=page_id, label=folder)

            folder_dir = output_dir / folder_slug
            folder_dir.mkdir(parents=True, exist_ok=True)
            out_path = folder_dir / f"{page_id}.html"
            out_path.write_text(hocr, encoding="utf-8")
            total += 1

    print(f"Done. Wrote {total} hOCR files to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Convert parquet OCR to hOCR")
    parser.add_argument("--input", default="data_downloaded/data",
                        help="Directory containing .parquet files")
    parser.add_argument("--output", default="hocr_output",
                        help="Directory for output .hocr files")
    args = parser.parse_args()

    convert(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
