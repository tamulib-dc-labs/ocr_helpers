#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Convert ABBYY FineReader XML (plain or .gz) to hOCR.

Produces one .hocr file per page, with block, line, and word bounding boxes.
Word boundaries are detected via the wordFirst="1" attribute on charParams.

Usage:
    python abbyy_to_hocr.py INPUT [--output DIR]

INPUT may be a plain .xml file or a .gz compressed ABBYY file.

Defaults:
    --output  hocr_output  (created next to the input file if not specified)
"""

import argparse
import gzip
import html
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "http://www.abbyy.com/FineReader_xml/FineReader10-schema-v1.xml"
TAG = lambda name: f"{{{NS}}}{name}"  # noqa: E731

HOCR_HEADER = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
  <meta name="ocr-system" content="ABBYY FineReader" />
  <meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_line ocrx_word" />
  <title>{title}</title>
</head>
<body>
"""

HOCR_FOOTER = "</body>\n</html>\n"


def bbox(l, t, r, b):
    return f"bbox {l} {t} {r} {b}"


def chars_to_words(char_elements):
    """Group charParams elements into words using wordFirst attribute."""
    words = []
    current = []
    for ch in char_elements:
        text = ch.text or ""
        if not text.strip() and not current:
            continue
        is_first = ch.get("wordFirst") == "1"
        if is_first and current:
            words.append(current)
            current = []
        current.append(ch)
    if current:
        words.append(current)
    return words


def word_bbox(chars):
    ls = [int(c.get("l", 0)) for c in chars]
    ts = [int(c.get("t", 0)) for c in chars]
    rs = [int(c.get("r", 0)) for c in chars]
    bs = [int(c.get("b", 0)) for c in chars]
    return min(ls), min(ts), max(rs), max(bs)


def word_text(chars):
    return "".join(c.text or "" for c in chars)


def render_page(page_elem, page_num, stem):
    pw = page_elem.get("width", "0")
    ph = page_elem.get("height", "0")
    page_id = f"{stem}_p{page_num:04d}"

    lines_out = [
        f'<div class="ocr_page" id="page_{page_num}" '
        f"title=\"image '{page_id}'; {bbox(0, 0, pw, ph)}\">",
    ]

    block_idx = 0
    line_idx = 0
    word_idx = 0

    for block in page_elem.findall(TAG("block")):
        block_type = block.get("blockType", "Text")
        bl = block.get("l", "0")
        bt = block.get("t", "0")
        br = block.get("r", "0")
        bb_ = block.get("b", "0")

        lines_out.append(
            f'  <div class="ocr_carea" id="block_{block_idx}" '
            f'title="{bbox(bl, bt, br, bb_)}; x_block_type {html.escape(block_type)}">'
        )

        for par in block.findall(f".//{TAG('par')}"):
            for line in par.findall(TAG("line")):
                ll = line.get("l", bl)
                lt = line.get("t", bt)
                lr = line.get("r", br)
                lb = line.get("b", bb_)
                baseline = line.get("baseline", "")
                title_extra = f"; baseline {baseline}" if baseline else ""

                lines_out.append(
                    f'    <span class="ocr_line" id="line_{line_idx}" '
                    f'title="{bbox(ll, lt, lr, lb)}{title_extra}">'
                )

                all_chars = line.findall(f".//{TAG('charParams')}")
                words = chars_to_words(all_chars)

                for word_chars in words:
                    wl, wt, wr, wb = word_bbox(word_chars)
                    text = html.escape(word_text(word_chars).strip())
                    if not text:
                        continue
                    conf_vals = [
                        int(c.get("charConfidence", "0"))
                        for c in word_chars
                        if c.get("charConfidence")
                    ]
                    conf = int(sum(conf_vals) / len(conf_vals)) if conf_vals else 0
                    lines_out.append(
                        f'      <span class="ocrx_word" id="word_{word_idx}" '
                        f'title="{bbox(wl, wt, wr, wb)}; x_wconf {conf}">'
                        f"{text}</span>"
                    )
                    word_idx += 1

                lines_out.append("    </span>")
                line_idx += 1

        lines_out.append("  </div>")
        block_idx += 1

    lines_out.append("</div>")
    return "\n".join(lines_out)


def convert(input_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem.replace(".xml", "").replace("_abbyy", "")

    print(f"Parsing {input_path.name} ...")
    if input_path.suffix == ".gz":
        with gzip.open(input_path, "rb") as fh:
            tree = ET.parse(fh)
    else:
        tree = ET.parse(input_path)

    root = tree.getroot()
    pages = root.findall(TAG("page"))
    print(f"Found {len(pages)} pages")

    for i, page in enumerate(pages, start=1):
        page_html = render_page(page, i, stem)
        out = HOCR_HEADER.format(title=f"{stem} page {i}") + page_html + "\n" + HOCR_FOOTER
        out_path = output_dir / f"{stem}_p{i:04d}.html"
        out_path.write_text(out, encoding="utf-8")

    print(f"Done. Wrote {len(pages)} hOCR files to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Convert ABBYY XML to hOCR")
    parser.add_argument("input", help="ABBYY .xml or .gz file, or a directory containing them")
    parser.add_argument(
        "--output",
        help="Output directory (default: hocr_output next to input)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    if input_path.is_dir():
        files = sorted(
            f for f in input_path.iterdir()
            if f.suffix in {".gz", ".xml"} or f.name.endswith("_abbyy")
        )
        if not files:
            print(f"Error: no .xml or .gz files found in {input_path}", file=sys.stderr)
            sys.exit(1)
        output_dir = Path(args.output) if args.output else input_path / "hocr_output"
        for f in files:
            convert(f, output_dir)
    else:
        output_dir = Path(args.output) if args.output else input_path.parent / "hocr_output"
        convert(input_path, output_dir)


if __name__ == "__main__":
    main()
