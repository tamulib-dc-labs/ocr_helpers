# ocr-helpers

Utilities for converting OCR output to [hOCR](https://kba.github.io/hocr-spec/) format.

Both scripts use [uv inline script metadata](https://docs.astral.sh/uv/guides/scripts/) and require no manual environment setup.

## Scripts

### `parquet_to_hocr.py`

Converts OCR data stored in Parquet files (e.g. from HuggingFace datasets) to hOCR HTML files.

Expects each row to have:
- `image` — image bytes
- `markdown` — JSON array of OCR blocks with `bbox` (`[x1, y1, x2, y2]`), `category`, and `text`
- `label` — classification label (used as metadata, not as a unique page ID)

Produces one `.html` file per row, named `{shard}_{row_index}.html`. Word bounding boxes are estimated by dividing each block's width evenly across its words, since the source data only provides block-level coordinates.

```bash
uv run parquet_to_hocr.py [--input DIR] [--output DIR]

# Defaults:
#   --input   data_downloaded/data
#   --output  hocr_output
```

### `abbyy_to_hocr.py`

Converts ABBYY FineReader XML to hOCR HTML files. Accepts plain `.xml` or `.gz` compressed input.

Produces one `.html` file per page with block, line, and word bounding boxes. Word boundaries are detected from the `wordFirst="1"` attribute on each character element. Per-word confidence scores (`x_wconf`) are averaged from character-level confidence values.

```bash
uv run abbyy_to_hocr.py INPUT [--output DIR]

# Defaults:
#   --output  hocr_output/  (created next to the input file)
```

## Requirements

- [uv](https://docs.astral.sh/uv/) — dependencies are declared inline and installed automatically on first run
- Python 3.11+
