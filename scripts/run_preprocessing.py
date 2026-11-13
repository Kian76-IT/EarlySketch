
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
SAMPLED_DIR = DATA_DIR / "sampled"
SPLITS_DIR = DATA_DIR / "splits"
PROCESSED_DIR = DATA_DIR / "processed"

# Agar modul dari folder src dapat diimpor
sys.path.insert(0, str(PROJECT_ROOT))

from src.progressive_generator import (
    generate_partial_sketch,
    COMPLETION_LEVELS
)

from src.rasterizer import save_sketch


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
PADDING = 20
STROKE_WIDTH = 3

SPLIT_NAMES = ["train", "validation", "test"]


# ============================================================
# DATA LOADING
# ============================================================

def normalize_key_id(key_id):
    """Samakan format key_id dari CSV dan NDJSON."""
    value = str(key_id).strip()

    # Mengatasi key_id yang terbaca sebagai angka desimal di CSV
    if value.endswith(".0"):
        value = value[:-2]
    return value

def load_category_drawings(category):
    """
    Membaca satu file NDJSON dan menyimpan drawing
    berdasarkan key_id agar tidak membaca file berulang kali.
    """
    file_path = SAMPLED_DIR / f"{category}.ndjson"
    if not file_path.exists():
        raise FileNotFoundError(
            f"File dataset tidak ditemukan: {file_path}"
        )
    drawings = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            key_id = normalize_key_id(record["key_id"])
            drawings[key_id] = record["drawing"]

    return drawings


def load_split_csv(split_name):
    """Membaca metadata train, validation, atau test."""
    csv_path = SPLITS_DIR / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"File split tidak ditemukan: {csv_path}"
        )
    df = pd.read_csv(csv_path)
    required_columns = {"category", "key_id"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Kolom berikut tidak ditemukan di {csv_path.name}: "
            f"{missing_columns}. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    return df


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image_path):
    """Memastikan gambar tersimpan dalam RGB 224x224."""

    with Image.open(image_path) as image:
        if image.mode != "RGB":
            raise ValueError(
                f"Gambar bukan RGB: {image_path}"
            )
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Ukuran gambar tidak sesuai: {image_path}"
            )
    return True


# ============================================================
# PROCESS ONE DRAWING
# ============================================================

def process_drawing(
    drawing,
    category,
    key_id,
    split_name,
    overwrite=False
):
    """
    Membuat lima tingkat progressive sketch dari satu drawing.
    """
    drawing_id = f"{category}_{key_id}"
    total_strokes = len(drawing)
    if total_strokes == 0:
        raise ValueError(
            f"Drawing kosong: {drawing_id}"
        )
    metadata = []

    for percentage in COMPLETION_LEVELS:
        # Membuat partial drawing berdasarkan urutan stroke
        partial_drawing = generate_partial_sketch(
            drawing=drawing,
            percentage=percentage
        )
        selected_strokes = len(partial_drawing)
        # Folder berdasarkan split dan completion level
        output_dir = (
            PROCESSED_DIR
            / split_name
            / str(percentage)
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )
        output_path = (
            output_dir / f"{drawing_id}.png"
        )

        if overwrite or not output_path.exists():
            save_sketch(
                drawing=partial_drawing,
                output_path=output_path,
                image_size=IMAGE_SIZE,
                padding=PADDING,
                stroke_width=STROKE_WIDTH,
                reference_drawing=drawing
            )

        validate_image(output_path)
        metadata.append({
            "split": split_name,
            "category": category,
            "key_id": key_id,
            "drawing_id": drawing_id,
            "completion": percentage,
            "total_strokes": total_strokes,
            "selected_strokes": selected_strokes,
            "image_path": str(
                output_path.relative_to(PROJECT_ROOT)
            )
        })

    return metadata

# ============================================================
# PROCESS ONE DATASET SPLIT
# ============================================================

def process_split(
    split_name,
    limit=None,
    overwrite=False
):
    print(f"\n{'=' * 55}")
    print(f"Processing {split_name.upper()} dataset")
    print(f"{'=' * 55}")
    df = load_split_csv(split_name)
    if limit is not None:
        df = df.head(limit)
    print(f"Original drawings: {len(df)}")
    results = []
    skipped = 0

    # Membaca NDJSON sekali per kategori
    for category, category_df in df.groupby("category"):
        category = str(category).strip()
        print(f"\nCategory: {category}")
        category_drawings = load_category_drawings(
            category
        )
        for _, row in tqdm(
            category_df.iterrows(),
            total=len(category_df),
            desc=category
        ):
            key_id = normalize_key_id(row["key_id"])
            drawing = category_drawings.get(key_id)
            if drawing is None:
                print(
                    f"\n[WARNING] Drawing tidak ditemukan: "
                    f"{category}_{key_id}"
                )
                skipped += 1
                continue
            try:
                drawing_metadata = process_drawing(
                    drawing=drawing,
                    category=category,
                    key_id=key_id,
                    split_name=split_name,
                    overwrite=overwrite
                )
                results.extend(drawing_metadata)
            except Exception as error:
                print(
                    f"\n[ERROR] {category}_{key_id}: {error}"
                )
                skipped += 1

    print(f"\n{split_name.upper()} selesai.")
    print(f"Processed images: {len(results)}")
    print(f"Skipped drawings: {skipped}")

    return results


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="EarlySketch dataset preprocessing"
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Batasi jumlah drawing per split untuk testing"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Proses ulang gambar yang sudah tersedia"
    )
    args = parser.parse_args()
    all_metadata = []

    print("\nEARLYSKETCH PREPROCESSING")
    print(f"Project directory: {PROJECT_ROOT}")
    print(f"Completion levels: {COMPLETION_LEVELS}")

    for split_name in SPLIT_NAMES:
        metadata = process_split(
            split_name=split_name,
            limit=args.limit_per_split,
            overwrite=args.overwrite
        )
        all_metadata.extend(metadata)

    # Menyimpan manifest seluruh gambar
    manifest_path = (
        PROCESSED_DIR / "processed_metadata.csv"
    )
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(
        manifest_path,
        index=False
    )

    print("\n" + "=" * 55)
    print("PREPROCESSING SUMMARY")
    print("=" * 55)

    if not metadata_df.empty:
        print(
            metadata_df.groupby(
                ["split", "completion"]
            ).size()
        )
        print(
            f"\nTotal processed images: {len(metadata_df)}"
        )
    else:
        print("Tidak ada gambar yang berhasil diproses.")
    print(f"\nMetadata saved: {manifest_path}")
    print("Preprocessing selesai!")


if __name__ == "__main__":
    main()
