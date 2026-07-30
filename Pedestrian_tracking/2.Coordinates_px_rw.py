#!/usr/bin/env python3
"""Add pixel and real-world coordinates to pedestrian trajectory CSV files in place.

Input CSV columns:
    frame, track_id, confidence, x_px_min, x_px_max, y_px_min, y_px_max

Added columns:
    x_px = (x_px_min + x_px_max) / 2
    y_px = y_px_max
    x_rw, y_rw = homography transform of x_px, y_px
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "Pedestrian_data"

DEFAULT_INPUT_ROOT = DATA_ROOT / "1.Pedestrian_trajectory_raw"
DEFAULT_HOMOGRAPHY_PATH = ROOT / "georeference" / "homography_matrix.csv"

REQUIRED_COLUMNS = [
    "frame",
    "track_id",
    "confidence",
    "x_px_min",
    "x_px_max",
    "y_px_min",
    "y_px_max",
]
ADDED_COLUMNS = ["x_px", "y_px", "x_rw", "y_rw"]


def load_homography(matrix_path: Path) -> np.ndarray:
    """Read the 3x3 pixel-to-real-world homography matrix."""
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Homography matrix not found: {matrix_path}")

    h = np.loadtxt(matrix_path, delimiter=",", dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 homography matrix, got shape {h.shape}.")
    return h


def validate_columns(csv_path: Path, fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError(f"CSV header not found: {csv_path}")

    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {', '.join(missing)}"
        )


def pixel_to_real_world(x_px: float, y_px: float, h: np.ndarray) -> tuple[float, float]:
    """Apply H @ [x_px, y_px, 1] to one pixel coordinate."""
    denominator = h[2, 0] * x_px + h[2, 1] * y_px + h[2, 2]
    if abs(denominator) < 1e-12:
        raise ZeroDivisionError(
            f"Homography denominator is near zero for pixel ({x_px}, {y_px})."
        )

    x_rw = (h[0, 0] * x_px + h[0, 1] * y_px + h[0, 2]) / denominator
    y_rw = (h[1, 0] * x_px + h[1, 1] * y_px + h[1, 2]) / denominator
    return float(x_rw), float(y_rw)


def format_float(value: float) -> str:
    return f"{value:.3f}"


def output_fieldnames(input_fieldnames: list[str]) -> list[str]:
    """Keep original columns first and append/rewrite coordinate columns at the end."""
    kept = [column for column in input_fieldnames if column not in ADDED_COLUMNS]
    return kept + ADDED_COLUMNS


def temporary_csv_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.name}.tmp")


def process_csv(csv_path: Path, h: np.ndarray) -> int:
    """Read one trajectory CSV and overwrite it with pixel/world coordinates."""
    temp_path = temporary_csv_path(csv_path)
    if temp_path.exists():
        temp_path.unlink()

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as src:
            reader = csv.DictReader(src)
            validate_columns(csv_path, reader.fieldnames)
            fieldnames = output_fieldnames(list(reader.fieldnames or []))

            with temp_path.open("w", encoding="utf-8-sig", newline="") as dst:
                writer = csv.DictWriter(dst, fieldnames=fieldnames)
                writer.writeheader()

                row_count = 0
                for row_count, row in enumerate(reader, start=1):
                    x_px_min = float(row["x_px_min"])
                    x_px_max = float(row["x_px_max"])
                    y_px_max = float(row["y_px_max"])

                    x_px = (x_px_min + x_px_max) / 2.0
                    y_px = y_px_max
                    x_rw, y_rw = pixel_to_real_world(x_px, y_px, h)

                    row["x_px"] = format_float(x_px)
                    row["y_px"] = format_float(y_px)
                    row["x_rw"] = format_float(x_rw)
                    row["y_rw"] = format_float(y_rw)
                    writer.writerow(
                        {column: row.get(column, "") for column in fieldnames}
                    )

        temp_path.replace(csv_path)
        return row_count
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def iter_input_csvs(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root not found: {input_root}")
    return sorted(path for path in input_root.rglob("*.csv") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Add bottom-center pixel coordinates and homography-based real-world "
            "coordinates to every pedestrian trajectory CSV in place."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Raw trajectory CSV root. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--homography",
        type=Path,
        default=DEFAULT_HOMOGRAPHY_PATH,
        help=f"3x3 homography matrix CSV. Default: {DEFAULT_HOMOGRAPHY_PATH}",
    )
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    homography_path = args.homography.resolve()

    h = load_homography(homography_path)
    csv_paths = iter_input_csvs(input_root)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under: {input_root}")

    total_rows = 0
    for csv_path in csv_paths:
        row_count = process_csv(csv_path, h)
        total_rows += row_count
        print(f"Updated {csv_path} ({row_count} rows)")

    print(
        f"Done. Updated {len(csv_paths)} CSV files in place and processed "
        f"{total_rows} rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
