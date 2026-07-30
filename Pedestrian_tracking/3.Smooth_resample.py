#!/usr/bin/env python3
"""Smooth and resample pedestrian real-world trajectories.

For each track_id in each CSV:
    1. sample on the global 0.5-second grid: 0.0, 0.5, 1.0, ...;
    2. smooth x_rw/y_rw with the real observations in +/- 0.5 seconds;
    3. drop tracks with fewer than 2 sampled points;
    4. write only time_s, track_id, x_rw_smooth, y_rw_smooth.

Missing points are not interpolated or filled. Track IDs are preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "Pedestrian_data"

DEFAULT_INPUT_ROOT = DATA_ROOT / "1.Pedestrian_trajectory_raw"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "2.Pedestrian_trajectory_processed"

FRAME_COL = "frame"
TRACK_ID_COL = "track_id"
X_RW_COL = "x_rw"
Y_RW_COL = "y_rw"

OUTPUT_COLUMNS = ["time_s", "track_id", "x_rw_smooth", "y_rw_smooth"]
REQUIRED_COLUMNS = [FRAME_COL, TRACK_ID_COL, X_RW_COL, Y_RW_COL]

SOURCE_FPS = 30.0
SAMPLE_INTERVAL_S = 0.5
SMOOTH_HALF_WINDOW_S = 0.5
MIN_POINTS_IN_WINDOW = 1
MIN_OUTPUT_POINTS_PER_TRACK = 2
FLOAT_DECIMALS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smooth x_rw/y_rw with a centered time window and resample every "
            "0.5 seconds while preserving the input folder structure."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Input CSV root. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output CSV root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=SOURCE_FPS,
        help=f"Source video FPS used to convert frame to seconds. Default: {SOURCE_FPS}",
    )
    parser.add_argument(
        "--sample-interval-s",
        type=float,
        default=SAMPLE_INTERVAL_S,
        help=f"Output sample interval in seconds. Default: {SAMPLE_INTERVAL_S}",
    )
    parser.add_argument(
        "--smooth-half-window-s",
        type=float,
        default=SMOOTH_HALF_WINDOW_S,
        help=(
            "Half width of the smoothing window in seconds. "
            f"Default: {SMOOTH_HALF_WINDOW_S}"
        ),
    )
    parser.add_argument(
        "--min-points-in-window",
        type=int,
        default=MIN_POINTS_IN_WINDOW,
        help=(
            "Minimum real observations required in a smoothing window. "
            f"Default: {MIN_POINTS_IN_WINDOW}"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.source_fps <= 0:
        raise ValueError("--source-fps must be positive.")
    if args.sample_interval_s <= 0:
        raise ValueError("--sample-interval-s must be positive.")
    if args.smooth_half_window_s < 0:
        raise ValueError("--smooth-half-window-s cannot be negative.")
    if args.min_points_in_window <= 0:
        raise ValueError("--min-points-in-window must be positive.")


def iter_csv_files(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root not found: {input_root}")
    return sorted(path for path in input_root.rglob("*.csv") if path.is_file())


def check_required_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}")


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    check_required_columns(df, csv_path)

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df[FRAME_COL] = np.rint(df[FRAME_COL].to_numpy(dtype=float)).astype(np.int64)
    df[TRACK_ID_COL] = np.rint(df[TRACK_ID_COL].to_numpy(dtype=float)).astype(np.int64)
    return df


def global_sample_frames_for_track(
    frames: np.ndarray,
    *,
    sample_step_frames: int,
) -> np.ndarray:
    start_frame = int(frames[0])
    end_frame = int(frames[-1])
    first_sample_frame = (
        (start_frame + sample_step_frames - 1) // sample_step_frames
    ) * sample_step_frames
    if first_sample_frame > end_frame:
        return np.empty(0, dtype=np.int64)

    return np.arange(
        first_sample_frame,
        end_frame + 1,
        sample_step_frames,
        dtype=np.int64,
    )


def smooth_sample_track(
    track_id: int,
    track_df: pd.DataFrame,
    *,
    source_fps: float,
    sample_step_frames: int,
    half_window_frames: int,
    min_points_in_window: int,
) -> list[dict[str, float | int]]:
    track_df = track_df.sort_values(FRAME_COL, kind="mergesort")

    frames = track_df[FRAME_COL].to_numpy(dtype=np.int64)
    x_values = track_df[X_RW_COL].to_numpy(dtype=np.float64)
    y_values = track_df[Y_RW_COL].to_numpy(dtype=np.float64)

    if len(frames) == 0:
        return []

    output_rows: list[dict[str, float | int]] = []
    for sample_frame in global_sample_frames_for_track(
        frames,
        sample_step_frames=sample_step_frames,
    ):
        left_frame = sample_frame - half_window_frames
        right_frame = sample_frame + half_window_frames
        left_index = int(np.searchsorted(frames, left_frame, side="left"))
        right_index = int(np.searchsorted(frames, right_frame, side="right"))

        point_count = right_index - left_index
        if point_count < min_points_in_window:
            continue

        output_rows.append(
            {
                "time_s": sample_frame / source_fps,
                "track_id": track_id,
                "x_rw_smooth": float(np.mean(x_values[left_index:right_index])),
                "y_rw_smooth": float(np.mean(y_values[left_index:right_index])),
            }
        )

    return output_rows


def process_csv(
    csv_path: Path,
    *,
    source_fps: float,
    sample_step_frames: int,
    half_window_frames: int,
    min_points_in_window: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    df = load_csv(csv_path)
    output_rows: list[dict[str, float | int]] = []
    input_track_count = 0

    for track_id, track_df in df.groupby(TRACK_ID_COL, sort=False):
        input_track_count += 1
        track_rows = smooth_sample_track(
            int(track_id),
            track_df,
            source_fps=source_fps,
            sample_step_frames=sample_step_frames,
            half_window_frames=half_window_frames,
            min_points_in_window=min_points_in_window,
        )
        if len(track_rows) >= MIN_OUTPUT_POINTS_PER_TRACK:
            output_rows.extend(track_rows)

    if output_rows:
        out_df = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
        out_df = out_df.sort_values(
            ["time_s", "track_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        out_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    stats = {
        "input_rows": int(len(df)),
        "input_tracks": int(input_track_count),
        "output_rows": int(len(out_df)),
    }
    return out_df, stats


def output_path_for(csv_path: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / csv_path.relative_to(input_root)


def main() -> int:
    args = parse_args()
    validate_args(args)

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    sample_step_frames = max(1, int(round(args.source_fps * args.sample_interval_s)))
    half_window_frames = max(0, int(round(args.source_fps * args.smooth_half_window_s)))

    csv_paths = iter_csv_files(input_root)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    total_input_rows = 0
    total_output_rows = 0
    total_tracks = 0

    print(f"Input root: {input_root}", flush=True)
    print(f"Output root: {output_root}", flush=True)
    print(f"CSV files: {len(csv_paths)}", flush=True)
    print(
        f"Sample interval: {args.sample_interval_s:g}s "
        f"({sample_step_frames} frames)",
        flush=True,
    )
    print(
        f"Smoothing window: +/-{args.smooth_half_window_s:g}s "
        f"(+/-{half_window_frames} frames)",
        flush=True,
    )

    for file_index, csv_path in enumerate(csv_paths, start=1):
        out_df, stats = process_csv(
            csv_path,
            source_fps=args.source_fps,
            sample_step_frames=sample_step_frames,
            half_window_frames=half_window_frames,
            min_points_in_window=args.min_points_in_window,
        )
        out_path = output_path_for(csv_path, input_root, output_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(
            out_path,
            index=False,
            encoding="utf-8-sig",
            float_format=f"%.{FLOAT_DECIMALS}f",
        )

        total_input_rows += stats["input_rows"]
        total_output_rows += stats["output_rows"]
        total_tracks += stats["input_tracks"]

        if file_index <= 5 or file_index % 50 == 0 or file_index == len(csv_paths):
            print(
                f"[{file_index}/{len(csv_paths)}] {csv_path.name}: "
                f"input_rows={stats['input_rows']} "
                f"tracks={stats['input_tracks']} "
                f"output_rows={stats['output_rows']}",
                flush=True,
            )

    print("Done.", flush=True)
    print(f"Files: {len(csv_paths)}", flush=True)
    print(f"Input rows: {total_input_rows}", flush=True)
    print(f"Input tracks: {total_tracks}", flush=True)
    print(f"Output rows: {total_output_rows}", flush=True)
    print(f"Output folder: {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
