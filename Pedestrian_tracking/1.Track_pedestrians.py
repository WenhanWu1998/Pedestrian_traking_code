#!/usr/bin/env python3
"""Batch-extract pedestrian trajectories with YOLOv12 + BoT-SORT + ReID.

The trajectory CSV contains seven columns: frame, track_id, confidence,
x_px_min, x_px_max, y_px_min, y_px_max. Frame is the zero-based
source-frame index; bounding-box coordinates are measured in source pixels.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path


os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

ROOT = Path(__file__).resolve().parent
YOLO_REPO = ROOT / "yolov12"
BOXMOT_REPO = ROOT / "boxmot"
SOURCE_DIR = ROOT.parent / "Video_data"
YOLO_WEIGHTS = YOLO_REPO / "weights" / "best.pt"
REID_WEIGHTS = BOXMOT_REPO / "models" / "osnet_x1_0_msmt17.pt"
OUTPUT_ROOT = (
    ROOT.parent
    / "Pedestrian_data"
    / "1.Pedestrian_trajectory_raw"
)
VIDEO_DATE_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})(?:_|$)"
)
# ROI coordinates use a top-left origin and are valid for this size only.
ROI_FRAME_SIZE = (1920, 1080)
ROI_VERTICES: tuple[tuple[int, int], ...] = (
    (0, 943),
    (583, 793),
    (1012, 658),
    (1377, 530),
    (1582, 444),
    (1851, 477),
    (1839, 657),
    (1809, 861),
    (1759, 1079),
    (0, 1079),
)
FFMPEG_FALLBACK = Path(
    r"D:\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
)
FRAME_STRIDE = 3
IMAGE_SIZE = 960
CONFIDENCE_THRESHOLD = 0.10
NMS_IOU_THRESHOLD = 0.70
MAX_DETECTIONS = 300
DEVICE = "0"
USE_HALF = True
SAVE_ANNOTATED_VIDEO = False
TRACK_BUFFER = 150
OUTPUT_WIDTH = 960
OUTPUT_FPS = 10.0
VIDEO_CRF = 24
BOTSORT_SETTINGS: dict[str, object] = {
    "track_high_thresh": 0.6,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.7,
    "match_thresh": 0.8,
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.25,
    "cmc_method": None,
    "fuse_first_associate": False,
    "with_reid": True,
}

# Force imports to use the local YOLOv12 and BoxMOT repositories.
sys.path.insert(0, str(BOXMOT_REPO))
sys.path.insert(0, str(YOLO_REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
import boxmot  # noqa: E402
import ultralytics  # noqa: E402
from boxmot.trackers.tracker_zoo import create_tracker  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def tracker_device() -> str:
    if DEVICE.lower() == "cpu":
        return "cpu"
    return f"cuda:{DEVICE.split(',')[0]}"


def validate_local_import(
    module_file: str,
    repository: Path,
    name: str,
) -> None:
    imported = Path(module_file).resolve()
    try:
        imported.relative_to(repository.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"{name} was imported from {imported}, not {repository}"
        ) from exc


def find_ffmpeg() -> Path:
    executable = shutil.which("ffmpeg")
    if executable:
        return Path(executable)
    if FFMPEG_FALLBACK.is_file():
        return FFMPEG_FALLBACK
    raise FileNotFoundError(
        "ffmpeg was not found on PATH or at "
        f"{FFMPEG_FALLBACK}"
    )


def validate_setup() -> Path | None:
    if Path(sys.executable).parent.name.lower() != "track":
        raise RuntimeError(
            "Please run this script in the Conda 'track' environment. "
            f"Current Python: {sys.executable}"
        )

    required = [
        (SOURCE_DIR, "input video directory"),
        (YOLO_WEIGHTS, "YOLOv12 best weights"),
        (REID_WEIGHTS, "ReID weights"),
        (YOLO_REPO, "local YOLOv12 repository"),
        (BOXMOT_REPO, "local BoxMOT repository"),
    ]
    for path, description in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")

    if DEVICE.lower() != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    if FRAME_STRIDE <= 0 or IMAGE_SIZE <= 0 or MAX_DETECTIONS <= 0:
        raise ValueError("Detection size, stride, and max detections must be positive.")
    if not 0.0 <= CONFIDENCE_THRESHOLD <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    if not 0.0 <= NMS_IOU_THRESHOLD <= 1.0:
        raise ValueError("NMS IoU threshold must be between 0 and 1.")
    if TRACK_BUFFER <= 0:
        raise ValueError("Track buffer setting is invalid.")
    if SAVE_ANNOTATED_VIDEO:
        if OUTPUT_WIDTH < 2 or OUTPUT_FPS <= 0:
            raise ValueError("Output width/FPS settings are invalid.")
        if not 0 <= VIDEO_CRF <= 51:
            raise ValueError("Video CRF setting is invalid.")

    validate_local_import(ultralytics.__file__, YOLO_REPO, "Ultralytics")
    validate_local_import(boxmot.__file__, BOXMOT_REPO, "BoxMOT")
    return find_ffmpeg() if SAVE_ANNOTATED_VIDEO else None


def botsort_parameters(fps: float) -> dict[str, object]:
    parameters = dict(BOTSORT_SETTINGS)
    parameters["frame_rate"] = int(round(fps))
    parameters["track_buffer"] = TRACK_BUFFER
    return parameters


def build_botsort(
    fps: float,
    reid_model: object | None = None,
) -> object:
    return create_tracker(
        tracker_type="botsort",
        reid_weights=REID_WEIGHTS,
        reid_model=reid_model,
        device=tracker_device(),
        half=USE_HALF and DEVICE.lower() != "cpu",
        per_class=False,
        evolve_param_dict=botsort_parameters(fps),
        tracker_backend="python",
    )


def detections_from_result(result) -> np.ndarray:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidence = boxes.conf.detach().cpu().numpy().reshape(-1, 1)
    class_id = boxes.cls.detach().cpu().numpy().reshape(-1, 1)
    return np.hstack((xyxy, confidence, class_id)).astype(np.float32)


def build_roi_polygon() -> np.ndarray:
    """Build the OpenCV polygon from the coordinates embedded above."""
    polygon = np.asarray(
        ROI_VERTICES,
        dtype=np.float32,
    ).reshape((-1, 1, 2))
    if len(polygon) < 3 or cv2.contourArea(polygon) <= 0.0:
        raise ValueError("Embedded ROI polygon is invalid.")
    return polygon


def validate_roi_polygon(
    polygon: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> None:
    """Require the calibrated resolution and valid ROI coordinates."""
    if (frame_width, frame_height) != ROI_FRAME_SIZE:
        raise ValueError(
            f"ROI is calibrated for {ROI_FRAME_SIZE[0]}x"
            f"{ROI_FRAME_SIZE[1]}, but video is "
            f"{frame_width}x{frame_height}."
        )
    points = polygon.reshape((-1, 2))
    invalid = (
        (points[:, 0] < 0.0)
        | (points[:, 0] > frame_width - 1)
        | (points[:, 1] < 0.0)
        | (points[:, 1] > frame_height - 1)
    )
    if np.any(invalid):
        invalid_points = points[invalid].tolist()
        raise ValueError(
            f"ROI vertices outside {frame_width}x{frame_height} frame: "
            f"{invalid_points}"
        )


def filter_boxes_by_roi(
    boxes: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """Keep boxes whose bottom-center point is inside or on the ROI."""
    if len(boxes) == 0:
        return boxes
    foot_points = np.column_stack(
        (
            (boxes[:, 0] + boxes[:, 2]) / 2.0,
            boxes[:, 3],
        )
    )
    keep = np.fromiter(
        (
            cv2.pointPolygonTest(
                polygon,
                (float(x), float(y)),
                False,
            )
            >= 0.0
            for x, y in foot_points
        ),
        dtype=bool,
        count=len(foot_points),
    )
    return boxes[keep]


def clip_boxes(
    boxes: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Clip box coordinates to valid source-image pixels."""
    if len(boxes) == 0:
        return boxes
    clipped = boxes.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(frame_width - 1))
    clipped[:, 2] = np.clip(clipped[:, 2], 0.0, float(frame_width - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(frame_height - 1))
    clipped[:, 3] = np.clip(clipped[:, 3], 0.0, float(frame_height - 1))
    return clipped


def color_for_id(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 37) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def prune_finished_trails(
    tracker: object,
    trails: dict[int, deque[tuple[int, int]]],
) -> None:
    """Remove a trail only after BoT-SORT can no longer recover its ID."""
    live_tracks = (
        tracker.get_active_tracks_for_display()
        + tracker.get_lost_tracks_for_display()
    )
    live_ids = {
        tracker.get_track_id_for_display(track)
        for track in live_tracks
    }
    for track_id in tuple(trails):
        if track_id not in live_ids:
            trails.pop(track_id, None)


def draw_tracks(
    frame: np.ndarray,
    rows: np.ndarray,
    trails: dict[int, deque[tuple[int, int]]],
    frame_index: int,
    source_fps: float,
) -> np.ndarray:
    """Draw the complete trajectory of every pedestrian visible now."""
    annotated = frame.copy()
    for row in rows:
        x1, y1, x2, y2, track_id, confidence, _, _ = row[:8]
        identity = int(track_id)
        color = color_for_id(identity)
        foot = (
            int(round((float(x1) + float(x2)) / 2.0)),
            int(round(float(y2))),
        )
        trails[identity].append(foot)

        cv2.rectangle(
            annotated,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            2,
        )
        cv2.circle(annotated, foot, 5, color, -1)
        cv2.putText(
            annotated,
            f"ID {identity} {float(confidence):.2f}",
            (int(round(x1)), max(24, int(round(y1)) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            2,
            cv2.LINE_AA,
        )
        points = np.asarray(list(trails[identity]), dtype=np.int32)
        if len(points) > 1:
            cv2.polylines(
                annotated,
                [points.reshape((-1, 1, 2))],
                False,
                color,
                2,
                cv2.LINE_AA,
            )

    header = (
        "YOLOv12 best + BoT-SORT + ReID | "
        f"time={frame_index / source_fps:.2f}s | active={len(rows)}"
    )
    cv2.rectangle(annotated, (0, 0), (1100, 42), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        header,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def output_geometry(
    source_width: int,
    source_height: int,
    source_fps: float,
    requested_width: int,
    requested_fps: float,
) -> tuple[int, int, float, int]:
    width = min(source_width, requested_width)
    width -= width % 2
    height = int(round(source_height * width / source_width))
    height -= height % 2
    step = max(
        1,
        int(round(source_fps / min(source_fps, requested_fps))),
    )
    return width, height, source_fps / step, step


class H264Writer:
    def __init__(
        self,
        ffmpeg: Path,
        path: Path,
        width: int,
        height: int,
        fps: float,
        crf: int,
    ) -> None:
        self.path = path
        self.closed = False
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        creation_flags = (
            subprocess.CREATE_NO_WINDOW
            if sys.platform == "win32"
            else 0
        )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creation_flags,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg input pipe is unavailable.")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        error_output = (
            self.process.stderr.read().decode("utf-8", errors="replace")
            if self.process.stderr is not None
            else ""
        )
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg failed for {self.path}: {error_output.strip()}"
            )


def discover_videos() -> list[Path]:
    videos = sorted(
        (
            path
            for path in SOURCE_DIR.iterdir()
            if path.is_file() and path.suffix.lower() == ".mp4"
        ),
        key=lambda path: path.name.lower(),
    )
    if not videos:
        raise FileNotFoundError(f"No MP4 videos found in: {SOURCE_DIR}")
    return videos


def date_from_video_name(source: Path) -> date:
    """Read the leading YYYY-MM-DD date from an input video name."""
    match = VIDEO_DATE_PATTERN.match(source.stem)
    if match is None:
        raise ValueError(
            "Input video name must start with YYYY-MM-DD: "
            f"{source.name}"
        )
    try:
        return datetime.strptime(
            match.group("date"),
            "%Y-%m-%d",
        ).date()
    except ValueError as exc:
        raise ValueError(
            f"Invalid date in input video name: {source.name}"
        ) from exc


def build_batch_output_dir(videos: list[Path]) -> Path:
    """Create YYYYMMDD-YYYYMMDD from the earliest/latest video dates."""
    video_dates = [date_from_video_name(video) for video in videos]
    first_date = min(video_dates)
    last_date = max(video_dates)
    output_dir = OUTPUT_ROOT / (
        f"{first_date:%Y%m%d}-{last_date:%Y%m%d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def prepare_csv_paths(
    source: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    csv_path = output_dir / f"{source.stem}.csv"
    temp_csv = output_dir / f"{source.stem}.part.csv"
    temp_csv.unlink(missing_ok=True)
    return csv_path, temp_csv


def prepare_video_paths(
    source: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    video_path = output_dir / source.name
    temp_video = output_dir / f"{source.stem}.part.mp4"
    temp_video.unlink(missing_ok=True)
    return video_path, temp_video


def process_video(
    source: Path,
    output_dir: Path,
    ffmpeg: Path | None,
    model: YOLO,
    roi_polygon: np.ndarray,
    reid_model: object | None,
    video_index: int,
    video_count: int,
) -> tuple[int, object]:
    csv_path, temp_csv = prepare_csv_paths(
        source,
        output_dir,
    )
    video_path: Path | None = None
    temp_video: Path | None = None
    if SAVE_ANNOTATED_VIDEO:
        video_path, temp_video = prepare_video_paths(
            source,
            output_dir,
        )
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {source}")

    progress: tqdm | None = None
    video_writer: H264Writer | None = None
    frames_processed = 0
    success = False
    tracker = None

    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if source_width < 2 or source_height < 2 or total_frames <= 0:
            raise RuntimeError(
                f"Invalid video metadata for {source}: "
                f"{source_width}x{source_height}, frames={total_frames}"
            )
        validate_roi_polygon(roi_polygon, source_width, source_height)

        sampled_frame_limit = (
            total_frames + FRAME_STRIDE - 1
        ) // FRAME_STRIDE
        effective_fps = source_fps / FRAME_STRIDE
        if SAVE_ANNOTATED_VIDEO:
            output_width, output_height, output_fps, output_step = (
                output_geometry(
                    source_width,
                    source_height,
                    effective_fps,
                    OUTPUT_WIDTH,
                    OUTPUT_FPS,
                )
            )
        use_half = USE_HALF and DEVICE.lower() != "cpu"

        print("=" * 78, flush=True)
        print(
            f"Video {video_index}/{video_count}: {source.name}",
            flush=True,
        )
        print(
            f"Frames: {total_frames}; source FPS: {source_fps:.3f}; "
            f"stride: {FRAME_STRIDE}; trajectory rate: "
            f"{effective_fps:.3f} Hz; sampled frames: "
            f"{sampled_frame_limit}; imgsz: {IMAGE_SIZE}",
            flush=True,
        )
        print(
            f"BoT-SORT: frame_rate={int(round(effective_fps))}, "
            f"track_buffer={TRACK_BUFFER}, cmc_method=None",
            flush=True,
        )
        if SAVE_ANNOTATED_VIDEO:
            print(f"Video output: {video_path}", flush=True)
        else:
            print("Annotated video output: disabled", flush=True)
        print(f"Trajectory output: {csv_path}", flush=True)
        print("=" * 78, flush=True)

        tracker = build_botsort(
            effective_fps,
            reid_model=reid_model,
        )
        trails: dict[int, deque[tuple[int, int]]] | None = None
        if SAVE_ANNOTATED_VIDEO:
            trails = defaultdict(deque)
        progress = tqdm(
            total=sampled_frame_limit,
            desc=f"[{video_index}/{video_count}] {source.stem}",
            unit="sample",
        )
        if SAVE_ANNOTATED_VIDEO:
            if ffmpeg is None or temp_video is None:
                raise RuntimeError("Video output was enabled without FFmpeg.")
            video_writer = H264Writer(
                ffmpeg,
                temp_video,
                output_width,
                output_height,
                output_fps,
                VIDEO_CRF,
            )

        with temp_csv.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            trajectory_writer = csv.writer(csv_file)
            trajectory_writer.writerow(
                [
                    "frame",
                    "track_id",
                    "confidence",
                    "x_px_min",
                    "x_px_max",
                    "y_px_min",
                    "y_px_max",
                ]
            )

            with torch.inference_mode():
                for sample_index in range(sampled_frame_limit):
                    source_frame_index = sample_index * FRAME_STRIDE
                    if sample_index > 0:
                        for _ in range(FRAME_STRIDE - 1):
                            if not cap.grab():
                                raise RuntimeError(
                                    f"Video decoding stopped early: {source}"
                                )
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(
                            f"Video decoding stopped early: {source}"
                        )

                    result = model.predict(
                        frame,
                        imgsz=IMAGE_SIZE,
                        conf=CONFIDENCE_THRESHOLD,
                        iou=NMS_IOU_THRESHOLD,
                        classes=[0],
                        max_det=MAX_DETECTIONS,
                        device=DEVICE,
                        half=use_half,
                        verbose=False,
                    )[0]
                    detections = clip_boxes(
                        detections_from_result(result),
                        source_width,
                        source_height,
                    )
                    detections = filter_boxes_by_roi(
                        detections,
                        roi_polygon,
                    )
                    rows = np.asarray(
                        tracker.update(detections.copy(), frame),
                        dtype=np.float32,
                    )
                    if rows.size == 0:
                        rows = np.empty((0, 8), dtype=np.float32)
                    rows = filter_boxes_by_roi(
                        clip_boxes(rows, source_width, source_height),
                        roi_polygon,
                    )
                    if SAVE_ANNOTATED_VIDEO:
                        if trails is None:
                            raise RuntimeError("Video trail state is unavailable.")
                        prune_finished_trails(tracker, trails)

                    for row in rows:
                        x1, y1, x2, y2, track_id, confidence = row[:6]
                        trajectory_writer.writerow(
                            [
                                source_frame_index,
                                int(track_id),
                                f"{float(confidence):.2f}",
                                f"{float(x1):.2f}",
                                f"{float(x2):.2f}",
                                f"{float(y1):.2f}",
                                f"{float(y2):.2f}",
                            ]
                        )

                    if (
                        SAVE_ANNOTATED_VIDEO
                        and sample_index % output_step == 0
                    ):
                        if trails is None or video_writer is None:
                            raise RuntimeError(
                                "Video output state is unavailable."
                            )
                        annotated = draw_tracks(
                            frame,
                            rows,
                            trails,
                            source_frame_index,
                            source_fps,
                        )
                        video_writer.write(
                            cv2.resize(
                                annotated,
                                (output_width, output_height),
                                interpolation=cv2.INTER_AREA,
                            )
                        )

                    frames_processed += 1
                    progress.update(1)

        if frames_processed != sampled_frame_limit:
            raise RuntimeError(
                f"Incomplete processing for {source}: "
                f"{frames_processed}/{sampled_frame_limit} sampled frames"
            )
        if SAVE_ANNOTATED_VIDEO:
            if video_writer is None:
                raise RuntimeError("Video writer was not initialized.")
            video_writer.close()
            video_writer = None
        success = True
    finally:
        cap.release()
        if progress is not None:
            progress.close()
        if video_writer is not None:
            video_writer.close()
        if not success:
            if temp_video is not None:
                temp_video.unlink(missing_ok=True)
            temp_csv.unlink(missing_ok=True)

    if temp_video is not None:
        if video_path is None:
            raise RuntimeError("Final video output path is unavailable.")
        temp_video.replace(video_path)
    temp_csv.replace(csv_path)
    print(
        f"Completed {video_index}/{video_count}: "
        f"{frames_processed} sampled frames",
        flush=True,
    )
    if tracker is None or tracker.model is None:
        raise RuntimeError("BoT-SORT ReID model was not initialized.")
    return frames_processed, tracker.model


def run() -> None:
    ffmpeg = validate_setup()
    videos = discover_videos()
    output_dir = build_batch_output_dir(videos)
    roi_polygon = build_roi_polygon()

    print("=" * 78, flush=True)
    print("Batch pedestrian trajectory extraction", flush=True)
    print(f"Input: {SOURCE_DIR} ({len(videos)} videos)", flush=True)
    print(f"Detector: {YOLO_WEIGHTS}", flush=True)
    print(f"Tracker: BoT-SORT + ReID ({REID_WEIGHTS})", flush=True)
    print(
        f"ROI: {len(roi_polygon)} embedded vertices; "
        "bottom-center boundary included",
        flush=True,
    )
    print(f"Output folder: {output_dir}", flush=True)
    print(
        "Annotated video: "
        f"{'enabled' if SAVE_ANNOTATED_VIDEO else 'disabled'}",
        flush=True,
    )
    print("=" * 78, flush=True)

    model = YOLO(str(YOLO_WEIGHTS))
    shared_reid_model: object | None = None
    total_frames_processed = 0
    for video_index, source in enumerate(videos, start=1):
        frames_processed, shared_reid_model = process_video(
            source,
            output_dir,
            ffmpeg,
            model,
            roi_polygon,
            shared_reid_model,
            video_index,
            len(videos),
        )
        total_frames_processed += frames_processed

    print("=" * 78, flush=True)
    print(
        f"Batch completed: {len(videos)} videos, "
        f"{total_frames_processed} sampled frames",
        flush=True,
    )
    print("=" * 78, flush=True)


if __name__ == "__main__":
    run()
