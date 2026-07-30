# Pedestrian Trajectory Processing Code

This repository contains the processing code used to extract, georeference, and smooth pedestrian trajectories from fixed-camera street videos in Kabukicho, Tokyo. The workflow converts raw MP4 videos into pedestrian bounding-box trajectories, derives image and real-world coordinates, and produces smoothed trajectories sampled at 0.5-second intervals.

## Repository Structure

```text
Code/
+-- Video_data/
|   +-- *.mp4
+-- Pedestrian_data/
|   +-- 1.Pedestrian_trajectory_raw/
|   +-- 2.Pedestrian_trajectory_processed/
+-- Pedestrian_tracking/
    +-- 1.Track_pedestrians.py
    +-- 2.Coordinates_px_rw.py
    +-- 3.Smooth_resample.py
    +-- georeference/
    |   +-- homography_matrix.csv
    +-- yolov12/
    |   +-- weights/best.pt
    +-- boxmot/
        +-- models/osnet_x1_0_msmt17.pt
```

## Input Data

Place the source videos in:

```text
Code/Video_data/
```

The tracking script scans this folder for `.mp4` files. Each video filename must begin with a date in the format:

```text
YYYY-MM-DD
```

For example:

```text
2025-08-20_1200.mp4
```

The region of interest is calibrated for 1920 x 1080 videos. If videos with a different resolution are used, the ROI vertices in `Pedestrian_tracking/1.Track_pedestrians.py` must be recalibrated.

## Processing Workflow

Run the scripts from the `Code` folder, or call them with their full paths.

### 1. Track Pedestrians From Video

```bash
python Pedestrian_tracking/1.Track_pedestrians.py
```

This script uses YOLOv12 for pedestrian detection and BoT-SORT with ReID for tracking. It reads all MP4 files in `Video_data/` and writes raw trajectory CSV files to:

```text
Code/Pedestrian_data/1.Pedestrian_trajectory_raw/YYYYMMDD-YYYYMMDD/
```

Each raw CSV contains:

```text
frame, track_id, confidence, x_px_min, x_px_max, y_px_min, y_px_max
```

### 2. Convert Bounding Boxes to Pixel and Real-World Coordinates

```bash
python Pedestrian_tracking/2.Coordinates_px_rw.py
```

This script updates the raw trajectory CSV files in place. It computes the pedestrian location as the bottom-center point of each bounding box:

```text
x_px = (x_px_min + x_px_max) / 2
y_px = y_px_max
```

It then applies the homography matrix:

```text
Code/Pedestrian_tracking/georeference/homography_matrix.csv
```

and appends:

```text
x_px, y_px, x_rw, y_rw
```

The four added coordinate columns are saved with three decimal places.

### 3. Smooth and Resample Trajectories

```bash
python Pedestrian_tracking/3.Smooth_resample.py
```

This script reads the georeferenced raw CSV files from:

```text
Code/Pedestrian_data/1.Pedestrian_trajectory_raw/
```

and writes processed trajectories to:

```text
Code/Pedestrian_data/2.Pedestrian_trajectory_processed/
```

The folder structure of the raw trajectory directory is preserved. The output CSV files contain:

```text
time_s, track_id, x_rw_smooth, y_rw_smooth
```

The default settings are:

- source frame rate: 30 fps
- output sampling interval: 0.5 seconds
- smoothing window: +/- 0.5 seconds around each sampled timestamp
- track IDs are preserved; missing detections are not interpolated

Optional arguments can be used to change the input folder, output folder, source FPS, sampling interval, or smoothing window:

```bash
python Pedestrian_tracking/3.Smooth_resample.py --source-fps 30 --sample-interval-s 0.5 --smooth-half-window-s 0.5
```

## Output Data

The main outputs are:

```text
Code/Pedestrian_data/1.Pedestrian_trajectory_raw/
Code/Pedestrian_data/2.Pedestrian_trajectory_processed/
```

The raw trajectory files store frame-level tracking results and georeferenced coordinates. The processed trajectory files store smoothed real-world coordinates at regular 0.5-second intervals and are intended for downstream analysis.

## Model and Calibration Files

The pedestrian detector weights are expected at:

```text
Code/Pedestrian_tracking/yolov12/weights/best.pt
```

The ReID model used by BoT-SORT is expected at:

```text
Code/Pedestrian_tracking/boxmot/models/osnet_x1_0_msmt17.pt
```

The pixel-to-real-world homography matrix is expected at:

```text
Code/Pedestrian_tracking/georeference/homography_matrix.csv
```

## Environment

The code was developed and tested in the following Python environment:

```text
python==3.11.15
numpy==1.26.4
pandas==2.3.3
opencv-python==4.10.0
torch==2.2.2+cu121
tqdm==4.68.1
ultralytics==8.3.63
boxmot==19.0.0
```

CUDA was available in the tested environment, and PyTorch was built with CUDA 12.1.

The repository includes local `yolov12` and `boxmot` folders. The scripts insert these local folders into `sys.path` so that the bundled versions are used during processing.

## Notes

- The tracking step can be computationally intensive and is intended to run on a CUDA-capable GPU.
- By default, annotated tracking videos are not saved. To save annotated videos, set `SAVE_ANNOTATED_VIDEO = True` in `Pedestrian_tracking/1.Track_pedestrians.py`.
- The coordinate conversion step modifies raw CSV files in place. Keep a backup of raw tracking results if the original bounding-box-only files are needed.
- The smoothing step writes new processed CSV files and does not overwrite the raw trajectory directory.
