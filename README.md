# DeepSORT People Detection

This project implements people detection and tracking using DeepSORT and YOLO models. It is designed to process video files, detect people, and track their movements across frames, outputting annotated videos and tracking results.

## Features
- **Object Detection**: Uses YOLOv8 and YOLOv10 models for robust people detection.
- **Tracking**: Employs DeepSORT for multi-object tracking, maintaining identities across frames.
- **Video Processing**: Supports input videos and outputs processed videos with bounding boxes and track IDs.
- **Evaluation**: Includes tools for evaluating tracking performance on MOTChallenge datasets.
- **Utilities**: Contains scripts for freezing models, generating detections, and visualizing results.

## Directory Structure
- `main.py`: Entry point for running detection and tracking on videos.
- `tracker.py`: Tracking logic and integration with detection models.
- `deep_sort/`: Main DeepSORT implementation and utilities.
  - `deep_sort_app.py`: Application logic for DeepSORT tracking.
  - `evaluate_motchallenge.py`: Evaluation script for MOTChallenge datasets.
  - `generate_videos.py`: Script to generate annotated videos.
  - `show_results.py`: Visualization of tracking results.
  - `application_util/`: Utility functions for preprocessing, visualization, and image handling.
  - `deep_sort/`: Core DeepSORT modules (detection, tracking, matching, Kalman filter, etc.).
  - `tools/`: Scripts for model freezing and detection generation.
- `data/`: Contains input videos (e.g., `people.mp4`).
- `model_data/`: Pretrained models (e.g., `mars-small128.pb`).
- `yolov8n.pt`, `yolov10n.pt`: YOLO model weights.
- `out.mp4`: Example output video.

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Other dependencies as required by DeepSORT and YOLO implementations

## Usage
1. Place your input video in the `data/` directory.
2. Run `main.py` to process the video and generate output:
  # Object Detection + DeepSORT Webapp

  Small project that runs YOLO object detection and DeepSORT tracking on video files and provides a minimal Flask webapp to upload videos, watch processing progress, and view processed results.

  Contents
  - `webapp.py` — Flask + Flask-SocketIO webapp (upload, processing, progress, demos).
  - `main.py` — simple example processor that runs detection + tracking over a video file.
  - `tracker.py` — wrapper around DeepSORT tracker and encoder.
  - `data/` — source videos used for demos and examples.
  - `processed/` — processed output videos (ignored by default, two demo outputs may be whitelisted).
  - `uploads/` — uploaded user videos (ignored by git).
  - `deep_sort/` — DeepSORT library and utilities.
  - `model_data/` — model data used by DeepSORT.

  Quick start (development)
  1. Create and activate a virtual environment:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

  2. Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

  3. Run the webapp:

  ```bash
  python webapp.py
  ```

  4. Open http://localhost:5000

  Webapp behavior
  - Upload a video from the main page. The original video appears on the left. The right side shows processing progress and serves the processed (tracked) video when ready.
  - The server processes uploads in a background thread and emits progress updates over Socket.IO.

  Demos
  - Visit `/demos` to see pre-generated demo pairs (original / processed). If the processed demo files don't exist they will be generated in the background and placed under `processed/`.
  - The repository ignores generated processed files by default but two demo processed files can be whitelisted and committed:
    - `processed/proc_people.mp4`
    - `processed/proc_people2.mp4`

  Auto-cleanup
  - The webapp includes a background cleanup task that deletes files inside `uploads/` and `processed/` older than 1 hour. Whitelisted demo files are preserved.
  - This runs inside the webapp process and requires no external cloud resources.

  Packaging & scripts
  - `pyproject.toml` and `Makefile` are provided for packaging and convenience.
  - Use `make install` to create venv and install dependencies; `make run` runs the app.

  Notes & troubleshooting
  - The app tries to use `yolov10n.pt` if present and falls back to `yolov8n.pt`.
  - Processing runs on CPU by default; GPU support requires installing the appropriate PyTorch build.
  - If browsers report "no video with supported format", ensure the processed files are valid MP4 files and that the server serves them with `video/mp4` Content-Type (the app uses Flask's `send_from_directory` with conditional responses to support range requests).

  Contributing
  - Please avoid committing large processed videos except the two demo files above. Add other demo files to the whitelist if you want to include them.

  License
  See `deep_sort/LICENSE` for license details.
