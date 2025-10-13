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
   ```bash
   python main.py
   ```
3. Output video will be saved as `out.mp4`.

## Evaluation
- Use `deep_sort/evaluate_motchallenge.py` to evaluate tracking performance on MOTChallenge datasets.

## References
- [YOLO: You Only Look Once](https://pjreddie.com/darknet/yolo/)
- [DeepSORT: Simple Online and Realtime Tracking](https://github.com/nwojke/deep_sort)

## License
See `deep_sort/LICENSE` for license details.

Webapp
------

This repository includes a simple Flask webapp to upload a video, process it with the YOLO+Tracker pipeline, and display progress and the resulting tracked video.

Quick start

1. Create a virtual environment and install dependencies:

  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

2. Run the webapp:

  python webapp.py

3. Open http://localhost:5000 in your browser. Upload a video, the original will appear on the left and the right will show processing progress and the processed video when ready.

Notes
- The webapp uses the `yolov8n.pt` model in the repo root by default. Ensure it's present and that required ML dependencies (torch) are installed.
- Processing runs on CPU by default (model(..., device='cpu')).

Committing demo files
--------------------

This repository ignores generated uploads and processed outputs by default. To include the two demo processed videos in the repository (so demos show immediately), place the processed files at:

  processed/proc_people.mp4
  processed/proc_people2.mp4

These two files are explicitly whitelisted in `.gitignore` so they will be included when you commit. All other files under `processed/` will be ignored.
