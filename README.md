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
- [Yolov8 object detection + deep sort object tracking](https://www.youtube.com/watch?v=jIRRuGN0j5E&list=PLb49csYFtO2HGELdc-RLRCNVNy0g2UMwc&index=1)

## License
See `deep_sort/LICENSE` for license details.
