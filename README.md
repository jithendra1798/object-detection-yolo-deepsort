# DeepSORT People Detection

This project implements people detection and tracking using DeepSORT and YOLO models. It is designed to process video files, detect people, and track their movements across frames, outputting annotated videos and tracking results.

## Live Demo [![Demos](https://img.shields.io/badge/Text-ColorCode)](https://objtrackapp60791.azurewebsites.net/demos)
<a href="https://objtrackapp60791.azurewebsites.net/demos" style="background:#0366d6;color:white;padding:6px 12px;text-decoration:none;border-radius:4px;">
  Open Demos
</a>

- Demos page: [https://objtrackapp60791.azurewebsites.net/demos](https://objtrackapp60791.azurewebsites.net/demos)
- App: [https://objtrackapp60791.azurewebsites.net/](https://objtrackapp60791.azurewebsites.net/)

> **Heads up:** The hosted instance runs on Azure’s free tier compute. Expect slower processing compared with running the project locally on a full machine (limited CPU, no GPU acceleration).

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

  Local Docker testing & CI
  - Use the included Makefile targets to build and run the container locally:

  ```bash
  make docker-build
  make docker-run
  # or use the helper script
  ./scripts/test_docker.sh
  ```

  - CI workflow: `.github/workflows/ci-deploy.yml` will build/push the image to ACR and deploy to App Service when you push to `main`.

  Required GitHub secrets for CI (set these in your repository settings):
  - `ACR_LOGIN_SERVER` (e.g. myacrname.azurecr.io)
  - `ACR_USERNAME`
  - `ACR_PASSWORD`
  - `AZURE_WEBAPP_NAME`

  Notes & troubleshooting
  - The app tries to use `yolov10n.pt` if present and falls back to `yolov8n.pt`.
  - Processing runs on CPU by default; GPU support requires installing the appropriate PyTorch build.
  - If browsers report "no video with supported format", ensure the processed files are valid MP4 files and that the server serves them with `video/mp4` Content-Type (the app uses Flask's `send_from_directory` with conditional responses to support range requests).

  Contributing
  - Please avoid committing large processed videos except the two demo files above. Add other demo files to the whitelist if you want to include them.

  License
  See `deep_sort/LICENSE` for license details.

  ## Azure deployment (Docker + App Service)

  You can deploy this webapp to Microsoft Azure using a container image. The recommended flow is:

  1. Build a production Docker image locally (or via CI).
  2. Push the image to Azure Container Registry (ACR).
  3. Create an Azure App Service (Web App for Containers) and point it to the ACR image.

  Below are concise, copy-pasteable steps.

  1) Prepare a production Dockerfile

  - Use a slim Python base, install system deps required by OpenCV/ffmpeg and your model runtime, install `requirements.txt`, and run Gunicorn + eventlet for Flask-SocketIO. Your local `Dockerfile` may already be present; if not, use the example below as a starting point.

  ```dockerfile
  FROM python:3.10-slim
  ENV DEBIAN_FRONTEND=noninteractive
  RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ffmpeg libsm6 libxext6 libgl1 git ca-certificates \
      && rm -rf /var/lib/apt/lists/*
  WORKDIR /app
  COPY requirements.txt ./
  RUN pip install --upgrade pip && pip install -r requirements.txt
  COPY . /app
  RUN mkdir -p uploads processed
  EXPOSE 5000
  # Use gunicorn with eventlet for Socket.IO
  CMD ["gunicorn", "-k", "eventlet", "-w", "1", "webapp:app", "--bind", "0.0.0.0:5000"]
  ```

  2) Build and test locally

  ```bash
  docker build -t object-tracker:latest .
  docker run --rm -p 5000:5000 -v $(pwd)/uploads:/app/uploads -v $(pwd)/processed:/app/processed object-tracker:latest
  # then open http://localhost:5000
  ```

  3) Push image to Azure Container Registry (ACR)

  ```bash
  # login to Azure
  az login
  az group create -n myResourceGroup -l eastus
  az acr create -n myacrname -g myResourceGroup --sku Basic --admin-enabled true
  az acr login -n myacrname
  docker tag object-tracker:latest myacrname.azurecr.io/object-tracker:latest
  docker push myacrname.azurecr.io/object-tracker:latest
  ```

  4) Create App Service (Linux) and point to the ACR image

  ```bash
  az appservice plan create -g myResourceGroup -n myPlan --is-linux --sku B1
  az webapp create -g myResourceGroup -p myPlan -n my-webapp-name \
    --deployment-container-image-name myacrname.azurecr.io/object-tracker:latest

  # Configure container registry credentials (if needed)
  ACR_LOGIN_SERVER=$(az acr show -n myacrname -g myResourceGroup --query loginServer -o tsv)
  ACR_USERNAME=$(az acr credential show -n myacrname -g myResourceGroup --query username -o tsv)
  ACR_PASSWORD=$(az acr credential show -n myacrname -g myResourceGroup --query passwords[0].value -o tsv)
  az webapp config container set -g myResourceGroup -n my-webapp-name \
    --docker-custom-image-name ${ACR_LOGIN_SERVER}/object-tracker:latest \
    --docker-registry-server-url https://${ACR_LOGIN_SERVER} \
    --docker-registry-server-user $ACR_USERNAME \
    --docker-registry-server-password $ACR_PASSWORD
  ```

  5) Enable WebSockets (required for Socket.IO)

  ```bash
  az webapp update -g myResourceGroup -n my-webapp-name --set clientAffinityEnabled=false
  az webapp config set -g myResourceGroup -n my-webapp-name --generic-configurations '{"webSockets":{"enabled":true}}'
  ```

  6) Persistent storage (optional)

  - App Service local filesystem is ephemeral. For persistent uploads/processed data either mount an Azure File share or change the app to use Azure Blob Storage for uploads/outputs.
  - To mount Azure Files into App Service: create a Storage Account + File Share, then in the App Service -> "Configuration" -> "Path mappings" add a storage mount and map it to `/app/uploads` and `/app/processed`.

  7) GPU inference (if needed)

  - App Service does not provide GPU instances. For GPU-based inference deploy to an Azure VM with GPU (NC-series) or use AKS with GPU node pools.

  8) CI/CD (recommended)

  - Use GitHub Actions to build & push the image to ACR and then deploy (or call `az webapp` to update). Typical steps:
    - actions/checkout
    - docker/build-push-action
    - azure/webapps-deploy or az cli

  Notes
  - Ensure `requirements.txt` pins compatible `torch`/`ultralytics` versions for the target environment.
  - Use `gunicorn` + `eventlet` worker for Socket.IO in production.
  - For multi-instance scaling, move uploads/processed to shared storage (Azure Files/Blob) or use a database/message queue to coordinate processing.

