# Dockerfile
FROM python:3.10-slim

ENV POETRY_VERSION=1.4.2
ENV DEBIAN_FRONTEND=noninteractive

# System deps for OpenCV/ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libsm6 libxext6 libgl1 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copy app
COPY . /app

# Create folders used by app
RUN mkdir -p uploads processed

EXPOSE 5000

# Run with Gunicorn + eventlet (works with Flask-SocketIO)
CMD ["gunicorn", "-k", "eventlet", "-w", "1", "webapp:app", "--bind", "0.0.0.0:5000", "--log-level", "info"]