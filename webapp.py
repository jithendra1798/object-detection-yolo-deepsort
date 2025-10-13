from flask import Flask, render_template, request, send_from_directory, jsonify, Response, abort
from werkzeug.utils import secure_filename
import mimetypes
from flask_socketio import SocketIO, emit
import os
import uuid
import threading
import time
import cv2
import random
from ultralytics import YOLO
from tracker import Tracker

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUT_DIR = os.path.join(BASE_DIR, 'processed')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev'
socketio = SocketIO(app, cors_allowed_origins='*')

# Demo originals (relative to BASE_DIR/data)
DEMO_ORIGINALS = [
    os.path.join(BASE_DIR, 'data', 'people1.mp4'),
    os.path.join(BASE_DIR, 'data', 'people2.mp4'),
]


def ensure_demo_processed(original_path):
    """Ensure the processed version of original_path exists under OUT_DIR.

    If missing, schedule background processing to create it.
    Returns the processed filename (basename) expected under /processed/.
    """
    base = os.path.splitext(os.path.basename(original_path))[0]
    proc_name = f'proc_{base}.mp4'
    proc_path = os.path.join(OUT_DIR, proc_name)
    if os.path.exists(proc_path) and os.path.getsize(proc_path) > 1024:
        app.logger.info('Processed demo exists: %s', proc_path)
        return proc_name

    # schedule background processing
    app.logger.info('Scheduling generation of demo processed video: %s -> %s', original_path, proc_path)
    # start a background task; sid=None so no socket events will be sent
    socketio.start_background_task(process_video, original_path, proc_path, None)
    return proc_name


def process_video(input_path, output_path, sid=None):
    """Process video: run YOLO detections and simple Tracker drawing, emit progress to client SID."""
    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    if not ret:
        if sid:
            socketio.emit('processing_error', {'error': 'cannot read video'}, to=sid)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # Try a list of codecs to maximize chance of producing a playable mp4
    preferred_codecs = ['avc1', 'mp4v', 'H264', 'X264']
    out = None
    for codec in preferred_codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            tmp_out = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
            if tmp_out.isOpened():
                out = tmp_out
                app.logger.info('Opened VideoWriter with codec %s for %s', codec, output_path)
                break
            else:
                tmp_out.release()
        except Exception:
            continue

    # Fallback: try MP4V by name
    if out is None:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

    if not out or not out.isOpened():
        app.logger.error('Failed to open VideoWriter for %s', output_path)
        if sid:
            socketio.emit('processing_error', {'error': 'failed to open video writer'}, to=sid)
        cap.release()
        return

    # Use YOLOv10 model for class labels if available
    try:
        model = YOLO('yolov10n.pt')
    except Exception:
        # fallback to yolov8 if v10 not available
        model = YOLO('yolov8n.pt')
    tracker = Tracker()
    colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(20)]

    # count frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    processed = 0

    while ret:
        results = model(frame, device='cpu')
        detections = []
        det_classes = []
        # collect detections and classes for this frame
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                detections.append([x1, y1, x2, y2, score])
                det_classes.append(int(class_id))

        tracker.update(frame, detections)

        # Associate a class label to each track using IoU matching with detections
        track_labels = {}
        def iou(boxA, boxB):
            # boxes are [x1,y1,x2,y2]
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
            boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
            union = boxAArea + boxBArea - interArea
            return interArea / union if union > 0 else 0.0

        for track in tracker.tracks:
            bbox = track.bbox
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, bbox)
            best_iou = 0.0
            best_idx = None
            for i, det in enumerate(detections):
                det_box = det[:4]
                val = iou([x1,y1,x2,y2], det_box)
                if val > best_iou:
                    best_iou = val
                    best_idx = i
            label = None
            if best_idx is not None and best_iou > 0.2 and best_idx < len(det_classes):
                cls_id = det_classes[best_idx]
                # resolve class name if available
                names = None
                try:
                    names = model.model.names
                except Exception:
                    try:
                        names = model.names
                    except Exception:
                        names = None
                if names and cls_id in names:
                    label = names[cls_id]
                else:
                    label = str(cls_id)

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[track_id % len(colors)], 3)
            if label:
                # draw filled rectangle for label background
                ((txt_w, txt_h), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - int(txt_h*1.6)), (x1 + txt_w + 6, y1), colors[track_id % len(colors)], -1)
                cv2.putText(frame, label, (x1+3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        out.write(frame)
        processed += 1
        # emit progress
        if sid and total_frames>0:
            percent = int(processed * 100 / total_frames)
            socketio.emit('processing_progress', {'percent': percent}, to=sid)

        # read next
        ret, frame = cap.read()

    cap.release()
    out.release()

    # Verify output file exists and has non-trivial size
    try:
        size = os.path.getsize(output_path)
    except Exception:
        size = 0

    if size < 1024:  # less than 1 KB - likely invalid
        app.logger.error('Processed file is too small (%d bytes): %s', size, output_path)
        if sid:
            socketio.emit('processing_error', {'error': 'processed file invalid or empty', 'size': size}, to=sid)
        # remove invalid file if present
        try:
            os.remove(output_path)
        except Exception:
            pass
        return

    if sid:
        socketio.emit('processing_done', {'output': os.path.basename(output_path)}, to=sid)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/demos')
def demos():
    # Prepare demo pairs: for each original, ensure a processed version exists (or is being generated)
    pairs = []
    for i, orig in enumerate(DEMO_ORIGINALS):
        proc_basename = ensure_demo_processed(orig)
        proc_url = f'/processed/{proc_basename}'
        # fallback to out/out2 if processed not yet available
        if not os.path.exists(os.path.join(OUT_DIR, proc_basename)):
            fallback = f'/out{"" if i==0 else "2"}.mp4'
            proc_url = fallback
        orig_url = f'/data/{os.path.basename(orig)}'
        pairs.append({'original': orig_url, 'processed': proc_url})
    return render_template('demos.html', pairs=pairs)


@app.route('/out.mp4')
def out_mp4():
    return send_from_directory(BASE_DIR, 'out.mp4')


@app.route('/out2.mp4')
def out2_mp4():
    return send_from_directory(BASE_DIR, 'out2.mp4')


@app.route('/data/<path:filename>')
def data_file(filename):
    # Serve files from the data directory used for demos
    return send_from_directory(os.path.join(BASE_DIR, 'data'), filename)


@app.route('/upload', methods=['POST'])
def upload():
    # Upload handled via JS; returns file ids and URLs
    if 'video' not in request.files:
        return jsonify({'error': 'no file'}), 400
    f = request.files['video']
    if f.filename == '':
        return jsonify({'error': 'empty filename'}), 400
    # sanitize filename and prefix with UUID
    orig_name = secure_filename(f.filename)
    uid = str(uuid.uuid4())
    filename = uid + '_' + orig_name
    save_path = os.path.join(UPLOAD_DIR, filename)
    f.save(save_path)
    return jsonify({'filename': filename, 'url': f'/uploads/{filename}'}), 201


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route('/processed/<path:filename>')
def processed_file(filename):
    # Prefer Flask's send_from_directory with conditional=True which handles
    # Range requests and sets appropriate Content-Type/Length headers.
    full_path = os.path.join(OUT_DIR, filename)
    if not os.path.exists(full_path):
        abort(404)
    size = os.path.getsize(full_path)
    app.logger.info('Serving processed file %s (%d bytes)', full_path, size)
    # 'conditional=True' allows Flask to handle Range requests and return 206
    return send_from_directory(OUT_DIR, filename, conditional=True)


@socketio.on('start_processing')
def handle_start(data):
    # data: { filename }
    filename = data.get('filename')
    if not filename:
        emit('processing_error', {'error': 'no filename provided'})
        return

    input_path = os.path.join(UPLOAD_DIR, filename)
    app.logger.info("start_processing requested for %s", input_path)

    # If file missing, try to find a matching file in uploads (fallback)
    if not os.path.exists(input_path):
        matches = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(filename)]
        if matches:
            input_path = os.path.join(UPLOAD_DIR, matches[0])
            app.logger.info("Found upload by suffix: %s", input_path)
        else:
            app.logger.error("File not found: %s; uploads: %s", input_path, os.listdir(UPLOAD_DIR))
            emit('processing_error', {'error': 'file not found', 'checked': input_path})
            return

    # Always produce an .mp4 output filename (browsers expect mp4 container)
    base_no_ext = os.path.splitext(filename)[0]
    out_name = f'proc_{base_no_ext}.mp4'
    output_path = os.path.join(OUT_DIR, out_name)

    # run processing in background task (use socketio helper)
    socketio.start_background_task(process_video, input_path, output_path, request.sid)
    emit('processing_started', {'processing_url': f'/processed/{out_name}'})


def main(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Run the Flask-SocketIO application.

    This function is exposed as an entrypoint for packaging.
    """
    socketio.run(app, host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()
