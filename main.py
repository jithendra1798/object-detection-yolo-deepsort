import os
import cv2
from ultralytics import YOLO
from tracker import Tracker
import random


video_inp_path = os.path.join('.', 'data', 'people2.mp4')
video_out_path = os.path.join('.', 'out2.mp4')

cap = cv2.VideoCapture(video_inp_path)

ret, frame = cap.read()

cap_out = cv2.VideoWriter(
    video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
    (frame.shape[1], frame.shape[0])
)
model = YOLO('yolov8n.pt')

tracker = Tracker()

colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)
]


while ret:
    results = model(frame, device='cpu')
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            class_id = int(class_id)
            detections.append([x1, y1, x2, y2, score])
        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            track_id = track.track_id
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[track_id % len(colors)], 3)


    # # Show the peopke detection 
    # cv2.imshow('frame', frame)
    # cv2.waitKey(5)

    # Save the video
    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()