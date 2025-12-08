import time
from pathlib import Path

import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO


#!/usr/bin/env python3
"""
YOLO Track 기반 Video Localizer
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class VideoLocalizer:
    """YOLO track 기반 객체 추적 및 시각화."""

    COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (255, 128, 0),
    ]

    def __init__(self, model_path="yolo11s.pt", conf=0.1):
        self.class_mapping = {
            32: "ball",
            41: "drone",
        }
        self.conf = conf
        self.model = YOLO(model_path)
        self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

        self.track_history = {}
        self.max_history = 50

    def process_frame(self, frame):
        """프레임 처리 후 (detections, counts) 반환."""
        results = self.model.track(
            frame,
            classes=list(self.class_mapping.keys()),
            conf=self.conf,
            persist=True,
            verbose=False,
        )[0]

        detections = []
        counts = {label: 0 for label in self.class_mapping.values()}

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                label = self.class_mapping.get(cls_id, "unknown")

                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    self.track_history[track_id].append((cx, cy))
                    if len(self.track_history[track_id]) > self.max_history:
                        self.track_history[track_id].pop(0)

                counts[label] += 1
                detections.append({
                    "label": label,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "track_id": track_id,
                    "score": score,
                })

        return detections, counts

    def draw_detections(self, frame, detections, counts):
        """프레임에 detection 결과 시각화."""
        # 궤적
        for track_id, history in self.track_history.items():
            if len(history) >= 2:
                color = self.COLORS[track_id % len(self.COLORS)]
                pts = np.array(history, dtype=np.int32)
                cv2.polylines(frame, [pts], False, color, 2)

        # bbox
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            track_id = det["track_id"]
            color = self.COLORS[track_id % len(self.COLORS)] if track_id else (128, 128, 128)

            cv2.rectangle(frame, p1, p2, color, 2)

            label_text = f"{det['label']}"
            if track_id is not None:
                label_text += f" #{track_id}"
            cv2.putText(frame, label_text, (p1[0], p1[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 카운트
        y = 25
        for label, count in counts.items():
            cv2.putText(frame, f"{label}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 25

        return frame

    def process_video(self, source, output_path=None, display=True):
        """비디오 처리."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        print(f"[INFO] Input video: {fps:.2f} FPS, {width}x{height}")

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("Press ESC to stop.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections, counts = self.process_frame(frame)
            annotated = self.draw_detections(frame.copy(), detections, counts)

            if writer:
                writer.write(annotated)

            if display:
                cv2.imshow("YOLO Tracker", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    
    
    
input_dir = "./input"
output_dir = "./output_tracking"

tracking1 = input_dir + "/tracking1.mp4"
tracking2 = input_dir + "/tracking2.mp4"
tracking_pp1 = input_dir + "/tracking_pp1.mp4"
tracking_pp2 = input_dir + "/tracking_pp2.mp4"

tracking1_o = output_dir + "/tracking1_o.mp4"
tracking2_o = output_dir + "/tracking2_o.mp4"
tracking_pp1_o = output_dir + "/tracking_pp1_o.mp4"
tracking_pp2_o = output_dir + "/tracking_pp2_o.mp4"

d_gc1 = input_dir + "/d_gc1.mp4"
d_gc2 = input_dir + "/d_gc2.mp4"


d_gc1_o = output_dir + "/d_gc1_o.mp4"
d_gc2_o = output_dir + "/d_gc2_o.mp4"

d_pp = input_dir + "/drone_pp.mp4"
d_sc = input_dir + "/drone_sc.mp4"
d_pp1 = input_dir + "/drone_pp1.mp4"
d_pp2 = input_dir + "/drone_pp2.mp4"
d_sc1 = input_dir + "/drone_sc1.mp4"
d_sc2 = input_dir + "/drone_sc2.mp4"

d_pp_o = output_dir + "/drone_pp_o.mp4"
d_sc_o = output_dir + "/drone_sc_o.mp4"
d_pp1_o = output_dir + "/drone_pp1_o.mp4"
d_pp2_o = output_dir + "/drone_pp2_o.mp4"
d_sc1_o = output_dir + "/drone_sc1_o.mp4"
d_sc2_o = output_dir + "/drone_sc2_o.mp4"


if __name__ == "__main__":
    localizer1 = VideoLocalizer(model_path="yolo11s.pt") # interval=frame
    # localizer1.process_video(d_sc1, d_sc1_o, display=False)
    # localizer1.process_video(d_sc2, d_sc2_o, display=False)
    localizer1.process_video(d_gc1, d_gc1_o, display=False)
    # localizer1.process_video(d_gc2, d_gc2_o, display=False)