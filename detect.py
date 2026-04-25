import os
import threading
import time

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
ESP32_IP = "http://192.168.29.212"
STREAM_URL = f"{ESP32_IP}:81/stream"

CONFIDENCE = 0.22           # lower threshold for occluded/partial humans
USE_GPU = True
INFERENCE_WIDTH = 640
DETECTION_INTERVAL = 0.45
CAMERA_FLIP_HORIZONTAL = True
CAMERA_FLIP_VERTICAL = True

TARGET_MEMORY = 0.8
CENTER_DEADZONE = 0.18
TRACKING_ALPHA = 0.35
CAMERA_STOP_WIDTH_RATIO = 0.38
HUD_UPDATE_INTERVAL = 0.10
WINDOW_NAME = "AI Robot Control"

DARK_THRESHOLD = 60         # mean pixel brightness below this = dark frame
DARK_CONFIDENCE = 0.18      # even lower conf for dark scenes

DEVICE = "cuda" if USE_GPU else "cpu"

# Switch to fine-tuned model after running prepare_dataset.py
# Falls back to base model if fine-tuned weights don't exist yet
_FINETUNED = "runs/detect/runs/rescue/rescue_finetune3/weights/best.pt"
MODEL_PATH = _FINETUNED if os.path.exists(_FINETUNED) else "yolov8n.pt"

print("[INFO] Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
model.to(DEVICE)
print(f"[INFO] Running on {DEVICE.upper()}")


# --------------------------------------------------
# FRAME GRABBER
# --------------------------------------------------
class FrameGrabber:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.session = requests.Session()
        self.response = None
        self.lock = threading.Lock()
        self.frame = None
        self.last_frame_time = 0.0
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return True

    def _reconnect(self):
        if self.response is not None:
            self.response.close()
        self.response = None
        time.sleep(0.5)

        while self.running and self.response is None:
            print("[WARN] Reconnecting stream...")
            try:
                self.response = self.session.get(
                    self.stream_url,
                    stream=True,
                    timeout=(2.0, 2.0),
                )
                self.response.raise_for_status()
            except requests.RequestException:
                self.response = None
                time.sleep(1.0)

    def _reader(self):
        buffer = b""

        while self.running:
            if self.response is None:
                self._reconnect()
                continue

            try:
                for chunk in self.response.iter_content(chunk_size=4096):
                    if not self.running:
                        return

                    if not chunk:
                        continue

                    buffer += chunk
                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9")

                    while start != -1 and end != -1 and end > start:
                        jpg = buffer[start : end + 2]
                        buffer = buffer[end + 2 :]

                        decoded = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        if decoded is not None:
                            with self.lock:
                                self.frame = decoded
                                self.last_frame_time = time.time()

                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9")

                raise requests.RequestException("stream ended")
            except (requests.RequestException, cv2.error, ValueError) as exc:
                print(f"[WARN] Stream error, reconnecting: {exc}")
                self._reconnect()
                continue

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)
        if self.response is not None:
            self.response.close()


# --------------------------------------------------
# FRAME UTILITIES
# --------------------------------------------------
def orient_frame(frame):
    if CAMERA_FLIP_HORIZONTAL and CAMERA_FLIP_VERTICAL:
        return cv2.flip(frame, -1)
    if CAMERA_FLIP_HORIZONTAL:
        return cv2.flip(frame, 1)
    if CAMERA_FLIP_VERTICAL:
        return cv2.flip(frame, 0)
    return frame


def prepare_inference_frame(frame):
    frame_h, frame_w = frame.shape[:2]
    if frame_w <= INFERENCE_WIDTH:
        return frame, 1.0, 1.0

    scale = INFERENCE_WIDTH / frame_w
    resized_h = int(frame_h * scale)
    resized = cv2.resize(
        frame,
        (INFERENCE_WIDTH, resized_h),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized, frame_w / INFERENCE_WIDTH, frame_h / resized_h


def is_dark_frame(frame, threshold=DARK_THRESHOLD):
    """Returns True if the frame is dark/shadowy."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < threshold


def enhance_frame(frame):
    """
    CLAHE contrast enhancement on the L channel (LAB space).
    Brings out detail in shadowy areas like under beds or in rubble.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# --------------------------------------------------
# DETECTION
# --------------------------------------------------
def analyze_detections(frame, results, scale_x=1.0, scale_y=1.0):
    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id != 0 or conf < CONFIDENCE:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w  = max(0, x2 - x1)
            h  = max(0, y2 - y1)
            area = w * h

            # Prone human fix:
            # Normal YOLO expects tall boxes (h > w) for standing people.
            # Under beds/rubble, humans are horizontal — don't penalize wide boxes.
            # aspect < 0.25 = extremely flat box → likely floor/furniture, skip it.
            # aspect 0.25–0.75 = prone human, keep it and flag it.
            aspect = h / (w + 1e-6)
            if aspect < 0.25:
                continue

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "area": area,
                "conf": conf,
                "width": w,
                "height": h,
                "prone": aspect < 0.75,
            })

    return detections


def choose_target(detections, previous_cx):
    if not detections:
        return None

    if previous_cx is None:
        return max(detections, key=lambda det: det["area"])

    return min(
        detections,
        key=lambda det: abs(det["center"][0] - previous_cx) - (det["area"] * 0.0005),
    )


# --------------------------------------------------
# DRAWING
# --------------------------------------------------
def draw_scene(frame, detections, target, frame_w):
    frame_h = frame.shape[0]
    deadzone_half = int(frame_w * CENTER_DEADZONE / 2)
    center_x = frame_w // 2

    cv2.line(frame, (center_x, 0), (center_x, frame_h), (255, 255, 0), 1)
    cv2.line(
        frame,
        (center_x - deadzone_half, 0),
        (center_x - deadzone_half, frame_h),
        (100, 100, 255),
        1,
    )
    cv2.line(
        frame,
        (center_x + deadzone_half, 0),
        (center_x + deadzone_half, frame_h),
        (100, 100, 255),
        1,
    )

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        is_target = det is target
        color = (0, 200, 255) if is_target else (0, 255, 0)

        label = "Target" if is_target else "Person"
        if det.get("prone"):
            label += " [PRONE]"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {det['conf']:.0%}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def draw_status_chip(frame, text, color, x, y):
    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        1,
    )
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + text_w + 18, y + text_h + baseline + 10), color, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.putText(
        frame,
        text,
        (x + 9, y + text_h + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (18, 18, 18),
        1,
    )


def draw_hud(frame, fps, action_text, detections, tracker, offset, dark_mode):
    if "LEFT" in action_text:
        chip_color = (80, 190, 255)
    elif "RIGHT" in action_text:
        chip_color = (111, 233, 168)
    elif "CLOSE" in action_text:
        chip_color = (255, 191, 87)
    else:
        chip_color = (198, 208, 218)
    draw_status_chip(frame, action_text, chip_color, 14, 14)

    # Dark mode indicator
    if dark_mode:
        draw_status_chip(frame, "ENHANCE ON", (60, 60, 200), 14, 52)

    stats_text = f"FPS {fps:4.1f}   HUMANS {len(detections)}"
    overlay = frame.copy()
    (stats_w, stats_h), stats_base = cv2.getTextSize(
        stats_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        1,
    )
    stats_x = 14
    stats_y = frame.shape[0] - 20
    cv2.rectangle(
        overlay,
        (stats_x - 8, stats_y - stats_h - 8),
        (stats_x + stats_w + 8, stats_y + stats_base + 6),
        (20, 26, 32),
        -1,
    )
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.putText(
        frame,
        stats_text,
        (stats_x, stats_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (236, 240, 243),
        1,
    )

    if tracker["smoothed_cx"] is not None and offset is not None:
        aim_text = f"TRACK {int(tracker['smoothed_cx'])}   WIDTH {int(tracker['smoothed_width'])}   OFFSET {offset:+.2f}"
        overlay = frame.copy()
        (aim_w, aim_h), aim_base = cv2.getTextSize(
            aim_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            1,
        )
        aim_x = max(14, frame.shape[1] - aim_w - 24)
        aim_y = 26
        cv2.rectangle(
            overlay,
            (aim_x - 8, aim_y - aim_h - 8),
            (aim_x + aim_w + 8, aim_y + aim_base + 6),
            (20, 26, 32),
            -1,
        )
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(
            frame,
            aim_text,
            (aim_x, aim_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (236, 240, 243),
            1,
        )


# --------------------------------------------------
# TRACKER
# --------------------------------------------------
def update_tracker(target, tracker, now):
    if target is None:
        return

    target_cx     = float(target["center"][0])
    target_area   = float(target["area"])
    target_width  = float(target["width"])
    target_height = float(target["height"])

    if tracker["smoothed_cx"] is None:
        tracker["smoothed_cx"]     = target_cx
        tracker["smoothed_area"]   = target_area
        tracker["smoothed_width"]  = target_width
        tracker["smoothed_height"] = target_height
    else:
        tracker["smoothed_cx"] = (
            (1.0 - TRACKING_ALPHA) * tracker["smoothed_cx"]
            + TRACKING_ALPHA * target_cx
        )
        tracker["smoothed_area"] = (
            (1.0 - TRACKING_ALPHA) * tracker["smoothed_area"]
            + TRACKING_ALPHA * target_area
        )
        tracker["smoothed_width"] = (
            (1.0 - TRACKING_ALPHA) * tracker["smoothed_width"]
            + TRACKING_ALPHA * target_width
        )
        tracker["smoothed_height"] = (
            (1.0 - TRACKING_ALPHA) * tracker["smoothed_height"]
            + TRACKING_ALPHA * target_height
        )

    tracker["last_seen"] = now


def decide_action(tracker, frame_w, now):
    if now - tracker["last_seen"] <= TARGET_MEMORY and tracker["smoothed_cx"] is not None:
        width_ratio = tracker["smoothed_width"] / float(frame_w)
        if width_ratio >= CAMERA_STOP_WIDTH_RATIO:
            return "HUMAN CLOSE", 0.0

        offset = (tracker["smoothed_cx"] - (frame_w / 2)) / frame_w
        if offset < -(CENTER_DEADZONE / 2):
            return "HUMAN LEFT", offset
        if offset > (CENTER_DEADZONE / 2):
            return "HUMAN RIGHT", offset
        return "HUMAN CENTER", offset

    return "IDLE - NO HUMAN", None


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    print(f"[INFO] Connecting to stream: {STREAM_URL}")
    grabber = FrameGrabber(STREAM_URL)
    if not grabber.start():
        print("[ERROR] Cannot open stream.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_count      = 0
    fps_time         = time.time()
    last_detect_time = 0.0
    detections       = []
    target           = None
    scale_x          = 1.0
    scale_y          = 1.0
    action_text      = "WAIT"
    display_frame    = None
    last_hud_time    = 0.0
    dark_mode        = False

    tracker = {
        "smoothed_cx":     None,
        "smoothed_area":   0.0,
        "smoothed_width":  0.0,
        "smoothed_height": 0.0,
        "last_seen":       0.0,
    }

    try:
        while True:
            frame = grabber.read()
            if frame is None:
                time.sleep(0.01)
                continue
            frame = orient_frame(frame)

            frame_count += 1
            now = time.time()
            frame_h, frame_w = frame.shape[:2]

            if now - last_detect_time >= DETECTION_INTERVAL:
                infer_frame, scale_x, scale_y = prepare_inference_frame(frame)

                # Dark/shadow detection and enhancement
                dark_mode = is_dark_frame(infer_frame)
                if dark_mode:
                    infer_frame = enhance_frame(infer_frame)

                dynamic_conf = DARK_CONFIDENCE if dark_mode else CONFIDENCE

                h_inf, w_inf = infer_frame.shape[:2]
                results = model.predict(
                    infer_frame,
                    classes=[0],
                    conf=dynamic_conf,
                    imgsz=(h_inf, w_inf),
                    augment=True,       # test-time augmentation: catches prone/occluded humans
                    verbose=False,
                    device=DEVICE,
                )
                detections = analyze_detections(frame, results, scale_x, scale_y)
                target = choose_target(detections, tracker["smoothed_cx"])
                update_tracker(target, tracker, now)
                last_detect_time = now

            action_text, offset = decide_action(tracker, frame_w, now)

            elapsed = now - fps_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            if elapsed >= 1.0:
                frame_count = 0
                fps_time = now

            if display_frame is None or now - last_hud_time >= HUD_UPDATE_INTERVAL:
                display_frame = frame.copy()
                draw_scene(display_frame, detections, target, frame_w)
                draw_hud(
                    display_frame,
                    fps,
                    action_text,
                    detections,
                    tracker,
                    offset,
                    dark_mode,
                )
                last_hud_time = now

            cv2.imshow(WINDOW_NAME, display_frame if display_frame is not None else frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        grabber.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()