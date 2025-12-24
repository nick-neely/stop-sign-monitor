import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO
try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from .config import (
    BASE_PIXELS_PER_METER,
    HORIZONTAL_WEIGHT,
    LICENSE_PLATE_MODEL_PATH,
    MIN_HISTORY_FRAMES,
    MODELS_DIR,
    OUTPUT_DIR,
    PERSPECTIVE_FACTOR,
    REFERENCE_Y_RATIO,
    SPEED_INTERVALS,
    SPEED_SMOOTHING_ALPHA,
    STOP_SPEED_THRESHOLD,
    VERTICAL_WEIGHT,
    VIDEO_PATH,
)
from .report import print_final_report
from .speed import SpeedEstimator

MIN_SPEED_SENTINEL = 999.0
VIOLATION_SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "violation_snapshots")
PLATE_SNAPSHOT_SUFFIX = "_plate.jpg"
PROGRESS_BAR_WIDTH = 32
PROGRESS_UPDATE_EVERY = 5


def setup_stop_zone(cap, window_name):
    """Capture a polygonal stop zone from user clicks on the first frame."""
    stop_zone_polygon = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            stop_zone_polygon.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("--- SETUP ---")
    print("Draw your zone (4 clicks). Press SPACE to start.")

    ret, frame = cap.read()
    if not ret:
        return stop_zone_polygon

    while True:
        temp_frame = frame.copy()
        if stop_zone_polygon:
            cv2.polylines(
                temp_frame, [np.array(stop_zone_polygon)], True, (0, 255, 255), 2
            )
        cv2.imshow(window_name, temp_frame)
        if cv2.waitKey(1) == 32 and len(stop_zone_polygon) >= 3:
            break

    return stop_zone_polygon


def update_car_stats(car_stats, track_id, speed, in_zone):
    """Update per-car stats and return the color for rendering."""
    if track_id not in car_stats:
        car_stats[track_id] = {"min_speed": MIN_SPEED_SENTINEL, "status": "Approaching"}

    if in_zone:
        current_min = car_stats[track_id]["min_speed"]
        if speed < current_min:
            car_stats[track_id]["min_speed"] = speed

        if car_stats[track_id]["min_speed"] < STOP_SPEED_THRESHOLD:
            car_stats[track_id]["status"] = "Clean Stop"
            return (255, 0, 0)

        car_stats[track_id]["status"] = "Rolling..."
        return (0, 0, 255)

    if car_stats[track_id]["status"] == "Rolling...":
        car_stats[track_id]["status"] = "ROLLING STOP VIOLATION"
        car_stats[track_id]["snapshot_needed"] = True

    return (0, 255, 0)


def resolve_model_path(model_path):
    """Resolve model path, optionally downloading Hugging Face weights."""
    if not model_path:
        return None

    if model_path.startswith("hf:"):
        if not HF_AVAILABLE:
            print("huggingface_hub not installed. Install it with: pip install huggingface_hub")
            return None
        hf_spec = model_path[3:]
        parts = hf_spec.split("/")
        if len(parts) < 3:
            print(f"Invalid Hugging Face model spec: {model_path}")
            return None
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = "/".join(parts[2:])
        local_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(local_path):
            return local_path
        try:
            print(f"Downloading model from Hugging Face: {repo_id}/{filename}")
            return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=MODELS_DIR)
        except Exception as e:
            print(f"Failed to download model from Hugging Face: {e}")
            return None

    return model_path


def load_license_plate_model(model_path):
    """Return a YOLO model if the plate weights can be loaded, otherwise None."""
    resolved_path = resolve_model_path(model_path)
    if not resolved_path:
        return None
    try:
        return YOLO(resolved_path)
    except Exception as e:
        print(f"Failed to load license plate model from {resolved_path}: {e}")
        print("Skipping plate capture.")
        return None


def detect_plate_crop(plate_model, vehicle_crop):
    """Return a cropped plate image from a vehicle crop, or None if not found."""
    if plate_model is None or vehicle_crop.size == 0:
        return None
    results = plate_model.predict(vehicle_crop, verbose=False)
    if not results or results[0].boxes is None or results[0].boxes.xyxy is None:
        return None
    boxes = results[0].boxes.xyxy.cpu().tolist()
    if not boxes:
        return None
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(vehicle_crop.shape[1], int(x2))
    y2 = min(vehicle_crop.shape[0], int(y2))
    plate_crop = vehicle_crop[y1:y2, x1:x2]
    return plate_crop if plate_crop.size else None


def format_time(seconds):
    """Format seconds into H:MM:SS."""
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_progress(current_frame, total_frames, fps, start_time):
    """Render a progress bar with elapsed/remaining time."""
    if total_frames <= 0 or fps <= 0:
        return
    progress = min(1.0, current_frame / total_frames)
    filled = int(PROGRESS_BAR_WIDTH * progress)
    bar = "█" * filled + "░" * (PROGRESS_BAR_WIDTH - filled)
    elapsed = time.time() - start_time
    total_seconds = total_frames / fps
    remaining = max(0.0, total_seconds - (current_frame / fps))
    line = (
        f"\r[{bar}] {progress * 100:5.1f}% "
        f"{format_time(elapsed)} / {format_time(total_seconds)} "
        f"(ETA {format_time(remaining)})"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def main():
    """Main entry point for the traffic simulation."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video FPS: {fps:.2f}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_time = time.time()

    window_name = "Traffic Sim"
    stop_zone_polygon = setup_stop_zone(cap, window_name)

    model = YOLO(os.path.join(MODELS_DIR, "yolov8n.pt"))
    plate_model = load_license_plate_model(LICENSE_PLATE_MODEL_PATH)
    car_stats = {}
    frame_height = None
    speed_estimator = None
    os.makedirs(VIOLATION_SNAPSHOT_DIR, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            if total_frames > 0:
                render_progress(total_frames, total_frames, fps, start_time)
                sys.stdout.write("\n")
                sys.stdout.flush()
            print("\nVideo ended. Generating report...")
            break

        if frame_height is None:
            frame_height = frame.shape[0]
            speed_estimator = SpeedEstimator(
                fps=fps,
                frame_height=frame_height,
                base_pixels_per_meter=BASE_PIXELS_PER_METER,
                perspective_factor=PERSPECTIVE_FACTOR,
                reference_y_ratio=REFERENCE_Y_RATIO,
                intervals=SPEED_INTERVALS,
                smoothing_alpha=SPEED_SMOOTHING_ALPHA,
                min_history_frames=MIN_HISTORY_FRAMES,
                horizontal_weight=HORIZONTAL_WEIGHT,
                vertical_weight=VERTICAL_WEIGHT,
            )
            reference_y = frame_height * REFERENCE_Y_RATIO
            print(f"Frame height: {frame_height}, Reference Y: {reference_y:.1f}")

        results = model.track(frame, persist=True, classes=[2, 7], verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                bottom_center = (float(x), float(y + h / 2))
                speed = speed_estimator.update(track_id, bottom_center)

                center = (float(x), float(y))
                in_zone = (
                    cv2.pointPolygonTest(np.array(stop_zone_polygon), center, False) >= 0
                    if stop_zone_polygon
                    else False
                )
                color = update_car_stats(car_stats, track_id, speed, in_zone)

                p1 = (int(x - w / 2), int(y - h / 2))
                p2 = (int(x + w / 2), int(y + h / 2))
                cv2.rectangle(frame, p1, p2, color, 2)
                cv2.putText(frame, f"{int(speed)}mph", (p1[0], p1[1] - 10), 0, 0.6, color, 2)

                if car_stats.get(track_id, {}).get("snapshot_needed"):
                    x1, y1 = max(0, p1[0]), max(0, p1[1])
                    x2, y2 = min(frame.shape[1], p2[0]), min(frame.shape[0], p2[1])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        filename = f"car_{track_id}_violation.jpg"
                        cv2.imwrite(f"{VIOLATION_SNAPSHOT_DIR}/{filename}", crop)
                        plate_crop = detect_plate_crop(plate_model, crop)
                        if plate_crop is not None:
                            plate_name = f"car_{track_id}{PLATE_SNAPSHOT_SUFFIX}"
                            cv2.imwrite(f"{VIOLATION_SNAPSHOT_DIR}/{plate_name}", plate_crop)
                    car_stats[track_id]["snapshot_needed"] = False

        if stop_zone_polygon:
            cv2.polylines(frame, [np.array(stop_zone_polygon)], True, (0, 255, 255), 2)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == ord("q"):
            break
        if total_frames > 0 and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % PROGRESS_UPDATE_EVERY == 0:
            render_progress(int(cap.get(cv2.CAP_PROP_POS_FRAMES)), total_frames, fps, start_time)

    cap.release()
    cv2.destroyAllWindows()

    print_final_report(car_stats, min_speed_sentinel=MIN_SPEED_SENTINEL)


if __name__ == "__main__":
    main()
