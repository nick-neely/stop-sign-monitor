import argparse
import json
import os
import subprocess
from collections import deque
import sys
import time
import threading
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from ultralytics import YOLO
try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

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
MAX_STOP_ZONE_POINTS = 4
SETUP_BAR_HEIGHT = 70
STATUS_BAR_HEIGHT = 30
LIVE_DRAIN_GRABS = 6
MJPEG_BOUNDARY = "frame"


def parse_args(argv=None):
    """Parse CLI args for video file or stream input."""
    parser = argparse.ArgumentParser(
        description="Run stop sign monitor on a video file or live stream.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--video-path",
        help="Path to local video file (defaults to config VIDEO_PATH).",
    )
    group.add_argument(
        "--stream-url",
        help="URL to a live stream (YouTube, RTSP, HLS, or direct camera feed).",
    )
    parser.add_argument(
        "--no-live-drop",
        action="store_true",
        help="Disable frame dropping for live streams (not recommended).",
    )
    parser.add_argument(
        "--no-threaded-capture",
        action="store_true",
        help="Disable background capture thread for live streams.",
    )
    parser.add_argument(
        "--stream-quality",
        type=int,
        default=0,
        help="Max stream height for YouTube (e.g., 360, 480, 720).",
    )
    parser.add_argument(
        "--yt-format",
        help="Override yt-dlp format selector (e.g., 'best[height<=480]').",
    )
    parser.add_argument(
        "--yt-cookies",
        help="Path to a cookies.txt file for yt-dlp (useful for restricted streams).",
    )
    parser.add_argument(
        "--yt-cookies-from-browser",
        help="Load cookies from a browser (e.g., chrome, firefox).",
    )
    parser.add_argument(
        "--ffmpeg-relay",
        action="store_true",
        help="Relay the live stream through ffmpeg for lower latency.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="Path to ffmpeg binary (defaults to ffmpeg in PATH).",
    )
    parser.add_argument(
        "--ffmpeg-loglevel",
        default="error",
        help="ffmpeg log level (quiet, error, warning, info).",
    )
    parser.add_argument(
        "--ffmpeg-fps",
        type=float,
        default=0.0,
        help="Limit ffmpeg relay FPS (0 = keep source rate).",
    )
    parser.add_argument(
        "--ffmpeg-scale",
        type=float,
        default=1.0,
        help="Scale factor for ffmpeg relay (e.g., 0.6).",
    )
    parser.add_argument(
        "--mjpeg-host",
        default="0.0.0.0",
        help="Host address for MJPEG preview server.",
    )
    parser.add_argument(
        "--mjpeg-port",
        type=int,
        default=0,
        help="Port for MJPEG preview server (0 disables).",
    )
    parser.add_argument(
        "--mjpeg-fps",
        type=float,
        default=0.0,
        help="Limit MJPEG preview to this FPS (0 = unlimited).",
    )
    parser.add_argument(
        "--mjpeg-scale",
        type=float,
        default=1.0,
        help="Scale factor for MJPEG preview frames.",
    )
    parser.add_argument(
        "--zone-file",
        help="Load stop-zone points from a JSON file.",
    )
    parser.add_argument(
        "--save-zone",
        help="Save stop-zone points to a JSON file after setup.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the OpenCV window (use with --zone-file or --save-zone).",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="Scale factor for the display window (e.g., 0.6).",
    )
    parser.add_argument(
        "--display-fps",
        type=float,
        default=0.0,
        help="Limit display updates to this FPS (0 = unlimited).",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Overlay the processing FPS.",
    )
    return parser.parse_args(argv)


def load_zone_points(path, max_points=MAX_STOP_ZONE_POINTS):
    """Load stop-zone points from a JSON file."""
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to read zone file {path}: {exc}")
        return None
    if not isinstance(data, list):
        print(f"Zone file {path} must contain a list of points.")
        return None
    points = []
    for entry in data:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        try:
            x = int(entry[0])
            y = int(entry[1])
        except (TypeError, ValueError):
            continue
        points.append((x, y))
    if len(points) < 3:
        print(f"Zone file {path} does not contain enough points.")
        return None
    if len(points) > max_points:
        points = points[:max_points]
    return points


def save_zone_points(path, points):
    """Save stop-zone points to a JSON file."""
    if not path or not points:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(points, handle, indent=2)
    except OSError as exc:
        print(f"Failed to save zone file {path}: {exc}")


class MJPEGServer:
    """Serve the latest frame as an MJPEG stream over HTTP."""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._frame_lock = threading.Lock()
        self._frame = None
        self._thread = None
        self._server = None

    def _make_handler(self):
        server_ref = self

        class MJPEGHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    html = (
                        "<!doctype html><html><head><title>Stop Sign Monitor</title>"
                        "<style>body{margin:0;background:#111;color:#eee;font-family:Arial,Helvetica,sans-serif;}"
                        ".wrap{display:flex;align-items:center;justify-content:center;height:100vh;}"
                        "img{max-width:100%;height:auto;border:2px solid #333;}</style>"
                        "</head><body><div class='wrap'>"
                        "<img src='/stream.mjpg' alt='Live preview'/>"
                        "</div></body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(html.encode("utf-8"))
                    return

                if self.path.startswith("/stream.mjpg"):
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
                    )
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    boundary = f"--{MJPEG_BOUNDARY}\r\n".encode("ascii")
                    while True:
                        frame = server_ref.get_frame()
                        if frame is None:
                            time.sleep(0.05)
                            continue
                        try:
                            self.wfile.write(boundary)
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            break
                        except Exception:
                            break
                    return

                self.send_response(404)
                self.end_headers()

            def log_message(self, format, *args):
                return

        return MJPEGHandler

    def get_frame(self):
        with self._frame_lock:
            return self._frame

    def set_frame(self, frame_bytes):
        with self._frame_lock:
            self._frame = frame_bytes

    def start(self):
        handler = self._make_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self._server.server_address

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=1)


class FFmpegPipeReader:
    """Read MJPEG frames from an ffmpeg stdout pipe."""

    def __init__(self, process):
        self.process = process
        self._frame_lock = threading.Lock()
        self._frame = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        buffer = bytearray()
        while not self._stop.is_set():
            if not self.process.stdout:
                break
            chunk = self.process.stdout.read(4096)
            if not chunk:
                if self.process.poll() is not None:
                    break
                time.sleep(0.01)
                continue
            buffer.extend(chunk)
            while True:
                start = buffer.find(b"\xff\xd8")
                if start == -1:
                    if len(buffer) > 1024:
                        buffer.clear()
                    break
                end = buffer.find(b"\xff\xd9", start + 2)
                if end == -1:
                    if start > 0:
                        del buffer[:start]
                    break
                jpg = buffer[start : end + 2]
                del buffer[: end + 2]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._frame_lock:
                        self._frame = frame

    def read(self):
        with self._frame_lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def wait_for_frame(self, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            ret, _frame = self.read()
            if ret:
                return True
            if self.process.poll() is not None:
                break
            time.sleep(0.05)
        return False

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)


class FFmpegRelay:
    """Relay a live stream through ffmpeg to a MJPEG pipe."""

    def __init__(self, input_url, fps, scale, ffmpeg_path, headers=None, loglevel="error"):
        self.input_url = input_url
        self.fps = fps
        self.scale = scale
        self.ffmpeg_path = ffmpeg_path
        self.headers = headers or {}
        self.loglevel = loglevel
        self.process = None
        self.reader = None
        self._stderr_thread = None
        self._stderr_lines = deque(maxlen=40)

    def _build_filter(self):
        filters = []
        if self.fps and self.fps > 0:
            filters.append(f"fps={self.fps}")
        if self.scale and self.scale > 0 and self.scale != 1.0:
            filters.append(f"scale=iw*{self.scale}:ih*{self.scale}:flags=fast_bilinear")
        return ",".join(filters)

    def _build_headers(self):
        allowed = {
            "user-agent",
            "referer",
            "origin",
            "cookie",
            "accept",
            "accept-language",
        }
        filtered = {}
        for key, value in self.headers.items():
            key_lower = key.lower()
            if key_lower in allowed:
                filtered[key_lower] = value
        user_agent = filtered.pop("user-agent", None)
        referer = filtered.pop("referer", None)
        header_lines = "".join(f"{key}: {value}\r\n" for key, value in filtered.items())
        return user_agent, referer, header_lines

    def start(self):
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            self.loglevel,
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_at_eof",
            "1",
            "-reconnect_delay_max",
            "2",
            "-rw_timeout",
            "5000000",
            "-protocol_whitelist",
            "file,http,https,tcp,tls,crypto",
            "-allowed_extensions",
            "ALL",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-probesize",
            "1000000",
            "-analyzeduration",
            "1000000",
        ]
        user_agent, referer, header_lines = self._build_headers()
        if user_agent:
            cmd.extend(["-user_agent", user_agent])
        if referer:
            cmd.extend(["-referer", referer])
        if header_lines:
            cmd.extend(["-headers", header_lines])
        cmd.extend(
            [
                "-i",
                self.input_url,
                "-an",
            "-sn",
            "-dn",
            "-map",
            "0:v:0",
            ]
        )
        vf = self._build_filter()
        if vf:
            cmd.extend(["-vf", vf])
        cmd.extend(
            [
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-q:v",
                "5",
                "pipe:1",
            ]
        )
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to start ffmpeg: {exc}") from exc
        self._stderr_thread = threading.Thread(target=self._capture_stderr, daemon=True)
        self._stderr_thread.start()
        self.reader = FFmpegPipeReader(self.process)
        if not self.reader.wait_for_frame(timeout=15.0):
            error_output = self.get_error_output() or self.get_recent_logs()
            raise RuntimeError(error_output or "ffmpeg did not produce any frames.")
        return "pipe"

    def _capture_stderr(self):
        if not self.process or not self.process.stderr:
            return
        for line in self.process.stderr:
            try:
                decoded = line.decode("utf-8", errors="replace").strip()
            except Exception:
                decoded = ""
            if decoded:
                self._stderr_lines.append(decoded)

    def get_error_output(self):
        if self.process and self.process.poll() is not None and self.process.stderr:
            return self.process.stderr.read().decode("utf-8", errors="replace").strip()
        return ""

    def get_recent_logs(self):
        return "\n".join(self._stderr_lines)

    def stop(self):
        if self.reader:
            self.reader.stop()
        if not self.process:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None


def describe_stream_source(stream_url):
    """Return a short label for the stream source."""
    parsed = urlparse(stream_url)
    host = parsed.netloc.lower()
    if "youtube" in host or "youtu.be" in host:
        return "YouTube Live"
    if stream_url.lower().startswith("rtsp://"):
        return "RTSP Stream"
    if host:
        return host
    return "Live Stream"


def resolve_stream_source(
    stream_url,
    yt_format=None,
    stream_quality=0,
    cookies_path=None,
    cookies_from_browser=None,
):
    """Resolve a stream URL to a direct media source and label."""
    if not stream_url:
        return None, None, {}
    stream_url_lower = stream_url.lower()
    label = describe_stream_source(stream_url)
    if "youtube.com" in stream_url_lower or "youtu.be" in stream_url_lower:
        if not YTDLP_AVAILABLE:
            raise RuntimeError("yt-dlp is required for YouTube streams. Install with: pip install yt-dlp")
        format_selector = yt_format
        if not format_selector:
            if stream_quality and stream_quality > 0:
                format_selector = f"best[height<={stream_quality}]"
            else:
                format_selector = "best"
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "format": format_selector,
            "noplaylist": True,
            "live_from_start": False,
            "hls_live_edge": 1,
        }
        if cookies_path:
            ydl_opts["cookiefile"] = cookies_path
        if cookies_from_browser:
            ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(stream_url, download=False)
            except Exception as exc:
                message = str(exc)
                if "Requested format is not available" in message:
                    raise RuntimeError(
                        "Requested format is not available. Run `yt-dlp -F <url>` "
                        "and choose a valid format, or use --yt-format \"best\"."
                    ) from exc
                raise RuntimeError(f"yt-dlp error: {message}") from exc
        if info and "entries" in info:
            info = next((entry for entry in info["entries"] if entry), None)
        if not info or "url" not in info:
            raise RuntimeError("Unable to resolve YouTube stream URL.")
        headers = info.get("http_headers") or {}
        return info["url"], label, headers
    return stream_url, label, {}


def draw_setup_overlay(frame, point_count, max_points):
    """Draw setup hints on top of the calibration frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], SETUP_BAR_HEIGHT), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(
        frame,
        "STOP ZONE SETUP",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Points: {point_count}/{max_points}",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Click to add  |  Backspace: undo  R: reset  Space: start  Q: quit",
        (220, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def draw_runtime_overlay(frame, source_label, is_live):
    """Render a small status banner during playback."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], STATUS_BAR_HEIGHT), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    badge_color = (0, 0, 255) if is_live else (0, 180, 0)
    cv2.circle(frame, (12, 15), 6, badge_color, -1)
    label = "LIVE" if is_live else "FILE"
    text = f"{label} | {source_label}" if source_label else label
    cv2.putText(
        frame,
        text,
        (26, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    quit_text = "Q: quit"
    text_size = cv2.getTextSize(quit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    x_pos = max(12, frame.shape[1] - text_size[0] - 12)
    cv2.putText(
        frame,
        quit_text,
        (x_pos, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def draw_stop_zone_points(frame, stop_zone_polygon):
    """Draw the zone points with labels."""
    for idx, (x, y) in enumerate(stop_zone_polygon, start=1):
        cv2.circle(frame, (x, y), 6, (0, 215, 255), -1)
        cv2.putText(
            frame,
            str(idx),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def read_latest_frame(cap, max_grabs=LIVE_DRAIN_GRABS):
    """Grab and return the most recent frame, dropping older ones."""
    grabbed = False
    for _ in range(max_grabs):
        if not cap.grab():
            break
        grabbed = True
    if grabbed:
        return cap.retrieve()
    return cap.read()


class FrameGrabber:
    """Continuously read frames in the background and keep only the latest."""

    def __init__(self, cap):
        self.cap = cap
        self._frame_lock = threading.Lock()
        self._frame = None
        self._ret = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def start(self):
        if not self._started:
            self._thread.start()
            self._started = True

    def _run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._frame_lock:
                self._frame = frame
                self._ret = True

    def read(self):
        with self._frame_lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)


def setup_stop_zone(
    cap,
    window_name,
    live_preview=False,
    max_points=MAX_STOP_ZONE_POINTS,
    frame_reader=None,
):
    """Capture a polygonal stop zone from user clicks on the first frame."""
    stop_zone_polygon = []

    window_initialized = False

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(stop_zone_polygon) < max_points:
            stop_zone_polygon.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("--- SETUP ---")
    print(f"Draw your zone ({max_points} clicks). Press SPACE to start.")

    def read_frame():
        if frame_reader:
            for _ in range(50):
                ret, frame = frame_reader()
                if ret:
                    return ret, frame
                time.sleep(0.02)
            return False, None
        if live_preview:
            return read_latest_frame(cap)
        return cap.read()

    ret, frame = read_frame()
    if not ret:
        return stop_zone_polygon

    dynamic_preview = live_preview or frame_reader is not None

    while True:
        if dynamic_preview:
            ret, frame = read_frame()
            if not ret:
                return stop_zone_polygon
        if not window_initialized:
            cv2.resizeWindow(window_name, frame.shape[1], frame.shape[0])
            window_initialized = True
        temp_frame = frame.copy()
        if len(stop_zone_polygon) >= 2:
            cv2.polylines(
                temp_frame,
                [np.array(stop_zone_polygon)],
                len(stop_zone_polygon) >= max_points,
                (0, 255, 255),
                2,
            )
        if stop_zone_polygon:
            draw_stop_zone_points(temp_frame, stop_zone_polygon)
        draw_setup_overlay(temp_frame, len(stop_zone_polygon), max_points)
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32 and len(stop_zone_polygon) >= max_points:
            break
        if key in (8, 127) and stop_zone_polygon:
            stop_zone_polygon.pop()
        if key in (ord("r"), ord("R")):
            stop_zone_polygon.clear()
        if key in (ord("q"), 27):
            return []

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


def main(argv=None):
    """Main entry point for the traffic simulation."""
    args = parse_args(argv)
    stream_url = args.stream_url
    video_path = args.video_path or VIDEO_PATH
    display_scale = args.display_scale if args.display_scale > 0 else 1.0
    display_fps = args.display_fps if args.display_fps > 0 else 0.0
    display_enabled = not args.no_display
    mjpeg_scale = args.mjpeg_scale if args.mjpeg_scale > 0 else 1.0
    mjpeg_fps = args.mjpeg_fps if args.mjpeg_fps > 0 else 0.0
    stream_quality = args.stream_quality if args.stream_quality and args.stream_quality > 0 else 0
    yt_format = args.yt_format
    ffmpeg_scale = args.ffmpeg_scale if args.ffmpeg_scale > 0 else 1.0
    ffmpeg_fps = args.ffmpeg_fps if args.ffmpeg_fps > 0 else 0.0
    cookies_path = args.yt_cookies
    cookies_from_browser = args.yt_cookies_from_browser
    if args.ffmpeg_relay and not yt_format:
        if stream_quality:
            yt_format = f"bestvideo[height<={stream_quality}]/best[height<={stream_quality}]"
        else:
            yt_format = "bestvideo/best"

    if stream_url is None and any(arg.startswith("--stream-url") for arg in sys.argv[1:]):
        print("Stream URL flag provided but not parsed. Check shell quoting.")
        return 1

    source_label = None
    ffmpeg_relay = None
    if stream_url:
        try:
            stream_source, source_label, stream_headers = resolve_stream_source(
                stream_url,
                yt_format=yt_format,
                stream_quality=stream_quality,
                cookies_path=cookies_path,
                cookies_from_browser=cookies_from_browser,
            )
        except RuntimeError as exc:
            print(str(exc))
            return 1
        if args.ffmpeg_relay:
            ffmpeg_relay = FFmpegRelay(
                stream_source,
                ffmpeg_fps,
                ffmpeg_scale,
                args.ffmpeg_path,
                headers=stream_headers,
                loglevel=args.ffmpeg_loglevel,
            )
            try:
                stream_source = ffmpeg_relay.start()
            except RuntimeError as exc:
                print(str(exc))
                return 1
            source_label = "FFmpeg Relay"
        if args.ffmpeg_relay:
            cap = None
        else:
            cap = cv2.VideoCapture(stream_source)
        source_label = source_label or "Live Stream"
    else:
        if args.ffmpeg_relay:
            print("--ffmpeg-relay requires --stream-url.")
            return 1
        cap = cv2.VideoCapture(video_path)
        source_label = os.path.basename(video_path)

    if cap is not None and not cap.isOpened():
        print(f"Failed to open video source: {source_label}")
        if ffmpeg_relay:
            error_output = ffmpeg_relay.get_error_output()
            if error_output:
                print(f"ffmpeg error: {error_output}")
            ffmpeg_relay.stop()
        return 1

    if cap is not None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    else:
        fps = ffmpeg_fps or 30.0
        total_frames = 0
    print(f"Video FPS: {fps:.2f}")
    start_time = time.time()
    is_live = bool(stream_url) or total_frames <= 0
    live_drop = is_live and cap is not None and not args.no_live_drop
    if is_live and cap is not None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    use_threaded = is_live and not args.no_threaded_capture and cap is not None
    if use_threaded:
        live_drop = False
    last_display_time = 0.0
    last_fps_time = time.time()
    fps_counter = 0
    fps_value = 0.0
    last_mjpeg_time = 0.0
    mjpeg_server = None
    if args.mjpeg_port > 0:
        mjpeg_server = MJPEGServer(args.mjpeg_host, args.mjpeg_port)
        try:
            host, port = mjpeg_server.start()
        except OSError as exc:
            print(f"Failed to start MJPEG server: {exc}")
            mjpeg_server = None
        else:
            preview_host = args.mjpeg_host
            if preview_host in ("0.0.0.0", "::"):
                preview_host = "localhost"
            print(f"MJPEG preview: http://{preview_host}:{port}/")
    if ffmpeg_relay:
        print("FFmpeg relay: active")

    window_name = "Traffic Sim"
    grabber = None
    if use_threaded:
        grabber = FrameGrabber(cap)
        grabber.start()
    frame_reader = ffmpeg_relay.reader.read if ffmpeg_relay and ffmpeg_relay.reader else None
    stop_zone_polygon = load_zone_points(args.zone_file)
    if args.zone_file and not stop_zone_polygon:
        print(f"Zone file {args.zone_file} not found or invalid.")
    if not stop_zone_polygon:
        if not display_enabled:
            print("Display disabled and no zone file available. Provide --zone-file.")
            if mjpeg_server:
                mjpeg_server.stop()
            if ffmpeg_relay:
                ffmpeg_relay.stop()
            if grabber:
                grabber.stop()
            cap.release()
            return 1
        stop_zone_polygon = setup_stop_zone(
            cap,
            window_name,
            live_preview=is_live,
            frame_reader=frame_reader or (grabber.read if grabber else None),
        )
    if not stop_zone_polygon:
        print("Stop zone setup canceled.")
        if mjpeg_server:
            mjpeg_server.stop()
        if ffmpeg_relay:
            ffmpeg_relay.stop()
        if grabber:
            grabber.stop()
        cap.release()
        cv2.destroyAllWindows()
        return 0

    if args.save_zone:
        save_zone_points(args.save_zone, stop_zone_polygon)

    if cap is not None and not is_live:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    model = YOLO(os.path.join(MODELS_DIR, "yolov8n.pt"))
    plate_model = load_license_plate_model(LICENSE_PLATE_MODEL_PATH)
    car_stats = {}
    frame_height = None
    speed_estimator = None
    os.makedirs(VIOLATION_SNAPSHOT_DIR, exist_ok=True)
    missed_frames = 0

    while True:
        if use_threaded and grabber:
            ret, frame = grabber.read()
        elif ffmpeg_relay and ffmpeg_relay.reader:
            ret, frame = ffmpeg_relay.reader.read()
        elif live_drop:
            ret, frame = read_latest_frame(cap)
        else:
            ret, frame = cap.read()
        if not ret:
            if is_live and missed_frames < 50:
                missed_frames += 1
                time.sleep(0.02)
                continue
            if total_frames > 0:
                render_progress(total_frames, total_frames, fps, start_time)
                sys.stdout.write("\n")
                sys.stdout.flush()
            if is_live:
                print("\nStream ended. Generating report...")
            else:
                print("\nVideo ended. Generating report...")
            break
        missed_frames = 0

        if frame_height is None:
            frame_height = frame.shape[0]
            if display_enabled:
                display_width = max(1, int(frame.shape[1] * display_scale))
                display_height = max(1, int(frame.shape[0] * display_scale))
                cv2.resizeWindow(window_name, display_width, display_height)
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

        now = time.time()
        fps_counter += 1
        if now - last_fps_time >= 1.0:
            fps_value = fps_counter / (now - last_fps_time)
            fps_counter = 0
            last_fps_time = now

        if stop_zone_polygon:
            cv2.polylines(frame, [np.array(stop_zone_polygon)], True, (0, 255, 255), 2)
        draw_runtime_overlay(frame, source_label, is_live)
        if args.show_fps:
            cv2.putText(
                frame,
                f"FPS: {fps_value:.1f}",
                (12, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if mjpeg_server:
            if mjpeg_fps <= 0 or now - last_mjpeg_time >= 1.0 / mjpeg_fps:
                mjpeg_frame = frame
                if mjpeg_scale != 1.0:
                    mjpeg_frame = cv2.resize(
                        frame,
                        (int(frame.shape[1] * mjpeg_scale), int(frame.shape[0] * mjpeg_scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                success, encoded = cv2.imencode(".jpg", mjpeg_frame)
                if success:
                    mjpeg_server.set_frame(encoded.tobytes())
                    last_mjpeg_time = now
        if display_enabled:
            display_frame = frame
            if display_scale != 1.0:
                display_frame = cv2.resize(
                    frame,
                    (int(frame.shape[1] * display_scale), int(frame.shape[0] * display_scale)),
                    interpolation=cv2.INTER_AREA,
                )
            if display_fps > 0:
                if now - last_display_time >= 1.0 / display_fps:
                    cv2.imshow(window_name, display_frame)
                    last_display_time = now
            else:
                cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) == ord("q"):
                break
        if not is_live and total_frames > 0 and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % PROGRESS_UPDATE_EVERY == 0:
            render_progress(int(cap.get(cv2.CAP_PROP_POS_FRAMES)), total_frames, fps, start_time)

    cap.release()
    if mjpeg_server:
        mjpeg_server.stop()
    if ffmpeg_relay:
        ffmpeg_relay.stop()
    if grabber:
        grabber.stop()
    cv2.destroyAllWindows()

    print_final_report(car_stats, min_speed_sentinel=MIN_SPEED_SENTINEL)


if __name__ == "__main__":
    main()
