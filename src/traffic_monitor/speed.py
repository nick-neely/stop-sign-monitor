from collections import defaultdict

import numpy as np

from .perspective import get_pixels_per_meter


class SpeedEstimator:
    """Estimate per-track speed with perspective correction and smoothing."""

    def __init__(
        self,
        fps,
        frame_height,
        base_pixels_per_meter,
        perspective_factor,
        reference_y_ratio,
        intervals,
        smoothing_alpha,
        min_history_frames,
        horizontal_weight,
        vertical_weight,
    ):
        self.fps = fps
        self.frame_height = frame_height
        self.base_pixels_per_meter = base_pixels_per_meter
        self.perspective_factor = perspective_factor
        self.reference_y_ratio = reference_y_ratio
        self.intervals = intervals
        self.smoothing_alpha = smoothing_alpha
        self.min_history_frames = min_history_frames
        self.horizontal_weight = horizontal_weight
        self.vertical_weight = vertical_weight
        self.track_history = defaultdict(list)
        self.smoothed_speeds = {}

    def update(self, track_id, bottom_center):
        history = self.track_history[track_id]
        history.append(bottom_center)

        if len(history) < self.min_history_frames:
            return 0.0

        speeds = []
        for interval in self.intervals:
            if len(history) < interval:
                continue
            prev_bottom_center = history[-interval]
            pixel_dist = self._weighted_pixel_distance(bottom_center, prev_bottom_center)
            pixels_per_meter = get_pixels_per_meter(
                prev_bottom_center[1],
                self.frame_height,
                self.base_pixels_per_meter,
                self.perspective_factor,
                self.reference_y_ratio,
            )
            meters = pixel_dist / pixels_per_meter
            time_seconds = interval / self.fps
            speed_ms = meters / time_seconds if time_seconds > 0 else 0.0
            speeds.append(speed_ms * 2.23694)

        if not speeds:
            return 0.0

        raw_speed = float(np.mean(speeds))
        previous = self.smoothed_speeds.get(track_id, raw_speed)
        smoothed = self.smoothing_alpha * raw_speed + (1 - self.smoothing_alpha) * previous
        self.smoothed_speeds[track_id] = smoothed
        return smoothed

    def _weighted_pixel_distance(self, current, previous):
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        return np.sqrt(
            (self.horizontal_weight * dx) ** 2
            + (self.vertical_weight * dy) ** 2
        )
