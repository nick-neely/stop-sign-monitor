def get_pixels_per_meter(
    y_position,
    frame_height,
    base_pixels_per_meter,
    perspective_factor,
    reference_y_ratio,
):
    """Return pixels-per-meter corrected for perspective at a given y-position."""
    reference_y = frame_height * reference_y_ratio

    if y_position < 10:
        y_position = 10

    y_ratio = y_position / reference_y

    if y_ratio >= 1.0:
        scale_factor = 1.0 + perspective_factor * (y_ratio - 1.0) ** 1.5
    else:
        scale_factor = 1.0 - perspective_factor * 0.5 * (1.0 - y_ratio) ** 1.2

    scale_factor = max(0.3, min(3.0, scale_factor))
    return base_pixels_per_meter * scale_factor
