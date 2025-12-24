# Stop Sign Monitor

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)

## Project Overview

Stop Sign Monitor is a computer vision system designed to monitor and analyze stop sign compliance at intersections. The system detects and tracks vehicles approaching stop signs, calculates their speed, and identifies rolling stops or complete failures to stop.

This project was initially developed to monitor traffic behavior at the intersection of **Carter Avenue and North Jefferson Street**, which features an atypical three-way stop configuration. The system utilizes deep learning object detection (YOLOv8) to track vehicles, calculate velocity vectors, and autonomously identify traffic violations in real-time from video files or live streams (YouTube, RTSP, or direct camera feeds).

## Objectives

1.  **Quantify Compliance:** Accurately distinguish between complete stops (0 mph), rolling stops (<3 mph), and violations (>=3 mph).
2.  **Automated Logging:** Capture time-stamped evidence of non-compliant vehicles.
3.  **Edge Deployment:** Operate fully autonomously on embedded hardware with solar power, independent of the electrical grid.

## Methodology

### The Detection Logic

The system relies on a "Digital Twin" simulation validated against video feeds before hardware deployment.

1.  **Object Tracking:** Uses YOLOv8 with built-in tracking (ByteTrack by default) to assign persistent IDs to vehicles.
2.  **Zone-Based Analysis:** A user-defined polygon (Stop Zone) monitors the specific area preceding the limit line.
3.  **Velocity Estimation:** Calculates speed from weighted pixel displacement over time, using perspective-corrected pixels-per-meter.
4.  **Violation Trigger:**
    - System records the _minimum_ speed achieved by a vehicle while inside the Stop Zone.
    - If `Min_Speed >= Threshold` upon exiting the zone, the event is flagged as a violation.

### Current Software Stack

- **Language:** Python 3.8+
- **Computer Vision:** OpenCV (`cv2`)
- **Inference Engine:** Ultralytics YOLOv8 (Nano/Small models)
- **Data Processing:** NumPy
- **Optional:** Hugging Face Hub (for model downloads)

## Hardware Implementation Plan

The project will transition from a desktop-based simulation to a ruggedized edge-computing unit mounted on-site.

### Compute Module

- **Device:** Raspberry Pi 5 (16GB RAM)
- **AI Accelerator:** Raspberry Pi AI HAT+ (26 TOPS Hailo-8 Module).
  - _Justification:_ The 26 TOPS accelerator allows for high-framerate processing of multiple video streams simultaneously, essential for accurate speed estimation.

### Optical Configuration

- **Placement:** North Jefferson Street Stop Sign (South-facing).
- **Dual-Camera "Cross-Fire" Setup:**
  - **Camera A (West-Facing):** Monitors Eastbound traffic approach.
  - **Camera B (East-Facing):** Monitors Westbound traffic approach.
- **Lens Selection:** Standard FOV (approx. 75°) to maximize pixel density on the stop line for precise motion tracking.

### Power & Autonomy

Due to the lack of grid infrastructure at the installation site, the unit operates on a standalone solar circuit.

- **Solar Array:** 100W+ Monocrystalline Rigid Panel (South-facing mount).
- **Energy Storage:** 12V 50Ah-100Ah LiFePO4 Battery.
- **Regulation:** MPPT Charge Controller to maximize solar efficiency during overcast Iowa winters.

## Project Structure

```
stop-sign-monitor/
├── src/
│   └── traffic_monitor/      # Source code package
│       ├── __init__.py
│       ├── config.py         # Configuration settings
│       ├── perspective.py    # Perspective correction utilities
│       ├── report.py         # Report generation
│       ├── speed.py          # Speed estimation logic
│       └── traffic_sim.py   # Main simulation logic
├── models/                    # Model files (.pt)
│   ├── yolov8n.pt
│   └── license-plate-finetune-v1x.pt
├── data/
│   ├── videos/               # Test video files
│   │   └── traffic_test.mp4
│   └── output/               # Generated outputs
│       └── violation_snapshots/
├── traffic_sim.py            # Entry point script
├── requirements.txt          # Core dependencies
├── requirements-dev.txt       # Development dependencies
├── pyproject.toml            # Package configuration
├── MANIFEST.in              # Package manifest
├── LICENSE                  # MIT License
├── .gitignore              # Git ignore rules
├── README.md
└── X11_SETUP.md
```

## Setup & Usage

### Prerequisites

- Python 3.8 or higher
- Virtual Environment (Recommended)

### Installation

#### Option 1: Install from Source (Recommended)

1.  Clone the repository and navigate to the project root:

    ```bash
    git clone https://github.com/yourusername/stop-sign-monitor.git
    cd stop-sign-monitor
    ```

2.  Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    Optional: For license plate detection from Hugging Face:

    ```bash
    pip install -r requirements.txt huggingface-hub
    ```

    Or install with optional dependencies:

    ```bash
    pip install -e ".[hf]"
    ```

#### Option 2: Install as Package

Install the package in development mode:

```bash
pip install -e .
```

Or with optional dependencies:

```bash
pip install -e ".[hf]"
```

#### Development Setup

For development with additional tools (linting, testing, etc.):

```bash
pip install -r requirements-dev.txt
```

### Execution

#### Using Video Files

1.  Place a test video file named `traffic_test.mp4` in the `data/videos/` directory.
2.  Ensure model files are in the `models/` directory:
    - `yolov8n.pt` (will be auto-downloaded by Ultralytics if missing)
    - License plate models will be downloaded from Hugging Face if configured
3.  Run the monitor:
    ```bash
    python traffic_sim.py
    ```
    Or if installed as a package:
    ```bash
    stop-sign-monitor
    ```
4.  **Calibration:**
    - The script will pause on the first frame.
    - Click four points to define the "Stop Zone" on the road surface.
    - Press `SPACE` to commence analysis.
    - Perspective correction is applied automatically using the frame height reference.

#### Using Live Streams

_Note: Live stream support (YouTube, RTSP, camera feeds) is planned for future releases. The current version supports video file analysis._

### Output

The system renders a real-time view with bounding boxes color-coded by status:

- **Green:** Approaching / Outside Zone
- **Blue:** Clean Stop (Speed dropped below threshold)
- **Red:** Rolling stop detected (flagged after the vehicle leaves the zone)

A summary report of all tracked vehicle IDs and their compliance status is generated in the console upon termination.
For each rolling stop, a cropped snapshot is saved to `data/output/violation_snapshots/` using the vehicle's track ID.
If a license plate model is available, a second crop of the plate is saved alongside it.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [OpenCV](https://opencv.org/) for computer vision utilities
