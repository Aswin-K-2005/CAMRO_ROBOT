# 🤖 CAMRO — AI Powered Victim Detection Rescue Robot

> A tank-treaded autonomous disaster robot designed to locate earthquake survivors in rubble using real-time YOLOv8 human detection.

---

## 📌 Overview

TERRA is an autonomous rescue robot built for post-earthquake search operations. It streams live video from an ESP32-CAM to a laptop, where a fine-tuned YOLOv8n model detects humans in real time — even in near-darkness and under rubble — and sends directional motor commands to navigate toward survivors.

---

## 🏗️ System Architecture

```
Environment (Rubble/Disaster Scene)
        │
        ▼
  ESP32-CAM (MJPEG stream on :81/stream)
        │  WiFi / Ethernet Tether / LoRa
        ▼
  Laptop — RTX 4070 (AI Brain)
  ├── detect.py
  │   ├── FrameGrabber (threaded MJPEG parser)
  │   ├── CLAHE dark enhancement
  │   ├── YOLOv8n inference (CUDA)
  │   ├── Aspect ratio filter (prone human detection)
  │   ├── Exponential smoothing tracker
  │   └── Decision: LEFT / RIGHT / CENTER / CLOSE
        │  Serial (USB)
        ▼
  Arduino (Motor Controller)
  └── L298N Motor Driver → DD1 Tank Chassis
```

---

## ⚙️ Hardware

| Component | Role |
|---|---|
| ESP32-CAM | Live MJPEG video stream |
| Arduino | Motor control via serial commands |
| L298N Motor Driver | PWM signals to tank tracks |
| DD1 Military Tank Chassis | Locomotion |
| Laptop (RTX 4070, i9) | All AI inference and decision-making |

**Communication layers (redundant):**
1. Ethernet tether (primary)
2. WiFi mesh (secondary)
3. LoRa (backup — works underground/long range)

---

## 🧠 Software

### `detect.py` — Main Detection & Control Loop

- Connects to ESP32-CAM MJPEG stream via a dedicated `FrameGrabber` thread
- Parses raw byte stream using JPEG markers (`0xFF 0xD8` / `0xFF 0xD9`)
- Runs YOLOv8n inference every **0.45 seconds** (not every frame — avoids GPU saturation)
- Detects dark frames automatically → applies **CLAHE enhancement** (LAB color space)
- Filters detections by aspect ratio to catch **prone humans** (lying under rubble)
- Exponential smoothing tracker maintains target lock between detection cycles
- Issues directional commands: `HUMAN LEFT`, `HUMAN RIGHT`, `HUMAN CENTER`, `HUMAN CLOSE`

**Key config values:**

| Parameter | Value | Reason |
|---|---|---|
| `CONFIDENCE` | 0.22 | Low threshold — missing a human is worse than a false alarm |
| `DARK_CONFIDENCE` | 0.18 | Even lower for shadowy/dark scenes |
| `DARK_THRESHOLD` | 60 | Mean pixel brightness below this triggers CLAHE |
| `DETECTION_INTERVAL` | 0.45s | Balances real-time response vs GPU efficiency |
| `CAMERA_STOP_WIDTH_RATIO` | 0.38 | Triggers CLOSE when human fills 38% of frame width |
| `CENTER_DEADZONE` | 0.18 | Ignores small offsets to prevent jitter |
| `TRACKING_ALPHA` | 0.35 | Exponential smoothing factor for tracker |

---

### `prepare_dataset.py` — Dataset Builder & Trainer

- Downloads **2,000 person-annotated images** from COCO 2017 validation set
- Converts COCO `[x, y, w, h]` annotations → YOLO normalized `[cx, cy, w, h]` format
- Augments every image using **Albumentations**:
  - Random brightness reduction → simulates darkness
  - Gaussian noise → simulates low-quality camera feeds
  - Motion blur → simulates moving robot
  - Coarse dropout → simulates humans partially buried under rubble
  - Random fog → simulates dusty post-earthquake air
- Effectively expands dataset to ~**5,200 training images**
- Splits 90/10 into train/val sets
- Fine-tunes `yolov8n.pt` for **60 epochs** on CUDA

**Optional:** WiderPerson dataset support is already coded in — place `widerperson.zip` at `dataset/widerperson.zip` and re-run to significantly improve recall.

---

## 📊 Training Results (60 epochs, COCO 2017 only)

| Metric | Value | Target | Status |
|---|---|---|---|
| Precision | 65% | > 70% | 🟡 Decent |
| Recall | 41% | > 75% | 🔴 Needs improvement |
| mAP50 | 45.7% | > 60% | 🟡 Acceptable |
| mAP50-95 | 26% | Low priority | 🔵 OK for navigation |
| Best F1 | 0.50 @ conf 0.227 | > 0.65 | 🟡 |
| Max recall (any conf) | 69% | > 75% | 🟡 |

> **Key insight:** Recall is the critical metric for rescue — missing a survivor is catastrophic. More training epochs (120+) and adding the WiderPerson dataset are the primary paths to improvement.

---

## 🚀 Getting Started

### Requirements

```bash
pip install ultralytics albumentations opencv-python tqdm pyyaml requests numpy
```

### 1. Build Dataset & Train

```bash
python prepare_dataset.py
```

This will:
- Download COCO 2017 annotations and images (~2 GB)
- Augment and split the dataset
- Train YOLOv8n for 60 epochs
- Save weights to `runs/rescue/rescue_finetune/weights/best.pt`

### 2. Run Detection

```bash
python detect.py
```

Make sure to update the ESP32-CAM IP address in `detect.py`:
```python
ESP32_IP = "http://192.168.x.x"   # ← your ESP32-CAM's local IP
```

Press `Q` to quit.

---

## 🔧 Improving Performance

| Problem | Fix |
|---|---|
| Low recall (missing humans) | Increase `EPOCHS` to 120, add WiderPerson dataset |
| Too many false alarms | Raise `CONFIDENCE` to 0.30 |
| Robot jitter | Increase `CENTER_DEADZONE` |
| Slow detection | Reduce `DETECTION_INTERVAL` (GPU dependent) |
| Dark scenes missed | Lower `DARK_THRESHOLD` or `DARK_CONFIDENCE` |

---

## 📁 Project Structure

```
robot/
├── detect.py               # Main detection & control loop
├── prepare_dataset.py      # Dataset builder & YOLOv8 trainer
├── dataset/
│   ├── coco/               # COCO 2017 annotations (auto-downloaded)
│   ├── merged/             # Final merged + augmented dataset
│   └── widerperson.zip     # (optional) Place here manually
└── runs/
    └── rescue/
        └── rescue_finetune/
            └── weights/
                └── best.pt # Fine-tuned model weights
```

---

## 📄 License

Academic / Research use. Built for earthquake disaster rescue competition.
