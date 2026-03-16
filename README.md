# Events Optics End-to-End

A research project combining event-based vision, optical flow estimation, and differentiable optics using neuromorphic simulation. The pipeline generates synthetic optical flow ground truth from NVIDIA Isaac Sim physics simulation, simulates event camera responses via SENPI, and trains a Spike-FlowNet spiking neural network for optical flow prediction. (The Spike-FlowNet neural netowrk does not currently train)

## Overview

### What it does

1. **Synthetic Data Generation** (`isaac_flow_data*.py`) — Uses NVIDIA Isaac Sim / Omniverse to render image sequences of objects with known ground-truth optical flow. Three versions of increasing complexity:
   - V1: basic orbiting object
   - V2: textured box with oscillating camera radius
   - V3: multi-trajectory segments with bumpy surfaces and diverse camera paths

2. **Optical Flow Training** (`isaac_flow_e2e.py`) — Trains Spike-FlowNet on simulated events supervised by GT flow. Uses multi-scale Charbonnier loss, Adam optimizer, and validates with Average End-point Error (AEE). (Does Not Train)
 
3. **Pose Estimation** (`isaac_pose_e2e.py`) — Separate pipeline for camera pose / target tracking from event data. (Does Not Train)

5. **DeepLens Integration** (`e2e_demo.py`) — End-to-end differentiable lens design optimization coupled with the event simulation.

6. **Visualization** (`visualize_isaac_data.py`, `isaac_flow_plot.py`) — Converts `.npy` data to PNG frames or MP4 video; plots training metrics dashboards.

## Project Structure

```
events_optics_e2e/
├── isaac_flow_e2e.py          # Main Spike-FlowNet training script
├── isaac_flow_data.py         # Isaac Sim data generation (V1)
├── isaac_flow_data_v2.py      # Isaac Sim data generation (V2)
├── isaac_flow_data_v3.py      # Isaac Sim data generation (V3)
├── isaac_flow_sequence.py     # PyTorch Dataset for flow sequences
├── isaac_flow_plot.py         # Training metrics visualization
├── isaac_pose_e2e.py          # Pose estimation training
├── isaac_pose_sequence.py     # PyTorch Dataset for pose sequences
├── isaac_pose_plot.py         # Pose training plots
├── e2e_demo.py                # DeepLens + SENPI end-to-end demo
├── visualize_isaac_data.py    # Convert .npy data to PNG/MP4
├── gradient_test.py           # Gradient flow tests for SENPI
├── gradient_test2.py          # Alternative gradient tests
└── plot_e2e.py                # CSV plotting utilities
```

## Setup

### Prerequisites

- Python 3.12+
- NVIDIA Isaac Sim / Omniverse (for data generation scripts)
- [Spike-FlowNet](https://github.com/ruizhao26/Spike-FlowNet) — clone alongside this repo and ensure `models/FlowNetS_spike.py` and `multiscaleloss.py` are importable
- [SENPI](https://github.com/kiwi-sherbet/SENPI) — event camera simulator
- [DeepLens](https://github.com/singer-yang/DeepLens) — differentiable optics library (required for `e2e_demo.py`)

### Python virtual environment

Create and activate a venv, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision numpy matplotlib opencv-python
```

#### Full package list

| Package | Purpose |
|---|---|
| `torch` | PyTorch — neural network training |
| `torchvision` | Image transforms and pretrained models |
| `numpy` | Numerical computing and array I/O |
| `matplotlib` | Plotting training metrics |
| `opencv-python` (`cv2`) | Optical flow visualization, image I/O |
| `senpi` | Event camera simulation (install from source) |
| `deeplens` | Differentiable optics (install from source, required for `e2e_demo.py` only) |
| `isaacsim` / `omni.*` / `pxr` | NVIDIA Isaac Sim / Omniverse (install via Isaac Sim, required for data generation only) |

> **Note**: `isaacsim`, `omni.*`, and `pxr` are provided by the Isaac Sim installation and should not be installed via pip. `senpi` and `deeplens` must be installed from their respective source repositories.
