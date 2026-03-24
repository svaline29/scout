# Scout

**GPU-accelerated semantic spatial mapping with persistent relocalization**

## What it does

Scout is a system that builds **persistent 3D maps** of physical environments using **GPU-accelerated visual SLAM**, then **relocalizes** into those maps from a cold start so the camera pose is recovered against saved structure. The roadmap adds **semantic object detection** anchored to real-world coordinates for scene understanding on top of the map.

## Results

Validated on **TUM RGB-D [freiburg3 long office household](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)**:

| Metric | Value |
|--------|--------|
| **Relocalization translation error** | **5.99 mm** (frame 155) |
| **Map build** | **301 RGB-D pairs** (frame indices **0–300**; `FRAME_LIMIT` in `test_relocalization.py`) |
| **Conditions** | Cold-start relocalization after map save; warm-up tracking on frames 150–155 before localization |

These numbers come from `src/test_relocalization.py` (full pass through 0–300 to build/save the map, then a fresh tracker and relocalize at frame 155).

## Stack

- **NVIDIA cuVSLAM** — GPU-accelerated visual SLAM and odometry  
- **Python 3.12** / **CUDA 12.9**  
- **Rerun** — spatial visualization (trajectory, point cloud)  
- **Open3D** — dense reconstruction (**upcoming**, e.g. TSDF fusion)  
- **YOLOv8** — semantic object detection (**upcoming**)

## Hardware

- **Development:** NVIDIA Tesla T4 (UMN CSE compute cluster)  
- **Target deployment:** NVIDIA Jetson Orin Nano Super  

## Project structure

| Path | Role |
|------|------|
| `src/track_tum.py` | End-to-end TUM RGB-D tracking with **cuVSLAM** and **Rerun** visualization |
| `src/map_manager.py` | Save/load **cuVSLAM** maps under a fixed on-disk layout |
| `src/dataset_utils.py` | Load RGB-D frames and time-aligned pairs for TUM-style datasets |
| `src/test_map_persistence.py` | Short integration run: build map, persist, relocalize |
| `src/test_relocalization.py` | Longer-sequence relocalization test (frames 0–300 map, frame-155 error report) |

## Roadmap

- GPU-accelerated SLAM pipeline  
- 3D landmark visualization  
- Map persistence and relocalization  
- Dense reconstruction (Open3D TSDF)  
- Semantic object detection (YOLOv8)  
- Web-based 3D visualization  
- Live deployment on Jetson Orin  

## Usage

Dataset paths and CUDA/cuVSLAM setup are environment-specific. Run the TUM scripts from the repo root with `PYTHONPATH` including `src`, for example:

```bash
python src/test_relocalization.py
python src/test_map_persistence.py
python src/track_tum.py
```

Ensure the TUM **freiburg3 long office household** dataset is available and that `TUM_DATASET_PATH` (or the path in each script) points to your local copy.
