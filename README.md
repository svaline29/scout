# Persistent Visual SLAM with Cold-Start Relocalization

An RGB-D pipeline built around NVIDIA's cuVSLAM. It loads TUM sequences, tracks camera pose, fuses depth into a dense point cloud, and saves the map so a fresh tracker can localize back into it from a cold start.

## What it does

cuVSLAM handles the visual odometry and map storage. The pipeline loads TUM RGB-D sequences with time-aligned color and depth frames, runs them through the tracker, fuses the depth into a dense world point cloud, saves the map to disk, and then measures whether a completely fresh tracker instance can localize itself back into that saved map from a cold start.

The relocalization test builds a map over 301 frames, saves it, then hands a fresh tracker with no history the same scene and asks where it is. It gets within 5.99 mm at frame 155. cuVSLAM is what powers the matching.

## Results

Tested on TUM RGB-D `freiburg3_long_office_household`:

| Metric | Value |
|---|---|
| Relocalization translation error | 5.99 mm at frame 155 |
| Map build | 301 RGB-D pairs (frames 0–300) |
| Dense reconstruction | 265,288 vertices after 2 cm voxel downsampling |
| Conditions | Cold start after map save, warm-up tracking on frames 150–155 |

## Stack

- **cuVSLAM** for GPU-accelerated visual odometry and map persistence
- **Open3D** for dense RGB-D fusion and point cloud output
- **YOLOv8** for object detection, back-projected into world coordinates using the tracked pose
- **Rerun** for live trajectory, landmark, and detection visualization
- Python 3.12, CUDA 12.9

Developed on an NVIDIA Tesla T4 through the UMN CSE compute cluster. 

## Running it
 
Paths and CUDA setup are environment-specific. Set `VSLAM_COLDSTART_SCRATCH` and `VSLAM_COLDSTART_DATASET`, or let them fall back to the defaults under the repo root. From the repo root with `src` on `PYTHONPATH`:
 
```bash
python src/test_relocalization.py
python src/dense_pointcloud.py
python src/track_tum.py
```
 
Requires the TUM `freiburg3_long_office_household` dataset available locally.
