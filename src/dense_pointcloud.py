"""Freiburg3 RGB-D: run cuVSLAM (0–300), fuse depth into a dense world point cloud."""

import os
import sys

import cuvslam
import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation

# Run as `python src/dense_pointcloud.py` from repo root
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dataset_utils import get_matched_rgbd_pairs, load_frame
from test_map_persistence import (
    RIG_CONFIG_PATH,
    TUM_DATASET_PATH,
    build_tracker,
)

# Frames 0–300 inclusive (same convention as test_relocalization.py)
FRAME_LIMIT = 301
OUTPUT_PLY = "/home/valin019/scout/data/office_dense.ply"
VOXEL_SIZE = 0.02
PROJECT_EVERY = 10
IMAGE_JITTER_THRESHOLD_NS = 40 * 1_000_000


def _focal_xy(focal) -> tuple[float, float]:
    if isinstance(focal, (list, tuple)) and len(focal) >= 2:
        return float(focal[0]), float(focal[1])
    f = float(focal)
    return f, f


def load_pinhole_intrinsic_and_depth_scale(
    rig_yaml_path: str,
) -> tuple[o3d.camera.PinholeCameraIntrinsic, float]:
    with open(rig_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rgb = cfg["rgb_camera"]
    w = int(rgb["image_width"])
    h = int(rgb["image_height"])
    fx, fy = _focal_xy(rgb["focal_length"])
    cx, cy = float(rgb["principal_point"][0]), float(rgb["principal_point"][1])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    depth_scale = float(cfg["depth_camera"]["scale"])
    return intrinsic, depth_scale


def pose_to_world_from_camera_matrix(pose: cuvslam.Pose) -> np.ndarray:
    """4x4 world-from-camera (rig) transform from cuVSLAM translation + quaternion."""
    t = np.asarray(pose.translation, dtype=np.float64).reshape(3)
    q = np.asarray(pose.rotation, dtype=np.float64).reshape(4)
    # scipy: [x, y, z, w]
    r_mat = Rotation.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
    t_world_cam = np.eye(4, dtype=np.float64)
    t_world_cam[:3, :3] = r_mat
    t_world_cam[:3, 3] = t
    return t_world_cam


def main() -> None:
    rgbd_pairs = get_matched_rgbd_pairs(
        TUM_DATASET_PATH, max_time_diff=0.02, max_gap=0.5
    )
    if len(rgbd_pairs) < FRAME_LIMIT:
        raise RuntimeError(
            f"Need at least {FRAME_LIMIT} RGB-D pairs; found {len(rgbd_pairs)}"
        )
    sequence = rgbd_pairs[:FRAME_LIMIT]

    intrinsic, depth_scale = load_pinhole_intrinsic_and_depth_scale(RIG_CONFIG_PATH)
    tracker = build_tracker()

    accumulated = o3d.geometry.PointCloud()
    prev_timestamp = None

    for frame_idx, (rgb_time, rgb_path, depth_path) in enumerate(sequence):
        color_frame = load_frame(rgb_path)
        depth_frame = load_frame(depth_path)
        timestamp = int(rgb_time * 1e9)

        if prev_timestamp is not None:
            if timestamp - prev_timestamp > IMAGE_JITTER_THRESHOLD_NS:
                print(
                    f"Warning: timestamp gap "
                    f"{(timestamp - prev_timestamp) / 1e6:.2f} ms exceeds 40 ms"
                )

        odom_pose_estimate, _ = tracker.track(
            timestamp, images=[color_frame], depths=[depth_frame]
        )
        prev_timestamp = timestamp

        if frame_idx % PROJECT_EVERY != 0:
            continue

        if odom_pose_estimate is None or odom_pose_estimate.world_from_rig is None:
            continue

        pose = odom_pose_estimate.world_from_rig.pose
        world_from_camera = pose_to_world_from_camera_matrix(pose)
        # Open3D multiplies by inv(extrinsic); pass camera_from_world so points land in world frame.
        o3d_extrinsic = np.linalg.inv(world_from_camera)

        # Same convention as create_from_depth_image: depth (meters) = raw / depth_scale.
        # Do not use 1.0/depth_scale here — that blows up depths and depth_trunc clears all points.
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_frame),
            o3d.geometry.Image(depth_frame),
            depth_scale=depth_scale,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False,
        )
        pcd_frame = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsic, o3d_extrinsic
        )
        accumulated += pcd_frame

    n_before = len(accumulated.points)
    print(f"Total points before downsampling: {n_before}")

    down = accumulated.voxel_down_sample(voxel_size=VOXEL_SIZE)
    n_after = len(down.points)
    print(f"Total points after downsampling: {n_after}")

    os.makedirs(os.path.dirname(OUTPUT_PLY), exist_ok=True)
    ok = o3d.io.write_point_cloud(OUTPUT_PLY, down)
    if not ok:
        raise RuntimeError(f"Failed to write point cloud: {OUTPUT_PLY}")
    print(f"Wrote {OUTPUT_PLY}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
