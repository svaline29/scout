"""TUM RGB-D: build map on 0–300, relocalize from mid-sequence (frame 155) after warm-up tracking."""

import os
import sys

import cuvslam
import numpy as np

# Run as `python src/test_relocalization.py` from repo root
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dataset_utils import get_matched_rgbd_pairs, load_frame
from map_manager import MapManager
from test_map_persistence import MAPS_DIR, TUM_DATASET_PATH, build_tracker

# Frames 0–300 inclusive → 301 frames
FRAME_LIMIT = 301
RELOCAL_FRAME = 155
MAP_NAME = "office_relocalization_test"
GROUNDTRUTH_PATH = os.path.join(TUM_DATASET_PATH, "groundtruth.txt")

IMAGE_JITTER_THRESHOLD_NS = 40 * 1_000_000


def _closest_groundtruth_row(
    gt_path: str, t_sec: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (timestamp, translation_xyz, quat_xyzw) for the GT row closest in time."""
    data = np.loadtxt(gt_path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ts = data[:, 0]
    idx = int(np.argmin(np.abs(ts - t_sec)))
    row = data[idx]
    t_gt = float(row[0])
    trans = np.asarray(row[1:4], dtype=np.float64)
    quat = np.asarray(row[4:8], dtype=np.float64)
    return t_gt, trans, quat


def main() -> None:
    rgbd_pairs = get_matched_rgbd_pairs(
        TUM_DATASET_PATH, max_time_diff=0.02, max_gap=0.5
    )
    if len(rgbd_pairs) < FRAME_LIMIT:
        raise RuntimeError(
            f"Need at least {FRAME_LIMIT} RGB-D pairs; found {len(rgbd_pairs)}"
        )
    sequence = rgbd_pairs[:FRAME_LIMIT]

    tracker = build_tracker()
    prev_timestamp = None
    pose_at_155 = None

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
        if frame_idx == RELOCAL_FRAME:
            if (
                odom_pose_estimate is not None
                and odom_pose_estimate.world_from_rig is not None
            ):
                pose_at_155 = odom_pose_estimate.world_from_rig.pose
            else:
                pose_at_155 = None

        prev_timestamp = timestamp

    if pose_at_155 is None:
        raise RuntimeError(
            f"Failed to obtain tracked pose at frame {RELOCAL_FRAME} during mapping"
        )

    t_original = np.asarray(pose_at_155.translation, dtype=np.float64)
    r_original = np.asarray(pose_at_155.rotation, dtype=np.float64)

    map_manager = MapManager(MAPS_DIR)
    saved = map_manager.save(
        tracker,
        MAP_NAME,
        {"dataset": "freiburg3", "frame_count": FRAME_LIMIT},
    )
    print(f"Map save succeeded: {saved}")

    # Fresh tracker: warm up with frames 150–155 (no map save)
    tracker2 = build_tracker()
    prev_timestamp = None
    rgb_time155 = None
    color_155 = None

    for frame_idx in range(150, 156):
        rgb_time, rgb_path, depth_path = rgbd_pairs[frame_idx]
        color_frame = load_frame(rgb_path)
        depth_frame = load_frame(depth_path)
        timestamp = int(rgb_time * 1e9)

        if prev_timestamp is not None:
            if timestamp - prev_timestamp > IMAGE_JITTER_THRESHOLD_NS:
                print(
                    f"Warning: timestamp gap "
                    f"{(timestamp - prev_timestamp) / 1e6:.2f} ms exceeds 40 ms"
                )

        tracker2.track(timestamp, images=[color_frame], depths=[depth_frame])
        prev_timestamp = timestamp

        if frame_idx == RELOCAL_FRAME:
            rgb_time155 = rgb_time
            color_155 = color_frame

    if color_155 is None or rgb_time155 is None:
        raise RuntimeError("Failed to load frame 155 for relocalization")

    loc_settings = cuvslam.Tracker.SlamLocalizationSettings(
        horizontal_search_radius=8.0,
        vertical_search_radius=2.0,
        horizontal_step=0.5,
        vertical_step=0.2,
        angular_step_rads=0.03,
    )
    relocalized = map_manager.load(
        tracker2, MAP_NAME, [color_155], loc_settings
    )
    pose = map_manager.last_localization_pose

    print(f"Relocalization succeeded: {relocalized}")

    t_relocalized: np.ndarray | None = None
    r_relocalized: np.ndarray | None = None
    if pose is not None:
        t_relocalized = np.asarray(pose.translation, dtype=np.float64)
        r_relocalized = np.asarray(pose.rotation, dtype=np.float64)
        print(f"Relocalized pose translation: {t_relocalized.tolist()}")
        print(f"Relocalized pose rotation (x,y,z,w): {r_relocalized.tolist()}")
    else:
        print("Relocalized pose: None")

    print("")
    print("--- Comparison: original tracked pose (mapping) vs relocalized ---")
    print(
        f"Original tracked translation (frame {RELOCAL_FRAME}): {t_original.tolist()}"
    )
    print(f"Original tracked rotation (x,y,z,w): {r_original.tolist()}")
    if t_relocalized is not None and r_relocalized is not None:
        print(f"Relocalized translation: {t_relocalized.tolist()}")
        print(f"Relocalized rotation (x,y,z,w): {r_relocalized.tolist()}")

    gt_t, gt_trans, gt_quat = _closest_groundtruth_row(
        GROUNDTRUTH_PATH, rgb_time155
    )
    print("")
    print("--- Ground truth (closest timestamp to frame 155 RGB time) ---")
    print(f"Frame {RELOCAL_FRAME} RGB time (s): {rgb_time155:.6f}")
    print(f"Closest GT timestamp (s): {gt_t:.6f} (delta {gt_t - rgb_time155:+.6f} s)")
    print(f"GT translation (tx, ty, tz): {gt_trans.tolist()}")
    print(f"GT rotation (qx, qy, qz, qw): {gt_quat.tolist()}")

    if t_relocalized is not None:
        dist = float(np.linalg.norm(t_relocalized - t_original))
        print("")
        print(
            f"Euclidean distance ||t_relocalized - t_original||: {dist:.6f} m"
        )


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
