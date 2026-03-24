"""Short TUM RGB-D run: build map, save, relocalize from the first frame."""

import os
import sys

import cuvslam
import numpy as np
import yaml

# Run as `python src/test_map_persistence.py` from repo root
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dataset_utils import get_matched_rgbd_pairs, load_frame
from map_manager import MapManager

TUM_DATASET_PATH = (
    "/home/valin019/cuVSLAM/examples/tum/dataset/"
    "rgbd_dataset_freiburg3_long_office_household"
)
RIG_CONFIG_PATH = (
    "/home/valin019/cuVSLAM/examples/tum/dataset/"
    "rgbd_dataset_freiburg3_long_office_household/freiburg3_rig.yaml"
)
MAPS_DIR = "/home/valin019/scout/data/maps"
FRAME_LIMIT = 200


def build_tracker() -> cuvslam.Tracker:
    """Same odometry/rig setup as track_tum.py (RGBD, freiburg3), plus SLAM for map I/O."""
    with open(RIG_CONFIG_PATH, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    camera = cuvslam.Camera()
    camera.size = (
        config_data["rgb_camera"]["image_width"],
        config_data["rgb_camera"]["image_height"],
    )
    camera.principal = config_data["rgb_camera"]["principal_point"]
    camera.focal = config_data["rgb_camera"]["focal_length"]
    camera.border_top = 20
    camera.border_bottom = 20
    camera.border_left = 10
    camera.border_right = 50

    rgbd_settings = cuvslam.Tracker.OdometryRGBDSettings()
    rgbd_settings.depth_scale_factor = config_data["depth_camera"]["scale"]
    rgbd_settings.depth_camera_id = 0
    rgbd_settings.enable_depth_stereo_tracking = False

    cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=True,
        enable_final_landmarks_export=True,
        enable_landmarks_export=True,
        odometry_mode=cuvslam.Tracker.OdometryMode.RGBD,
        rgbd_settings=rgbd_settings,
    )

    # Required for save_map / localize_in_map (track_tum.py omits SLAM)
    slam_cfg = cuvslam.Tracker.SlamConfig()
    slam_cfg.sync_mode = True

    return cuvslam.Tracker(cuvslam.Rig([camera]), cfg, slam_cfg)


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
    image_jitter_threshold_ns = 40 * 1_000_000

    for rgb_time, rgb_path, depth_path in sequence:
        color_frame = load_frame(rgb_path)
        depth_frame = load_frame(depth_path)
        timestamp = int(rgb_time * 1e9)

        if prev_timestamp is not None:
            if timestamp - prev_timestamp > image_jitter_threshold_ns:
                print(
                    f"Warning: timestamp gap "
                    f"{(timestamp - prev_timestamp) / 1e6:.2f} ms exceeds 40 ms"
                )

        tracker.track(timestamp, images=[color_frame], depths=[depth_frame])
        prev_timestamp = timestamp

    map_manager = MapManager(MAPS_DIR)
    saved = map_manager.save(
        tracker,
        "office_test",
        {"dataset": "freiburg3", "frame_count": FRAME_LIMIT},
    )
    print(f"Map save succeeded: {saved}")

    # Relocalization: fresh tracker, first frame of the dataset
    rgb_time0, rgb_path0, depth_path0 = rgbd_pairs[0]
    color0 = load_frame(rgb_path0)
    depth0 = load_frame(depth_path0)
    ts0 = int(rgb_time0 * 1e9)

    tracker2 = build_tracker()
    tracker2.track(ts0, images=[color0], depths=[depth0])

    loc_settings = cuvslam.Tracker.SlamLocalizationSettings(
        horizontal_search_radius=8.0,
        vertical_search_radius=2.0,
        horizontal_step=0.5,
        vertical_step=0.2,
        angular_step_rads=0.03,
    )
    relocalized = map_manager.load(
        tracker2, "office_test", [color0], loc_settings
    )
    pose = map_manager.last_localization_pose

    print(f"Relocalization succeeded: {relocalized}")
    if pose is not None:
        t = np.asarray(pose.translation)
        r = np.asarray(pose.rotation)
        print(f"Pose translation: {t.tolist()}")
        print(f"Pose rotation (x,y,z,w): {r.tolist()}")
    else:
        print("Pose: None")


if __name__ == "__main__":
    main()
    # Normal process teardown runs nanobind's exit-time leak checker, which is
    # noisy for cuVSLAM (callbacks + globals). Skip finalization after success
    # so the script exits quietly; failures still use the normal error path.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
