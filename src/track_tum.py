import os

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import cuvslam

from dataset_utils import load_frame, get_matched_rgbd_pairs
from detection_store import DetectionStore
from detector import Detector, pose_matrix_from_translation_quat
from scout_paths import get_scratch_dir, get_tum_dataset_dir

# TODO: move TUM freiburg3 intrinsics to config/freiburg3.yaml (hardcoded; no rig YAML on disk).
# TUM freiburg3 calibration (official TUM values)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FX, FY = 535.4, 539.2
CX, CY = 320.1, 247.6
DEPTH_SCALE = 5000.0

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRATCH_DIR = get_scratch_dir()
tum_dataset_path = get_tum_dataset_dir(_SRC_DIR)

OUTPUT_RRD = os.path.join(_SCRATCH_DIR, "tum_output.rrd")
DETECTIONS_JSON = os.path.join(_SCRATCH_DIR, "scout_detections.json")
POSES_NPZ = os.path.join(_SCRATCH_DIR, "scout_poses.npz")
YOLO_MODEL = os.path.join(_SCRATCH_DIR, "yolov8n.pt")


def color_from_id(identifier):
    """Generate a color from an identifier."""
    return [
        (identifier * 17) % 256,
        (identifier * 31) % 256,
        (identifier * 47) % 256,
    ]


IMAGE_JITTER_THRESHOLD_NS = 40 * 1e6  # 40ms in nanoseconds

rr.init("tum_dataset", strict=True)
rr.save(OUTPUT_RRD)
rr.send_blueprint(
    rrb.Blueprint(
        rrb.TimePanel(state="collapsed"),
        rrb.Horizontal(
            contents=[
                rrb.Vertical(
                    contents=[
                        rrb.Spatial2DView(
                            origin="world/camera/image", name="RGB Camera"
                        ),
                        rrb.Spatial2DView(
                            origin="world/camera/depth", name="Depth Camera"
                        ),
                    ]
                ),
                rrb.Spatial3DView(
                    name="3D",
                    defaults=[rr.components.ImagePlaneDistance(0.5)],
                ),
            ]
        ),
    ),
    make_active=True,
)

rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

rgbd_pairs = get_matched_rgbd_pairs(
    tum_dataset_path, max_time_diff=0.02, max_gap=0.5
)
print(f"Found {len(rgbd_pairs)} matched RGB-D pairs (dataset: {tum_dataset_path})")

camera = cuvslam.Camera()
camera.size = (IMAGE_WIDTH, IMAGE_HEIGHT)
camera.principal = [CX, CY]
camera.focal = [FX, FY]
camera.border_top = 20
camera.border_bottom = 20
camera.border_left = 10
camera.border_right = 50

rgbd_settings = cuvslam.Tracker.OdometryRGBDSettings()
rgbd_settings.depth_scale_factor = DEPTH_SCALE
rgbd_settings.depth_camera_id = 0
rgbd_settings.enable_depth_stereo_tracking = False

cfg = cuvslam.Tracker.OdometryConfig(
    async_sba=True,
    enable_final_landmarks_export=True,
    enable_landmarks_export=True,
    odometry_mode=cuvslam.Tracker.OdometryMode.RGBD,
    rgbd_settings=rgbd_settings,
)

tracker = cuvslam.Tracker(cuvslam.Rig([camera]), cfg)

detector = Detector(YOLO_MODEL, FX, FY, CX, CY, DEPTH_SCALE)
detection_store = DetectionStore()

frame_id = 0
prev_timestamp = None
trajectory = []
pose_frame_ids: list[int] = []
poses_trans: list[np.ndarray] = []
poses_rot: list[np.ndarray] = []

for rgb_time, rgb_path, depth_path in rgbd_pairs:
    try:
        color_frame = load_frame(rgb_path)
        depth_frame = load_frame(depth_path)
    except Exception as e:
        print(f"Warning: Failed to read image files: {rgb_path} or {depth_path}: {e}")
        continue

    timestamp = int(rgb_time * 1e9)

    if prev_timestamp is not None:
        timestamp_diff = timestamp - prev_timestamp
        if timestamp_diff > IMAGE_JITTER_THRESHOLD_NS:
            print(
                f"Warning: Camera stream message delayed: timestamp gap "
                f"({timestamp_diff / 1e6:.2f} ms) exceeds threshold "
                f"{IMAGE_JITTER_THRESHOLD_NS / 1e6:.2f} ms"
            )

    odom_pose_estimate, _ = tracker.track(
        timestamp, images=[color_frame], depths=[depth_frame]
    )

    if odom_pose_estimate.world_from_rig is None:
        print(f"Warning: Failed to track frame {frame_id}")
        continue

    odom_pose = odom_pose_estimate.world_from_rig.pose
    trajectory.append(odom_pose.translation)

    trans = np.asarray(odom_pose.translation, dtype=np.float64).reshape(3)
    rot = np.asarray(odom_pose.rotation, dtype=np.float64).reshape(4)
    pose_mat = pose_matrix_from_translation_quat(odom_pose.translation, odom_pose.rotation)
    dets = detector.detect(
        color_frame, depth_frame, pose_mat, trans, frame_id
    )
    detection_store.add(dets)
    pose_frame_ids.append(frame_id)
    poses_trans.append(trans.copy())
    poses_rot.append(rot.copy())

    if dets:
        xyz = [d["world_xyz"] for d in dets]
        rr.log(
            "world/detections",
            rr.Points3D(xyz, radii=0.05),
        )

    observations = [tracker.get_last_observations(0)]
    obs_uv = [[o.u, o.v] for o in observations[0]]
    obs_colors = [color_from_id(o.id) for o in observations[0]]

    rr.set_time_sequence("frame", frame_id)
    rr.log("trajectory", rr.LineStrips3D(trajectory), static=True)

    rr.log(
        "world/camera/rgb",
        rr.Transform3D(
            translation=odom_pose.translation, quaternion=odom_pose.rotation
        ),
        rr.Arrows3D(
            vectors=np.eye(3) * 0.2,
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )

    rr.log("world/camera/image", rr.Image(color_frame).compress(jpeg_quality=80))
    rr.log(
        "world/camera/image/observations",
        rr.Points2D(obs_uv, radii=5, colors=obs_colors),
    )
    rr.log("world/camera/depth", rr.Image(depth_frame))
    rr.log(
        "world/camera/depth/observations",
        rr.Points2D(obs_uv, radii=5, colors=obs_colors),
    )

    landmarks = tracker.get_last_landmarks()
    if landmarks:
        lm_coords = [l.coords for l in landmarks]
        lm_colors = [color_from_id(l.id) for l in landmarks]
        rr.log(
            "world/landmarks",
            rr.Points3D(lm_coords, colors=lm_colors, radii=0.02),
        )

    frame_id += 1
    prev_timestamp = timestamp

detection_store.save(DETECTIONS_JSON)
if poses_trans:
    np.savez(
        POSES_NPZ,
        frame_id=np.array(pose_frame_ids, dtype=np.int32),
        translation=np.stack(poses_trans),
        rotation=np.stack(poses_rot),
    )
print(f"Wrote {DETECTIONS_JSON} and {POSES_NPZ}")
