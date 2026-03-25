"""YOLOv8 keyframe detection; back-project bbox centers to world."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from ultralytics import YOLO


def depth_scale_to_meters_divisor(raw_scale: float) -> float:
    """TUM YAML: depth_m = raw / divisor. Small values (e.g. 0.0002) invert to divisor."""
    s = float(raw_scale)
    return (1.0 / s) if s < 1.0 else s


def focal_xy(focal) -> tuple[float, float]:
    if isinstance(focal, (list, tuple)) and len(focal) >= 2:
        return float(focal[0]), float(focal[1])
    f = float(focal)
    return f, f


def principal_xy(principal) -> tuple[float, float]:
    if isinstance(principal, (list, tuple)) and len(principal) >= 2:
        return float(principal[0]), float(principal[1])
    raise ValueError("principal_point must be [cx, cy]")


def pose_matrix_from_translation_quat(translation, rotation_xyzw) -> np.ndarray:
    """4x4 world-from-camera; quaternion is xyzw (scipy / cuVSLAM)."""
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    q = np.asarray(rotation_xyzw, dtype=np.float64).reshape(4)
    r_mat = Rotation.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = r_mat
    pose[:3, 3] = t
    return pose


class Detector:
    def __init__(
        self,
        model_path: str,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_scale: float,
        conf_threshold: float = 0.5,
    ) -> None:
        self._model = YOLO(model_path)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.depth_scale = float(depth_scale)
        self.conf_threshold = float(conf_threshold)
        self.last_detection_translation: np.ndarray | None = None
        self.keyframe_distance_threshold = 0.15

    def is_keyframe(self, current_translation: np.ndarray) -> bool:
        t = np.asarray(current_translation, dtype=np.float64).reshape(3)
        if self.last_detection_translation is None:
            return True
        return (
            float(np.linalg.norm(t - self.last_detection_translation))
            > self.keyframe_distance_threshold
        )

    def detect(
        self,
        rgb_frame_bgr: np.ndarray,
        depth_frame: np.ndarray,
        pose_matrix: np.ndarray,
        current_translation: np.ndarray,
        frame_idx: int,
    ) -> list[dict]:
        if not self.is_keyframe(current_translation):
            return []

        rgb = rgb_frame_bgr[:, :, ::-1].copy()
        results = self._model(rgb, verbose=False)[0]
        h, w = depth_frame.shape[:2]
        out: list[dict] = []
        names = results.names
        boxes = results.boxes
        if boxes is None:
            self.last_detection_translation = np.asarray(
                current_translation, dtype=np.float64
            ).reshape(3).copy()
            return []

        for box in boxes:
            conf = float(box.conf[0].item())
            if conf < self.conf_threshold:
                continue
            cls = int(box.cls[0].item())
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            u = int(np.clip(round((x1 + x2) / 2.0), 0, w - 1))
            v = int(np.clip(round((y1 + y2) / 2.0), 0, h - 1))

            z = float(depth_frame[v, u]) / self.depth_scale
            if z <= 0.0 or z > 10.0:
                continue

            x = (u - self.cx) / self.fx * z
            y = (v - self.cy) / self.fy * z
            point_cam = np.array([x, y, z, 1.0], dtype=np.float64)
            pw = pose_matrix @ point_cam

            out.append(
                {
                    "label": str(names[cls]),
                    "confidence": conf,
                    "bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
                    "world_xyz": pw[:3].tolist(),
                    "frame_idx": int(frame_idx),
                }
            )

        self.last_detection_translation = np.asarray(
            current_translation, dtype=np.float64
        ).reshape(3).copy()
        return out
