"""Shared paths: outputs under ``<repo>/data`` by default; dataset can live on scratch."""

from __future__ import annotations

import os


def get_repo_root() -> str:
    """Repository root (parent of ``src/``)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_scratch_dir() -> str:
    """
    Root for OUTPUT_RRD, DETECTIONS_JSON, POSES_NPZ, YOLO weights, OUTPUT_MESH, etc.

    Override with env ``VSLAM_COLDSTART_SCRATCH`` (absolute path).
    Default: ``<repo>/data``.
    """
    override = os.environ.get("VSLAM_COLDSTART_SCRATCH")
    if override:
        return os.path.abspath(override)
    return os.path.join(get_repo_root(), "data")


def get_tum_dataset_dir(src_dir: str) -> str:
    """
    TUM RGB-D sequence root (directory containing ``rgb.txt``).

    Resolution order:
    1. ``VSLAM_COLDSTART_DATASET`` if set (absolute path).
    2. ``/export/scratch/<USER>/rgbd_dataset_freiburg3_long_office_household`` if it
       exists and contains ``rgb.txt`` (keeps data off home quota).
    3. ``<repo>/src/dataset/rgbd_dataset_freiburg3_long_office_household`` (default).
    """
    override = os.environ.get("VSLAM_COLDSTART_DATASET")
    if override:
        return os.path.abspath(override)

    user = os.environ.get("USER", "user")
    scratch_ds = os.path.join(
        "/export/scratch",
        user,
        "rgbd_dataset_freiburg3_long_office_household",
    )
    rgb_txt = os.path.join(scratch_ds, "rgb.txt")
    if os.path.isfile(rgb_txt):
        return scratch_ds

    return os.path.join(
        src_dir,
        "dataset",
        "rgbd_dataset_freiburg3_long_office_household",
    )
