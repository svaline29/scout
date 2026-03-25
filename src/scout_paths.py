"""Shared paths: outputs on /export/scratch; dataset can live on scratch to save home quota."""

from __future__ import annotations

import os


def get_scratch_dir() -> str:
    """
    Root for OUTPUT_RRD, DETECTIONS_JSON, POSES_NPZ, YOLO weights, OUTPUT_MESH, etc.

    Override with env ``SCOUT_SCRATCH`` (absolute path).
    Default: ``/export/scratch/<USER>`` (e.g. ``/export/scratch/valin019``).
    """
    override = os.environ.get("SCOUT_SCRATCH")
    if override:
        return os.path.abspath(override)
    user = os.environ.get("USER", "user")
    return os.path.join("/export/scratch", user)


def get_tum_dataset_dir(src_dir: str) -> str:
    """
    TUM RGB-D sequence root (directory containing ``freiburg3_rig.yaml``).

    Resolution order:
    1. ``SCOUT_DATASET`` if set (absolute path).
    2. ``/export/scratch/<USER>/rgbd_dataset_freiburg3_long_office_household`` if it
       exists and contains ``freiburg3_rig.yaml`` (keeps data off home quota).
    3. ``<repo>/src/dataset/rgbd_dataset_freiburg3_long_office_household`` (default).
    """
    override = os.environ.get("SCOUT_DATASET")
    if override:
        return os.path.abspath(override)

    user = os.environ.get("USER", "user")
    scratch_ds = os.path.join(
        "/export/scratch",
        user,
        "rgbd_dataset_freiburg3_long_office_household",
    )
    rig_scratch = os.path.join(scratch_ds, "freiburg3_rig.yaml")
    if os.path.isfile(rig_scratch):
        return scratch_ds

    return os.path.join(
        src_dir,
        "dataset",
        "rgbd_dataset_freiburg3_long_office_household",
    )
