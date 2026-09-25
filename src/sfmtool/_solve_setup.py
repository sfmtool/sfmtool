# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared input and database preparation for the two SfM mappers."""

import textwrap
from pathlib import Path
from typing import NamedTuple

from ._path_summary import summarize_path_list
from ._workspace import load_workspace_config
from .camera.config import CameraConfigResolver
from .colmap.db_setup import _setup_for_sfm, _setup_for_sfm_from_matches
from .rig.config import _load_rig_config


class SolveInputs(NamedTuple):
    db_path: Path
    image_dir: Path
    image_paths: list[str | Path]
    has_rig: bool
    workspace_dir: Path


def prepare_solve_inputs(
    image_paths: list[str | Path],
    workspace_dir: str | Path,
    colmap_dir: str | Path,
    *,
    max_feature_count: int | None,
    camera_model: str | None,
    matching_mode: str,
    flow_preset: str,
    flow_wide_baseline_skip: int,
    matches_file: str | Path | None,
    range_expr: str | None,
) -> SolveInputs:
    """Prepare the database and resolve the images used by either mapper."""
    if matches_file is not None:
        db_path, image_dir, image_paths, has_rig = _setup_for_sfm_from_matches(
            matches_file,
            colmap_dir,
            camera_model=camera_model,
            range_expr=range_expr,
        )
        workspace_dir = Path(image_dir).absolute()
    else:
        print("Image files:")
        print(textwrap.indent(summarize_path_list(image_paths), "  "))
        print(f"Workspace: {workspace_dir}")

        workspace_dir = Path(workspace_dir).absolute()
        config = load_workspace_config(workspace_dir)
        rig_config = _load_rig_config(workspace_dir)
        if rig_config is not None:
            print(f"Rig config: {len(rig_config)} rig(s) detected")

        db_path, image_dir, has_rig = _setup_for_sfm(
            image_paths,
            colmap_dir,
            workspace_dir,
            max_feature_count=max_feature_count,
            feature_tool=config["feature_tool"],
            feature_options=config["feature_options"],
            feature_prefix_dir=config["feature_prefix_dir"],
            rig_config=rig_config,
            camera_model=camera_model,
            matching_mode=matching_mode,
            flow_preset=flow_preset,
            flow_wide_baseline_skip=flow_wide_baseline_skip,
            camera_config_resolver=CameraConfigResolver(workspace_dir),
        )

    return SolveInputs(db_path, image_dir, image_paths, has_rig, workspace_dir)
