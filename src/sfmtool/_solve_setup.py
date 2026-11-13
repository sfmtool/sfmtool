# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared input and database preparation for the two SfM mappers."""

import textwrap
from pathlib import Path
from typing import NamedTuple

from ._path_summary import summarize_path_list
from ._sfmtool.fileio import read_matches_metadata
from ._workspace import load_workspace_config
from .camera.config import CameraConfigResolver
from .colmap.db_setup import (
    MatchesImageFilter,
    _setup_for_sfm,
    _setup_for_sfm_from_matches,
)


class SolveInputs(NamedTuple):
    db_path: Path
    image_dir: Path
    image_paths: list[str | Path]
    has_rig: bool
    workspace_dir: Path
    features: dict


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
    image_filter: MatchesImageFilter | None,
) -> SolveInputs:
    """Prepare the database and resolve the images used by either mapper.

    `features` in the result is the feature definition the database's
    `.sift` files come from, as a workspace config holds it
    (`feature_tool`, `feature_options`, `feature_prefix_dir`): the
    `.matches` file's own when solving from one, which is not the
    workspace's current one after `sfm ws upgrade`, and the workspace's
    otherwise. The reconstruction records it, and reads its `.sift` files
    from its folder.
    """
    if matches_file is not None:
        db_path, image_dir, image_paths, has_rig = _setup_for_sfm_from_matches(
            matches_file,
            colmap_dir,
            camera_model=camera_model,
            image_filter=image_filter,
        )
        workspace_dir = Path(image_dir).absolute()
        features = dict(
            read_matches_metadata(str(matches_file))["workspace"]["contents"]
        )
    else:
        print("Image files:")
        print(textwrap.indent(summarize_path_list(image_paths), "  "))
        print(f"Workspace: {workspace_dir}")

        workspace_dir = Path(workspace_dir).absolute()
        config = load_workspace_config(workspace_dir)
        features = config

        db_path, image_dir, has_rig = _setup_for_sfm(
            image_paths,
            colmap_dir,
            workspace_dir,
            max_feature_count=max_feature_count,
            feature_tool=config["feature_tool"],
            feature_options=config["feature_options"],
            feature_prefix_dir=config["feature_prefix_dir"],
            camera_model=camera_model,
            matching_mode=matching_mode,
            flow_preset=flow_preset,
            flow_wide_baseline_skip=flow_wide_baseline_skip,
            camera_config_resolver=CameraConfigResolver(workspace_dir),
        )

    return SolveInputs(
        db_path, image_dir, image_paths, has_rig, workspace_dir, features
    )
