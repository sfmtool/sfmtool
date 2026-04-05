# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Incremental Structure from Motion using COLMAP."""

import os
import textwrap
from pathlib import Path

import pycolmap
from deadline.job_attachments.api import summarize_path_list

from ._colmap_db import _setup_for_sfm, _setup_for_sfm_from_matches
from ._colmap_io import (
    build_metadata,
    colmap_binary_to_rust_sfmr,
    pycolmap_to_rust_sfmr,
)
from ._rig_config import _load_rig_config
from ._sfm_reconstruction import get_next_sfm_filename as _get_next_sfm_filename
from ._workspace import load_workspace_config


def run_incremental_sfm(
    image_paths: list[str | Path],
    workspace_dir: str | Path,
    colmap_dir: str | Path,
    max_feature_count: int | None = None,
    sfmr_dir: str | Path | None = None,
    random_seed: int | None = None,
    output_sfm_file: str | None = None,
    refine_rig: bool = True,
    camera_model: str | None = None,
    matching_mode: str = "exhaustive",
    flow_preset: str = "default",
    flow_wide_baseline_skip: int = 5,
    matches_file: str | Path | None = None,
) -> Path:
    """Run incremental Structure from Motion on a list of images."""
    print("Running incremental SfM with COLMAP...")
    if random_seed is not None:
        print(f"Random seed: {random_seed}")
        pycolmap.set_random_seed(random_seed)

    if matches_file is not None:
        db_path, image_dir, image_paths = _setup_for_sfm_from_matches(
            matches_file,
            colmap_dir,
            camera_model=camera_model,
        )
        workspace_dir = Path(image_dir).absolute()
        has_rig = False
    else:
        print("Image files:")
        print(textwrap.indent(summarize_path_list(image_paths), "  "))
        print(f"Workspace: {workspace_dir}")

        workspace_dir = Path(workspace_dir).absolute()

        config = load_workspace_config(workspace_dir)
        feature_tool = config["feature_tool"]
        feature_options = config["feature_options"]
        feature_prefix_dir = config["feature_prefix_dir"]

        rig_config = _load_rig_config(workspace_dir)
        has_rig = rig_config is not None
        if has_rig:
            print(f"Rig config: {len(rig_config)} rig(s) detected")

        db_path, image_dir = _setup_for_sfm(
            image_paths,
            colmap_dir,
            workspace_dir,
            max_feature_count=max_feature_count,
            feature_tool=feature_tool,
            feature_options=feature_options,
            feature_prefix_dir=feature_prefix_dir,
            rig_config=rig_config,
            camera_model=camera_model,
            matching_mode=matching_mode,
            flow_preset=flow_preset,
            flow_wide_baseline_skip=flow_wide_baseline_skip,
        )

    reconstruction_path = os.path.join(colmap_dir, "reconstruction")
    os.makedirs(reconstruction_path, exist_ok=True)

    mapper_options = pycolmap.IncrementalPipelineOptions()
    if random_seed is not None:
        mapper_options.random_seed = random_seed
        pycolmap.set_random_seed(random_seed)
    if has_rig:
        mapper_options.ba_refine_sensor_from_rig = refine_rig

    reconstructions = pycolmap.incremental_mapping(
        db_path, image_dir, reconstruction_path, mapper_options
    )

    if not reconstructions:
        raise RuntimeError("Incremental mapping failed.")

    for idx in reconstructions:
        tool_options = {}
        if max_feature_count is not None:
            tool_options["max_features"] = max_feature_count
        if random_seed is not None:
            tool_options["random_seed"] = random_seed
        if has_rig:
            tool_options["rig_aware"] = True
            tool_options["refine_rig"] = refine_rig

        if output_sfm_file is None:
            if sfmr_dir is None:
                results_base = workspace_dir / "sfmr"
            else:
                results_base = Path(sfmr_dir).absolute()
            output_sfm_file = _get_next_sfm_filename(
                results_base, image_paths, operation="solve"
            )

        output_path = Path(output_sfm_file).absolute()

        if has_rig:
            pycolmap_recon = reconstructions[idx]
            image_count = len(pycolmap_recon.images)
            points3d_count = len(pycolmap_recon.points3D)
            obs_count = sum(
                len(pycolmap_recon.points3D[pid].track.elements)
                for pid in pycolmap_recon.points3D
            )
            camera_count = len(pycolmap_recon.cameras)
        else:
            from ._sfmtool import read_colmap_binary

            recon_dir = Path(reconstruction_path) / str(idx)
            colmap_data = read_colmap_binary(str(recon_dir))
            image_count = len(colmap_data["image_names"])
            points3d_count = len(colmap_data["positions_xyz"])
            obs_count = len(colmap_data["track_image_indexes"])
            camera_count = len(colmap_data["cameras"])

        metadata = build_metadata(
            workspace_dir=workspace_dir,
            output_path=output_path,
            workspace_config=load_workspace_config(workspace_dir),
            operation="sfm_solve",
            tool_name="colmap",
            tool_options=tool_options,
            inputs={
                "images": {
                    "image_dir": str(Path(image_dir).relative_to(workspace_dir)),
                    "image_count": len(image_paths),
                }
            },
            image_count=image_count,
            points3d_count=points3d_count,
            observation_count=obs_count,
            camera_count=camera_count,
        )

        if has_rig:
            recon = pycolmap_to_rust_sfmr(reconstructions[idx], image_dir, metadata)
        else:
            recon_dir = Path(reconstruction_path) / str(idx)
            recon = colmap_binary_to_rust_sfmr(recon_dir, image_dir, metadata)

        print(f"Found reconstruction {idx}:")
        print(
            f"  Cameras: {recon.camera_count}"
            f", Images: {recon.image_count}"
            f", Points: {recon.point_count}"
        )

        recon.save(str(output_path))
        print(f"Saved reconstruction to: {output_sfm_file}")
        return Path(output_sfm_file)
