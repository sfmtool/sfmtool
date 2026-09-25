# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Global Structure from Motion using GLOMAP."""

from pathlib import Path

import pycolmap

from ._incremental_sfm import _save_reconstructions
from ._solve_setup import prepare_solve_inputs
from ._workspace import load_workspace_config


def run_global_sfm(
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
    range_expr: str | None = None,
    detect_infinity: bool = True,
):
    """Run global Structure from Motion on a list of images using GLOMAP."""
    print("Running global SfM with GLOMAP...")
    if random_seed is not None:
        print(f"Random seed: {random_seed}")

    colmap_dir = Path(colmap_dir)
    db_path, image_dir, image_paths, has_rig, workspace_dir = prepare_solve_inputs(
        image_paths,
        workspace_dir,
        colmap_dir,
        max_feature_count=max_feature_count,
        camera_model=camera_model,
        matching_mode=matching_mode,
        flow_preset=flow_preset,
        flow_wide_baseline_skip=flow_wide_baseline_skip,
        matches_file=matches_file,
        range_expr=range_expr,
    )

    reconstruction_path = colmap_dir / "reconstruction"
    reconstruction_path.mkdir(exist_ok=True)

    mapper_options = pycolmap.GlobalPipelineOptions()
    if random_seed is not None:
        mapper_options.random_seed = random_seed
        pycolmap.set_random_seed(random_seed)
    if has_rig:
        mapper_options.mapper.bundle_adjustment.refine_sensor_from_rig = refine_rig

    reconstructions = pycolmap.global_mapping(
        db_path, image_dir, reconstruction_path, mapper_options
    )

    if not reconstructions:
        raise RuntimeError("Global mapping failed or produced no reconstructions.")

    tool_options: dict = {"algorithm": "global"}
    if max_feature_count is not None:
        tool_options["max_features"] = max_feature_count
    if random_seed is not None:
        tool_options["random_seed"] = random_seed
    if has_rig:
        tool_options["rig_aware"] = True
        tool_options["refine_rig"] = refine_rig

    workspace_config = load_workspace_config(workspace_dir)
    explicit_output = Path(output_sfm_file).absolute() if output_sfm_file else None

    return _save_reconstructions(
        reconstructions,
        has_rig=has_rig,
        reconstruction_path=Path(reconstruction_path),
        image_dir=image_dir,
        workspace_dir=workspace_dir,
        workspace_config=workspace_config,
        tool_name="glomap",
        tool_options=tool_options,
        input_image_count=len(image_paths),
        explicit_output=explicit_output,
        sfmr_dir=sfmr_dir,
        detect_infinity=detect_infinity,
    )
