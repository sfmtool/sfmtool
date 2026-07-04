# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""COLMAP database setup for SfM solvers.

The two orchestrators (`_setup_for_sfm`, `_setup_for_sfm_from_matches`) resolve
the camera-config source, dispatch to the per-source builders in
:mod:`db_builders`, and drive feature matching / match writing.
"""

from pathlib import Path

import pycolmap

from ..camera.config import CameraConfigResolver
from ..camrig.resolver import resolve_camrig_for_solve
from ..rig.frames import _build_cross_frame_pairs, _build_same_frame_index_pairs
from ..sift.file import image_files_to_sift_files
from .db_builders import (
    _setup_db_single_camera,
    _setup_db_with_camrig,
    _setup_db_with_rigs,
)


def _setup_for_sfm(
    image_paths: list[str | Path],
    colmap_dir: str | Path,
    workspace_dir: str | Path,
    max_feature_count: int | None = None,
    feature_tool: str | None = None,
    feature_options: dict | None = None,
    feature_prefix_dir: str | None = None,
    rig_config: list[dict] | None = None,
    camera_model: str | None = None,
    matching_mode: str = "exhaustive",
    flow_preset: str = "default",
    flow_wide_baseline_skip: int = 5,
    camera_config_resolver: CameraConfigResolver | None = None,
) -> tuple[Path, Path, bool]:
    """Prepare a COLMAP database for running the mapper.

    Returns:
        tuple: (db_path, image_dir, rig_used) — `rig_used` is True when a
        multi-sensor rig (a multi-sensor `.camrig` or `rig_config.json`) was
        set up, so the caller runs the rig-aware mapper path.
    """
    camrig = resolve_camrig_for_solve(
        image_paths, workspace_dir, camera_model, camera_config_resolver
    )
    camrig_multi = camrig is not None and camrig.is_multi_sensor
    use_rigs = rig_config is not None and camrig is None
    rig_used = use_rigs or camrig_multi
    if rig_config is not None and camrig is not None:
        print(
            "Note: a .camrig covers these images; rig_config.json is ignored "
            "(the .camrig takes precedence)."
        )

    sift_paths = image_files_to_sift_files(
        image_paths,
        feature_tool=feature_tool,
        feature_options=feature_options,
        feature_prefix_dir=feature_prefix_dir,
    )

    colmap_dir = Path(colmap_dir)
    colmap_dir.mkdir(exist_ok=True, parents=True)
    db_path = colmap_dir / "database.db"
    if db_path.exists():
        db_path.unlink()

    image_dir = Path(workspace_dir)

    if camrig_multi:
        _setup_db_with_camrig(
            image_paths,
            sift_paths,
            image_dir,
            db_path,
            max_feature_count,
            camrig.rig,
        )
    elif use_rigs:
        _setup_db_with_rigs(
            image_paths,
            sift_paths,
            image_dir,
            db_path,
            max_feature_count,
            rig_config,
            camera_model=camera_model,
            camera_config_resolver=camera_config_resolver,
        )
    else:
        _setup_db_single_camera(
            image_paths,
            sift_paths,
            image_dir,
            db_path,
            max_feature_count,
            camera_model=camera_model,
            camera_config_resolver=camera_config_resolver,
            camrig_camera=camrig.camera if camrig is not None else None,
        )

    # Build same-frame exclusion data for multi-sensor rigs
    same_frame_index_pairs: set[tuple[int, int]] | None = None
    if rig_used:
        same_frame_index_pairs = _build_same_frame_index_pairs(
            db_path, image_paths, image_dir
        )
        if not same_frame_index_pairs:
            same_frame_index_pairs = None

    # Run feature matching
    if matching_mode == "flow":
        from ..feature_match._run import _run_flow_matching

        _run_flow_matching(
            image_paths,
            sift_paths,
            image_dir,
            db_path,
            colmap_dir,
            max_feature_count=max_feature_count,
            flow_preset=flow_preset,
            flow_wide_baseline_skip=flow_wide_baseline_skip,
        )
    elif matching_mode == "cluster":
        from ..feature_match._run import _run_cluster_matching

        # The background-floor matcher does its own implicit pair selection
        # (only image pairs that share a cluster are verified) and writes the
        # surviving matches + two-view geometry straight into the DB, so it
        # slots in where match_exhaustive would, with the rig already set up.
        # For a multi-sensor rig, exclude same-frame pairs (back-to-back sensors
        # with no shared view) so their spurious cluster matches don't degenerate
        # the solve — the same exclusion the rig-aware exhaustive path applies.
        _run_cluster_matching(
            image_paths,
            sift_paths,
            image_dir,
            db_path,
            colmap_dir,
            max_feature_count=max_feature_count,
            exclude_index_pairs=same_frame_index_pairs,
        )
    elif rig_used and same_frame_index_pairs:
        cross_frame_pairs = _build_cross_frame_pairs(db_path)
        pairs_path = colmap_dir / "match_pairs.txt"
        with open(pairs_path, "w") as f:
            for name_i, name_j in cross_frame_pairs:
                f.write(f"{name_i} {name_j}\n")
        pairing_opts = pycolmap.ImportedPairingOptions()
        pairing_opts.match_list_path = str(pairs_path)
        pycolmap.match_image_pairs(db_path, pairing_options=pairing_opts)
    else:
        pycolmap.match_exhaustive(db_path)

    return db_path, image_dir, rig_used


def _setup_for_sfm_from_matches(
    matches_file: str | Path,
    colmap_dir: str | Path,
    camera_model: str | None = None,
    range_expr: str | None = None,
    camera_config_resolver: CameraConfigResolver | None = None,
) -> tuple[Path, Path, list[Path], bool]:
    """Prepare a COLMAP database from a .matches file for running the mapper.

    If `range_expr` is provided, restrict the solve to images whose filename
    number falls within the range; dropped images are excluded from the DB
    and any pairs that reference them are skipped.

    Returns:
        tuple: (db_path, image_dir, image_paths, rig_used) — `rig_used` is True
        when a multi-sensor rig (a multi-sensor `.camrig` or `rig_config.json`)
        was set up.
    """
    import numpy as np

    from .._filenames import number_from_filename
    from .._sfmtool import RangeExpr
    from .._sfmtool.io import read_matches
    from .._workspace import find_workspace_for_path

    matches_file = Path(matches_file)
    colmap_dir = Path(colmap_dir)

    print(f"Loading matches from: {matches_file}")
    matches_data = read_matches(matches_file)

    metadata = matches_data["metadata"]
    ws_meta = metadata["workspace"]
    all_image_names = matches_data["image_names"]
    full_image_count = metadata["image_count"]

    if range_expr:
        numbers = RangeExpr(range_expr)
        kept_old_indices = [
            i
            for i, name in enumerate(all_image_names)
            if (n := number_from_filename(Path(name).name)) is not None and n in numbers
        ]
        if not kept_old_indices:
            raise RuntimeError(
                f"Range '{range_expr}' excludes all {full_image_count} images "
                f"in {matches_file}"
            )
        image_names = [all_image_names[i] for i in kept_old_indices]
        old_to_new: dict[int, int] | None = {
            old: new for new, old in enumerate(kept_old_indices)
        }
    else:
        image_names = list(all_image_names)
        old_to_new = None

    image_count = len(image_names)

    # Resolve workspace directory
    workspace_dir = None
    matches_dir = matches_file.parent.absolute()
    rel_path = ws_meta.get("relative_path", "")
    if rel_path:
        candidate = (matches_dir / rel_path).resolve()
        ws_marker = candidate / ".sfm-workspace.json"
        if ws_marker.exists():
            workspace_dir = candidate

    if workspace_dir is None:
        abs_path = ws_meta.get("absolute_path", "")
        if abs_path:
            candidate = Path(abs_path)
            if (candidate / ".sfm-workspace.json").exists():
                workspace_dir = candidate

    if workspace_dir is None:
        workspace_dir = find_workspace_for_path(matches_dir)

    if workspace_dir is None:
        raise RuntimeError(
            f"Cannot resolve workspace for {matches_file}. "
            "Ensure the workspace exists and contains .sfm-workspace.json."
        )

    print(f"Workspace: {workspace_dir}")
    if old_to_new is None:
        print(
            f"Images: {image_count}, Pairs: {metadata['image_pair_count']}, "
            f"Matches: {metadata['match_count']}"
        )
    else:
        pairs_arr = matches_data["image_index_pairs"]
        counts_arr = matches_data["match_counts"]
        kept_arr = np.fromiter(old_to_new.keys(), dtype=pairs_arr.dtype)
        pair_mask = np.isin(pairs_arr[:, 0], kept_arr) & np.isin(
            pairs_arr[:, 1], kept_arr
        )
        kept_pairs = int(pair_mask.sum())
        kept_matches = int(counts_arr[pair_mask].sum())
        print(
            f"Images: {image_count} (filtered from {full_image_count} "
            f"via --range {range_expr}), "
            f"Pairs: {kept_pairs} (filtered from {metadata['image_pair_count']}), "
            f"Matches: {kept_matches} (filtered from {metadata['match_count']})"
        )

    # Resolve image paths and .sift paths
    feature_prefix_dir = ws_meta.get("contents", {}).get("feature_prefix_dir", "")
    image_paths = []
    sift_paths = []
    for name in image_names:
        img_path = workspace_dir / name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        image_paths.append(img_path)

        img_parent = Path(name).parent
        img_basename = Path(name).name
        sift_rel = img_parent / feature_prefix_dir / f"{img_basename}.sift"
        sift_path = workspace_dir / sift_rel
        if not sift_path.exists():
            raise FileNotFoundError(f"SIFT file not found: {sift_path}")
        sift_paths.append(sift_path)

    # Set up COLMAP directory and database
    colmap_dir.mkdir(exist_ok=True, parents=True)
    db_path = colmap_dir / "database.db"
    if db_path.exists():
        db_path.unlink()

    image_dir = workspace_dir

    # Check for rig config in the workspace
    from ..rig.config import _load_rig_config

    rig_config = _load_rig_config(workspace_dir)

    # Build a camera_config_resolver if the caller didn't already supply one. Without a
    # camera_config_resolver, intrinsics fall back to EXIF-only behavior.
    if camera_config_resolver is None:
        camera_config_resolver = CameraConfigResolver(workspace_dir)

    from ..camera.setup import _check_camera_model_conflict

    _check_camera_model_conflict(image_paths, camera_config_resolver, camera_model)

    camrig = resolve_camrig_for_solve(
        image_paths, workspace_dir, camera_model, camera_config_resolver
    )
    camrig_multi = camrig is not None and camrig.is_multi_sensor
    use_rigs = rig_config is not None and camrig is None
    rig_used = use_rigs or camrig_multi
    if rig_config is not None and camrig is not None:
        print(
            "Note: a .camrig covers these images; rig_config.json is ignored "
            "(the .camrig takes precedence)."
        )

    # Populate DB with features
    if camrig_multi:
        _setup_db_with_camrig(
            image_paths,
            sift_paths,
            image_dir,
            db_path,
            None,
            camrig.rig,
        )
    elif use_rigs:
        _setup_db_with_rigs(
            image_paths,
            sift_paths,
            image_dir,
            db_path,
            max_feature_count=None,
            rig_configs=rig_config,
            camera_model=camera_model,
            camera_config_resolver=camera_config_resolver,
        )
    else:
        _setup_db_single_camera(
            image_paths,
            sift_paths,
            image_dir,
            db_path,
            max_feature_count=None,
            camera_model=camera_model,
            camera_config_resolver=camera_config_resolver,
            camrig_camera=camrig.camera if camrig is not None else None,
        )

    # Write matches and TVGs to the database
    _write_matches_to_db(
        db_path, matches_data, image_names, image_dir, old_to_new=old_to_new
    )

    return db_path, image_dir, image_paths, rig_used


def _write_matches_to_db(
    db_path: Path,
    matches_data: dict,
    image_names: list[str],
    image_dir: Path,
    old_to_new: dict[int, int] | None = None,
) -> None:
    """Write matches and TVGs from a read_matches dict into a COLMAP database.

    If `old_to_new` is given, it maps original indices in `matches_data` to
    positions in `image_names`; pairs that reference an original index not
    present in the mapping are skipped (their slices of
    `match_feature_indexes` / `inlier_feature_indexes` are stepped over).
    """
    import numpy as np

    with pycolmap.Database.open(db_path) as db:
        db_images = db.read_all_images()
    name_to_db_id = {img.name: img.image_id for img in db_images}

    db_ids = []
    for name in image_names:
        if name not in name_to_db_id:
            raise RuntimeError(f"Image '{name}' not found in COLMAP database")
        db_ids.append(name_to_db_id[name])

    def _resolve(idx: int) -> int | None:
        if old_to_new is None:
            return db_ids[idx]
        new_idx = old_to_new.get(idx)
        return None if new_idx is None else db_ids[new_idx]

    image_index_pairs = matches_data["image_index_pairs"]
    match_counts = matches_data["match_counts"]
    match_feature_indexes = matches_data["match_feature_indexes"]
    pair_count = len(image_index_pairs)
    pairs_written = 0

    with pycolmap.Database.open(db_path) as db:
        match_offset = 0
        for k in range(pair_count):
            idx_i = int(image_index_pairs[k, 0])
            idx_j = int(image_index_pairs[k, 1])
            count = int(match_counts[k])

            db_id_i = _resolve(idx_i)
            db_id_j = _resolve(idx_j)

            if db_id_i is not None and db_id_j is not None:
                matches_slice = match_feature_indexes[
                    match_offset : match_offset + count
                ]
                db.write_matches(db_id_i, db_id_j, matches_slice)
                pairs_written += 1
            match_offset += count

        tvgs_written = 0
        total_inliers_written = 0
        if matches_data.get("has_two_view_geometries", False):
            config_types = matches_data["config_types"]
            config_indexes = matches_data["config_indexes"]
            inlier_counts = matches_data["inlier_counts"]
            inlier_feature_indexes = matches_data["inlier_feature_indexes"]

            CONFIG_STR_TO_INT = {
                "undefined": 0,
                "degenerate": 1,
                "calibrated": 2,
                "uncalibrated": 3,
                "planar": 4,
                "planar_or_panoramic": 5,
                "panoramic": 6,
                "multiple": 7,
                "watermark_clean": 8,
                "watermark_bad": 9,
            }

            inlier_offset = 0
            for k in range(pair_count):
                idx_i = int(image_index_pairs[k, 0])
                idx_j = int(image_index_pairs[k, 1])
                ic = int(inlier_counts[k])

                db_id_i = _resolve(idx_i)
                db_id_j = _resolve(idx_j)

                if db_id_i is None or db_id_j is None:
                    inlier_offset += ic
                    continue

                config_str = config_types[int(config_indexes[k])]
                config_int = CONFIG_STR_TO_INT.get(config_str, 0)

                inlier_slice = inlier_feature_indexes[
                    inlier_offset : inlier_offset + ic
                ]

                tvg = pycolmap.TwoViewGeometry()
                tvg.config = config_int
                tvg.inlier_matches = inlier_slice

                f_mat = matches_data["f_matrices"][k]
                e_mat = matches_data["e_matrices"][k]
                h_mat = matches_data["h_matrices"][k]
                if np.any(f_mat != 0):
                    tvg.F = f_mat
                if np.any(e_mat != 0):
                    tvg.E = e_mat
                if np.any(h_mat != 0):
                    tvg.H = h_mat

                q = matches_data["quaternions_wxyz"][k]
                t = matches_data["translations_xyz"][k]
                is_identity_q = (
                    abs(q[0] - 1.0) < 1e-15
                    and abs(q[1]) < 1e-15
                    and abs(q[2]) < 1e-15
                    and abs(q[3]) < 1e-15
                )
                is_zero_t = all(abs(v) < 1e-15 for v in t)
                if not (is_identity_q and is_zero_t):
                    # `.matches` stores canonical relative poses (cam2_from_cam1);
                    # this DB is a COLMAP-convention artifact written directly via
                    # pycolmap (bypassing the Rust matches<->DB path that would
                    # convert), so S-conjugate to COLMAP here. The stored F/E/H
                    # matrices are pixel-space and stay unchanged.
                    from .convention import relative_pose_conjugate_s

                    q_colmap, t_colmap = relative_pose_conjugate_s(
                        np.asarray(q, dtype=np.float64),
                        np.asarray(t, dtype=np.float64),
                    )
                    quat_xyzw = [
                        q_colmap[1],
                        q_colmap[2],
                        q_colmap[3],
                        q_colmap[0],
                    ]
                    pose = pycolmap.Rigid3d(
                        rotation=pycolmap.Rotation3d(quat_xyzw),
                        translation=t_colmap,
                    )
                    tvg.cam2_from_cam1 = pose

                db.write_two_view_geometry(db_id_i, db_id_j, tvg)
                inlier_offset += ic
                tvgs_written += 1
                total_inliers_written += ic

    print(f"Wrote {pairs_written} match pairs to database")
    if matches_data.get("has_two_view_geometries", False):
        print(
            f"Wrote {tvgs_written} two-view geometries "
            f"({total_inliers_written} total inliers)"
        )
