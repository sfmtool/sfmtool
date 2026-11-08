# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""COLMAP database setup for SfM solvers.

The two orchestrators (`_setup_for_sfm`, `_setup_for_sfm_from_matches`) resolve
the camera-config source, dispatch to the per-source builders in
:mod:`db_builders`, and drive feature matching / match writing.
"""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pycolmap

from ..camera.config import CameraConfigResolver
from ..camrig.resolver import resolve_camrig_for_solve
from ..rig.frames import _build_cross_frame_pairs, _build_same_frame_index_pairs
from ..sift.extract import image_files_to_sift_files
from .db_builders import (
    _setup_db_single_sensor,
    _setup_db_with_camrig,
    _setup_db_with_rigs,
)


class MatchesImageFilter(NamedTuple):
    """Which images of a `.matches` file a solve keeps.

    `keep(workspace_dir, image_names)` gets the workspace the file resolves to
    and the file's image names, and returns the indices of the names to keep,
    in the file's order; it raises when it keeps none. `label` names the
    filter in the summary line, such as `--range 5-12`.
    """

    label: str
    keep: Callable[[Path, list[str]], list[int]]


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
        settings_source=f"The workspace file {Path(workspace_dir) / '.sfm-workspace.json'}",
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
        _setup_db_single_sensor(
            image_paths,
            sift_paths,
            image_dir,
            db_path,
            max_feature_count,
            camera_model=camera_model,
            camera_config_resolver=camera_config_resolver,
            camrig_cameras=camrig.cameras if camrig is not None else None,
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


def resolve_image_and_sift_paths(
    workspace_dir: Path,
    image_names: list[str],
    feature_prefix_dir: str,
) -> tuple[list[Path], list[Path]]:
    """Resolve workspace-relative image names to image and `.sift` paths.

    The `.sift` file of `sub/dir/img.jpg` lives at
    `sub/dir/<feature_prefix_dir>/img.jpg.sift`, which is the layout
    `image_files_to_sift_files` writes; this is the read-side counterpart that
    resolves names recorded in a `.matches` file without consulting the
    workspace config. Raises `FileNotFoundError` naming the first file that
    does not exist.
    """
    image_paths: list[Path] = []
    sift_paths: list[Path] = []
    for name in image_names:
        img_path = workspace_dir / name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        image_paths.append(img_path)

        rel = Path(name)
        sift_path = workspace_dir / rel.parent / feature_prefix_dir / f"{rel.name}.sift"
        if not sift_path.exists():
            raise FileNotFoundError(f"SIFT file not found: {sift_path}")
        sift_paths.append(sift_path)
    return image_paths, sift_paths


def _setup_for_sfm_from_matches(
    matches_file: str | Path,
    colmap_dir: str | Path,
    camera_model: str | None = None,
    image_filter: MatchesImageFilter | None = None,
    camera_config_resolver: CameraConfigResolver | None = None,
    matches_data: dict | None = None,
) -> tuple[Path, Path, list[Path], bool]:
    """Prepare a COLMAP database from a .matches file for running the mapper.

    If `image_filter` is given, restrict the solve to the images it keeps;
    dropped images are excluded from the DB and any pairs that reference them
    are skipped.

    `matches_data` lets a caller that has already parsed the file hand the
    dict over instead of paying for a second parse; it must be what
    `read_matches(matches_file)` returned, including its `workspace_dir`.

    Returns:
        tuple: (db_path, image_dir, image_paths, rig_used) — `rig_used` is True
        when a multi-sensor rig (a multi-sensor `.camrig` or `rig_config.json`)
        was set up.
    """
    import numpy as np

    from .._sfmtool.fileio import read_matches
    from ..feature_match import pairs_from_matches

    matches_file = Path(matches_file)
    colmap_dir = Path(colmap_dir)

    if matches_data is None:
        print(f"Loading matches from: {matches_file}")
        matches_data = read_matches(matches_file)
    # The single pairwise view: stored pairs verbatim, or the canonical
    # cluster expansion for cluster-bearing files (descriptor distances are
    # not needed to populate the DB).
    pairs_data = pairs_from_matches(matches_data)

    metadata = matches_data["metadata"]
    ws_meta = metadata["workspace"]
    all_image_names = matches_data["image_names"]
    full_image_count = metadata["image_count"]

    if matches_data["workspace_dir"] is None:
        raise RuntimeError(
            f"Cannot resolve workspace for {matches_file}. "
            "Ensure the workspace exists and contains .sfm-workspace.json."
        )
    workspace_dir = Path(matches_data["workspace_dir"])

    if image_filter is not None:
        kept_old_indices = image_filter.keep(workspace_dir, list(all_image_names))
        image_names = [all_image_names[i] for i in kept_old_indices]
        old_to_new: dict[int, int] | None = {
            old: new for new, old in enumerate(kept_old_indices)
        }
    else:
        image_names = list(all_image_names)
        old_to_new = None

    image_count = len(image_names)

    total_pairs = len(pairs_data["image_index_pairs"])
    total_matches = len(pairs_data["match_feature_indexes"])
    pairs_label = "Pairs (derived)" if matches_data.get("has_clusters") else "Pairs"
    print(f"Workspace: {workspace_dir}")
    if old_to_new is None:
        print(
            f"Images: {image_count}, {pairs_label}: {total_pairs}, "
            f"Matches: {total_matches}"
        )
    else:
        pairs_arr = pairs_data["image_index_pairs"]
        counts_arr = pairs_data["match_counts"]
        kept_arr = np.fromiter(old_to_new.keys(), dtype=pairs_arr.dtype)
        pair_mask = np.isin(pairs_arr[:, 0], kept_arr) & np.isin(
            pairs_arr[:, 1], kept_arr
        )
        kept_pairs = int(pair_mask.sum())
        kept_matches = int(counts_arr[pair_mask].sum())
        print(
            f"Images: {image_count} (filtered from {full_image_count} "
            f"via {image_filter.label}), "
            f"{pairs_label}: {kept_pairs} (filtered from {total_pairs}), "
            f"Matches: {kept_matches} (filtered from {total_matches})"
        )

    # Resolve image paths and .sift paths
    feature_prefix_dir = ws_meta.get("contents", {}).get("feature_prefix_dir", "")
    image_paths, sift_paths = resolve_image_and_sift_paths(
        workspace_dir, image_names, feature_prefix_dir
    )

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
        _setup_db_single_sensor(
            image_paths,
            sift_paths,
            image_dir,
            db_path,
            max_feature_count=None,
            camera_model=camera_model,
            camera_config_resolver=camera_config_resolver,
            camrig_cameras=camrig.cameras if camrig is not None else None,
        )

    # Write matches and TVGs to the database
    _write_matches_to_db(
        db_path,
        matches_data,
        image_names,
        image_dir,
        old_to_new=old_to_new,
        pairs_data=pairs_data,
    )

    return db_path, image_dir, image_paths, rig_used


def _write_matches_to_db(
    db_path: Path,
    matches_data: dict,
    image_names: list[str],
    image_dir: Path,
    old_to_new: dict[int, int] | None = None,
    pairs_data: dict | None = None,
) -> None:
    """Write matches and TVGs from a read_matches dict into a COLMAP database.

    Pairwise arrays come from `pairs_data` (a `pairs_from_matches` view,
    derived here when not supplied), so cluster-bearing files work
    transparently; TVGs only exist for stored-pairs files, and their arrays
    are keyed per stored pair.

    If `old_to_new` is given, it maps original indices in `matches_data` to
    positions in `image_names`; pairs that reference an original index not
    present in the mapping are skipped (their slices of
    `match_feature_indexes` / `inlier_feature_indexes` are stepped over).
    """
    import numpy as np

    from ..feature_match import pairs_from_matches

    if pairs_data is None:
        pairs_data = pairs_from_matches(matches_data)

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

    image_index_pairs = pairs_data["image_index_pairs"]
    match_counts = pairs_data["match_counts"]
    match_feature_indexes = pairs_data["match_feature_indexes"]
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
