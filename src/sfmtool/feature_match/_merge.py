# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Merging of several `.matches` files into one, for `sfm match --merge`.

Split out of `_run.py`, which owns the "run a matching job" concern: the two
share no state and no imports, and only ever meet at the `_commands/match.py`
dispatch.
"""

import os
from pathlib import Path

import click
import numpy as np


def _run_merge(paths, output_path):
    """Merge multiple .matches files into a single file.

    Builds a unified image list, remaps pair indexes, concatenates matches
    for each pair, and deduplicates by feature index pair (keeping the match
    with the lowest descriptor distance). Two-view geometry is preserved: for
    each pair the best available TVG is carried into the output, and the
    output's has_two_view_geometries reflects whether any survived.
    """
    from datetime import datetime

    from .._sfmtool.io import read_matches, write_matches

    if not paths:
        raise click.UsageError("Must provide .matches file paths.")
    if len(paths) < 2:
        raise click.UsageError("Need at least 2 .matches files to merge.")
    if not output_path:
        raise click.UsageError("--output / -o is required for --merge.")

    input_paths = []
    for p in paths:
        path = Path(p)
        if path.suffix.lower() != ".matches":
            raise click.UsageError(f"Expected a .matches file: {p}")
        input_paths.append(path)

    out = Path(output_path)
    if out.suffix.lower() != ".matches":
        raise click.UsageError(f"Output path must be a .matches file: {output_path}")

    # Read all input files. Pairwise arrays come through the derived-pairs
    # helper, so cluster-bearing inputs merge like any other (their expanded
    # matches carry NaN descriptor distances).
    from ._pairs import pairs_from_matches

    click.echo(f"Merging {len(input_paths)} .matches files...")
    all_data = []
    for path in input_paths:
        click.echo(f"  Reading {path.name}...")
        data = read_matches(str(path))
        pairs_view = pairs_from_matches(data)
        click.echo(
            f"    {data['metadata']['image_count']} images, "
            f"{len(pairs_view['image_index_pairs'])} pairs, "
            f"{len(pairs_view['match_feature_indexes'])} matches"
        )
        all_data.append((data, pairs_view))

    # Build unified image list and validate consistency
    image_info = {}  # name -> {feature_count, tool_hash, content_hash, dims}
    for path, (data, _pairs_view) in zip(input_paths, all_data):
        names = list(data["image_names"])
        # Mandatory since .matches format version 4; a pre-version-4 input
        # (loaded without dims) cannot produce a valid merged file.
        if "image_dims" not in data:
            raise click.ClickException(
                f"{path.name} predates per-image dimensions (.matches format "
                "version 4); re-run matching to regenerate it before merging."
            )
        for i, name in enumerate(names):
            tool_hash = bytes(data["feature_tool_hashes"][i])
            content_hash = bytes(data["sift_content_hashes"][i])
            feature_count = int(data["feature_counts"][i])
            dims = tuple(int(v) for v in data["image_dims"][i])
            if name in image_info:
                existing = image_info[name]
                if existing["content_hash"] != content_hash:
                    raise click.ClickException(
                        f"Image '{name}' has different SIFT content hashes "
                        f"across input files — cannot merge matches from "
                        f"different feature extractions."
                    )
            else:
                image_info[name] = {
                    "feature_count": feature_count,
                    "tool_hash": tool_hash,
                    "content_hash": content_hash,
                    "dims": dims,
                }

    unified_names = sorted(image_info.keys())
    name_to_idx = {name: i for i, name in enumerate(unified_names)}
    image_count = len(unified_names)
    click.echo(f"  Unified image count: {image_count}")

    # Collect all matches per pair, remapping indexes
    # pair_key = (idx_i, idx_j) with idx_i < idx_j in the unified list
    pair_matches = {}  # (idx_i, idx_j) -> {(feat_i, feat_j): distance}
    pair_tvg = {}  # (idx_i, idx_j) -> tvg_info dict (best TVG for this pair)

    for data, pairs_view in all_data:
        names = list(data["image_names"])
        local_to_unified = [name_to_idx[n] for n in names]
        has_tvg = data.get("has_two_view_geometries", False)

        pairs = pairs_view["image_index_pairs"]
        counts = pairs_view["match_counts"]
        feat_idxs = pairs_view["match_feature_indexes"]
        distances = pairs_view["match_descriptor_distances"]

        if has_tvg:
            tvg_config_types = data["config_types"]
            tvg_config_indexes = data["config_indexes"]
            tvg_inlier_counts = data["inlier_counts"]
            tvg_inlier_feat_idxs = data["inlier_feature_indexes"]
            tvg_f_matrices = data["f_matrices"]
            tvg_e_matrices = data["e_matrices"]
            tvg_h_matrices = data["h_matrices"]
            tvg_quaternions = data["quaternions_wxyz"]
            tvg_translations = data["translations_xyz"]

        offset = 0
        inlier_offset = 0
        for k in range(len(counts)):
            local_i = int(pairs[k, 0])
            local_j = int(pairs[k, 1])
            uni_i = local_to_unified[local_i]
            uni_j = local_to_unified[local_j]
            swap = uni_i > uni_j
            if swap:
                uni_i, uni_j = uni_j, uni_i

            count = int(counts[k])
            key = (uni_i, uni_j)
            if key not in pair_matches:
                pair_matches[key] = {}

            for m in range(offset, offset + count):
                fi = int(feat_idxs[m, 0])
                fj = int(feat_idxs[m, 1])
                dist = float(distances[m])
                if swap:
                    fi, fj = fj, fi
                feat_key = (fi, fj)
                # Keep the match with the lowest descriptor distance; a known
                # distance beats NaN (cluster-derived matches carry NaN).
                prev = pair_matches[key].get(feat_key)
                if prev is None or (
                    not np.isnan(dist) and (np.isnan(prev) or dist < prev)
                ):
                    pair_matches[key][feat_key] = dist

            offset += count

            # Collect TVG for this pair (keep the one with the most inliers)
            if has_tvg:
                ic = int(tvg_inlier_counts[k])
                existing = pair_tvg.get(key)
                if existing is None or ic > existing["inlier_count"]:
                    config_str = tvg_config_types[int(tvg_config_indexes[k])]
                    inlier_slice = tvg_inlier_feat_idxs[
                        inlier_offset : inlier_offset + ic
                    ].copy()
                    f_mat = tvg_f_matrices[k].copy()
                    e_mat = tvg_e_matrices[k].copy()
                    h_mat = tvg_h_matrices[k].copy()
                    quat = tvg_quaternions[k].copy()
                    trans = tvg_translations[k].copy()

                    if swap:
                        inlier_slice = inlier_slice[:, ::-1].copy()
                        f_mat = f_mat.T.copy()
                        e_mat = e_mat.T.copy()
                        if np.any(h_mat != 0):
                            try:
                                h_mat = np.linalg.inv(h_mat)
                            except np.linalg.LinAlgError:
                                h_mat = np.zeros((3, 3), dtype=np.float64)
                        # Invert pose: q_inv = (w, -x, -y, -z), t_inv = -R^T @ t
                        w, x, y, z = quat
                        R = np.array(
                            [
                                [
                                    1 - 2 * (y * y + z * z),
                                    2 * (x * y - z * w),
                                    2 * (x * z + y * w),
                                ],
                                [
                                    2 * (x * y + z * w),
                                    1 - 2 * (x * x + z * z),
                                    2 * (y * z - x * w),
                                ],
                                [
                                    2 * (x * z - y * w),
                                    2 * (y * z + x * w),
                                    1 - 2 * (x * x + y * y),
                                ],
                            ]
                        )
                        quat = np.array([w, -x, -y, -z])
                        trans = -R.T @ trans

                    pair_tvg[key] = {
                        "config_type": config_str,
                        "inlier_count": ic,
                        "inlier_feature_indexes": inlier_slice,
                        "f_matrix": f_mat,
                        "e_matrix": e_mat,
                        "h_matrix": h_mat,
                        "quaternion_wxyz": quat,
                        "translation_xyz": trans,
                    }
                inlier_offset += ic

    # Build output arrays
    sorted_pairs = sorted(pair_matches.keys())
    total_matches = sum(len(m) for m in pair_matches.values())
    click.echo(f"  Merged: {len(sorted_pairs)} pairs, {total_matches} matches")

    out_image_index_pairs = np.zeros((len(sorted_pairs), 2), dtype=np.uint32)
    out_match_counts = np.zeros(len(sorted_pairs), dtype=np.uint32)
    out_match_feature_indexes = np.zeros((total_matches, 2), dtype=np.uint32)
    out_match_descriptor_distances = np.zeros(total_matches, dtype=np.float32)

    offset = 0
    for k, (idx_i, idx_j) in enumerate(sorted_pairs):
        out_image_index_pairs[k] = [idx_i, idx_j]
        matches = pair_matches[(idx_i, idx_j)]
        out_match_counts[k] = len(matches)
        for (fi, fj), dist in matches.items():
            out_match_feature_indexes[offset] = [fi, fj]
            out_match_descriptor_distances[offset] = dist
            offset += 1

    # Build TVG output arrays if any input had TVGs
    has_output_tvg = len(pair_tvg) > 0
    if has_output_tvg:
        config_type_set = {"undefined"}
        for tvg in pair_tvg.values():
            config_type_set.add(tvg["config_type"])
        config_type_list = sorted(config_type_set)
        config_type_to_idx = {ct: i for i, ct in enumerate(config_type_list)}
        undefined_idx = config_type_to_idx["undefined"]

        out_config_indexes = np.full(len(sorted_pairs), undefined_idx, dtype=np.uint8)
        out_inlier_counts = np.zeros(len(sorted_pairs), dtype=np.uint32)
        out_f_matrices = np.zeros((len(sorted_pairs), 3, 3), dtype=np.float64)
        out_e_matrices = np.zeros((len(sorted_pairs), 3, 3), dtype=np.float64)
        out_h_matrices = np.zeros((len(sorted_pairs), 3, 3), dtype=np.float64)
        out_quaternions = np.zeros((len(sorted_pairs), 4), dtype=np.float64)
        out_quaternions[:, 0] = 1.0  # identity quaternion default
        out_translations = np.zeros((len(sorted_pairs), 3), dtype=np.float64)

        all_inlier_idxs = []
        total_inliers = 0
        tvg_pair_count = 0

        for k, (idx_i, idx_j) in enumerate(sorted_pairs):
            tvg = pair_tvg.get((idx_i, idx_j))
            if tvg is not None:
                out_config_indexes[k] = config_type_to_idx[tvg["config_type"]]
                out_inlier_counts[k] = tvg["inlier_count"]
                out_f_matrices[k] = tvg["f_matrix"]
                out_e_matrices[k] = tvg["e_matrix"]
                out_h_matrices[k] = tvg["h_matrix"]
                out_quaternions[k] = tvg["quaternion_wxyz"]
                out_translations[k] = tvg["translation_xyz"]
                if tvg["inlier_count"] > 0:
                    all_inlier_idxs.append(tvg["inlier_feature_indexes"])
                total_inliers += tvg["inlier_count"]
                tvg_pair_count += 1

        if all_inlier_idxs:
            out_inlier_feature_indexes = np.concatenate(all_inlier_idxs, axis=0)
        else:
            out_inlier_feature_indexes = np.zeros((0, 2), dtype=np.uint32)

        click.echo(
            f"  TVGs: {tvg_pair_count}/{len(sorted_pairs)} pairs, "
            f"{total_inliers} total inliers"
        )

    # Build unified image arrays
    feature_counts = np.array(
        [image_info[n]["feature_count"] for n in unified_names], dtype=np.uint32
    )
    feature_tool_hashes = np.array(
        [list(image_info[n]["tool_hash"]) for n in unified_names], dtype=np.uint8
    )
    sift_content_hashes = np.array(
        [list(image_info[n]["content_hash"]) for n in unified_names], dtype=np.uint8
    )
    image_dims = np.array(
        [image_info[n]["dims"] for n in unified_names], dtype=np.uint32
    )

    # Build metadata from the first input file as a base
    base_meta = all_data[0][0]["metadata"]
    source_methods = []
    for data, _pairs_view in all_data:
        method = data["metadata"].get("matching_method", "unknown")
        source_methods.append(method)

    merged_data = {
        "metadata": {
            "version": 1,
            "matching_method": "merged",
            "matching_tool": "sfmtool",
            "matching_tool_version": "",
            "matching_options": {
                "source_files": [p.name for p in input_paths],
                "source_methods": source_methods,
            },
            "workspace": base_meta["workspace"],
            "timestamp": datetime.now().astimezone().isoformat(),
            "image_count": image_count,
            "image_pair_count": len(sorted_pairs),
            "match_count": total_matches,
            "has_two_view_geometries": has_output_tvg,
        },
        "content_hash": {},
        "image_names": unified_names,
        "feature_tool_hashes": feature_tool_hashes,
        "sift_content_hashes": sift_content_hashes,
        "feature_counts": feature_counts,
        "image_dims": image_dims,
        "image_index_pairs": out_image_index_pairs,
        "match_counts": out_match_counts,
        "match_feature_indexes": out_match_feature_indexes,
        "match_descriptor_distances": out_match_descriptor_distances,
        "has_two_view_geometries": has_output_tvg,
    }

    if has_output_tvg:
        merged_data["config_types"] = config_type_list
        merged_data["config_indexes"] = out_config_indexes
        merged_data["inlier_counts"] = out_inlier_counts
        merged_data["inlier_feature_indexes"] = out_inlier_feature_indexes
        merged_data["f_matrices"] = out_f_matrices
        merged_data["e_matrices"] = out_e_matrices
        merged_data["h_matrices"] = out_h_matrices
        merged_data["quaternions_wxyz"] = out_quaternions
        merged_data["translations_xyz"] = out_translations
        merged_data["tvg_metadata"] = {
            "image_pair_count": len(sorted_pairs),
            "inlier_count": total_inliers,
            "verification_tool": "merged",
            "verification_options": {
                "source_files": [p.name for p in input_paths],
            },
        }

    # Update workspace relative path for the output location
    out_abs = Path(os.path.abspath(out))
    ws_abs = base_meta["workspace"]["absolute_path"]
    merged_data["metadata"]["workspace"]["relative_path"] = os.path.relpath(
        ws_abs, out_abs.parent
    ).replace("\\", "/")

    out.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"Writing {out}...")
    write_matches(out, merged_data)
    click.echo(f"Done: {len(sorted_pairs)} pairs, {total_matches} matches")
