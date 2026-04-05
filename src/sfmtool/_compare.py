# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare two SfM reconstructions and report detailed differences."""

import textwrap
from pathlib import Path

import numpy as np

from ._align import ImageMatch, estimate_pairwise_alignment
from ._histogram_utils import print_histogram
from ._point_correspondence import find_point_correspondences
from ._sfmtool import RotQuaternion, SfmrReconstruction


def compare_reconstructions(
    recon1: SfmrReconstruction,
    recon2: SfmrReconstruction,
    recon1_name: str | None = None,
    recon2_name: str | None = None,
) -> None:
    """Compare two reconstructions and print detailed analysis."""
    if recon1_name is None:
        recon1_name = recon1.source_metadata.get("source_path", "recon1")
        if "/" in recon1_name or "\\" in recon1_name:
            recon1_name = Path(recon1_name).name
    if recon2_name is None:
        recon2_name = recon2.source_metadata.get("source_path", "recon2")
        if "/" in recon2_name or "\\" in recon2_name:
            recon2_name = Path(recon2_name).name

    print("\nComparing reconstructions:")
    print(f"  Reference: {recon1_name}")
    print(f"  Target:    {recon2_name}")
    print("=" * 70)

    print("\n[1/6] Analyzing reconstructions...")

    _print_reconstruction_summary(recon1, "Reference")
    _print_reconstruction_summary(recon2, "Target")

    # Compare camera intrinsics
    print("\n[2/6] Comparing camera intrinsics...")
    _compare_cameras(recon1, recon2)

    # Find matching images
    print("\n[3/6] Finding matching images...")
    matches = _find_matching_images(recon1, recon2)
    print(f"  Matching images: {len(matches)}")
    print(f"  Only in reference: {recon1.image_count - len(matches)}")
    print(f"  Only in target: {recon2.image_count - len(matches)}")

    if len(matches) == 0:
        print("\nNo matching images found. Cannot perform alignment.")
        return

    # Perform alignment
    print("\n[4/6] Performing alignment...")
    try:
        image_matches = []
        for idx1, idx2 in matches:
            quat1_wxyz = recon1.quaternions_wxyz[idx1]
            trans1 = recon1.translations[idx1]
            quat2_wxyz = recon2.quaternions_wxyz[idx2]
            trans2 = recon2.translations[idx2]

            quat1 = RotQuaternion.from_wxyz_array(quat1_wxyz)
            center1 = quat1.camera_center(trans1)

            quat2 = RotQuaternion.from_wxyz_array(quat2_wxyz)
            center2 = quat2.camera_center(trans2)

            # source = recon2 (target), target = recon1 (reference)
            image_name = Path(recon1.image_names[idx1]).name
            match = ImageMatch(
                image_name=image_name,
                source_index=idx2,
                target_index=idx1,
                source_quat=quat2,
                source_camera_center=center2,
                target_quat=quat1,
                target_camera_center=center1,
                quality=1.0,
            )
            image_matches.append(match)

        alignment_result = estimate_pairwise_alignment(
            matches=image_matches,
            confidence_threshold=0.0,
            source_id=recon2_name,
            target_id=recon1_name,
        )
        _print_alignment_results(alignment_result)
    except Exception as e:
        print(f"\nAlignment failed: {e}")
        alignment_result = None

    # Compare image extrinsics
    print("\n[5/6] Comparing image extrinsics...")
    _compare_image_extrinsics(recon1, recon2, matches, alignment_result)

    # Compare feature usage
    print("\n[6/7] Comparing feature usage...")
    _compare_feature_usage(recon1, recon2, matches)

    # Compare 3D points
    print("\n[7/7] Comparing 3D points...")
    _compare_3d_points(recon1, recon2, matches, alignment_result)

    print("\n" + "=" * 70)
    print("Comparison complete!")

    # Summarize relationship
    if alignment_result is not None:
        mean_pos_error = (
            np.mean(
                [
                    np.linalg.norm(
                        alignment_result.transform.apply_to_point(
                            RotQuaternion.from_wxyz_array(
                                recon2.quaternions_wxyz[idx2]
                            ).camera_center(recon2.translations[idx2])
                        )
                        - (
                            RotQuaternion.from_wxyz_array(
                                recon1.quaternions_wxyz[idx1]
                            ).camera_center(recon1.translations[idx1])
                        )
                    )
                    for idx1, idx2 in matches
                ]
            )
            if matches
            else 0.0
        )

        transform = alignment_result.transform
        is_near_identity = (
            abs(transform.scale - 1.0) < 0.001
            and np.linalg.norm(transform.translation) < 0.01
            and abs(transform.rotation.w) > 0.9999
        )

        num_matching = len(matches)
        num_ref_total = recon1.image_count
        num_target_total = recon2.image_count

        if mean_pos_error < 0.01:
            if is_near_identity:
                print(
                    f"\nCONCLUSION: The {num_matching} overlapping images are IDENTICAL"
                )
                print(
                    "            (no transform needed - already in same coordinate frame)."
                )
            else:
                print(
                    f"\nCONCLUSION: The {num_matching} overlapping images are IDENTICAL"
                )
                print(
                    "            except for a similarity transform (scale + rotation + translation)."
                )

            if num_ref_total > num_matching or num_target_total > num_matching:
                print()
                print(
                    f"NOTE: Reference has {num_ref_total - num_matching} additional image(s) not in target."
                )
                print(
                    f"      Target has {num_target_total - num_matching} additional image(s) not in reference."
                )
                print("      This comparison only covers the overlapping portion.")
        elif mean_pos_error < 0.1:
            print(
                f"\nCONCLUSION: The {num_matching} overlapping images are VERY SIMILAR,"
            )
            print("            differing primarily by a similarity transform.")
            if num_ref_total > num_matching or num_target_total > num_matching:
                print(
                    f"            (Ref: +{num_ref_total - num_matching}, Target: +{num_target_total - num_matching} additional images)"
                )
        else:
            print(
                f"\nCONCLUSION: The {num_matching} overlapping images have SIGNIFICANT DIFFERENCES"
            )
            print(
                f"            even after alignment (mean error: {mean_pos_error:.3f})."
            )
            if num_ref_total > num_matching or num_target_total > num_matching:
                print(
                    f"            (Ref: +{num_ref_total - num_matching}, Target: +{num_target_total - num_matching} additional images)"
                )


def _print_reconstruction_summary(recon: SfmrReconstruction, label: str) -> None:
    """Print summary of a reconstruction."""
    print(f"\n  {label}:")
    print(f"    Images: {recon.image_count}")

    try:
        from deadline.job_attachments.api import summarize_path_list

        print("    Image paths:")
        print(
            textwrap.indent(summarize_path_list(recon.image_names).rstrip(), "      ")
        )
    except ImportError:
        pass

    print(f"    Cameras: {recon.camera_count}")
    print(f"    3D points: {recon.point_count}")
    print(f"    Observations: {recon.observation_count}")


def _compare_cameras(recon1: SfmrReconstruction, recon2: SfmrReconstruction) -> None:
    """Compare camera intrinsics between two reconstructions."""
    cameras1 = recon1.cameras
    cameras2 = recon2.cameras

    print(f"  Reference has {len(cameras1)} camera(s)")
    print(f"  Target has {len(cameras2)} camera(s)")

    for i, cam1 in enumerate(cameras1):
        print(f"\n  Camera {i}:")
        print(f"    Reference model: {cam1.model}")
        print(f"    Reference resolution: {cam1.width}x{cam1.height}")
        print(f"    Reference params: {dict(cam1.parameters)}")

        if i < len(cameras2):
            cam2 = cameras2[i]
            print(f"    Target model: {cam2.model}")
            print(f"    Target resolution: {cam2.width}x{cam2.height}")
            print(f"    Target params: {dict(cam2.parameters)}")

            if cam1.model != cam2.model:
                print("    Different camera models!")
            elif cam1.width != cam2.width or cam1.height != cam2.height:
                print("    Different resolutions!")
            else:
                params1 = cam1.parameters
                params2 = cam2.parameters
                all_match = True
                for key in params1.keys():
                    if key not in params2:
                        print(f"    Parameter {key} not in target")
                        all_match = False
                    elif not np.isclose(params1[key], params2[key], rtol=1e-3):
                        diff = params2[key] - params1[key]
                        print(f"    Parameter {key} differs by {diff:.6f}")
                        all_match = False
                if all_match:
                    print("    Camera parameters match")
        else:
            print(f"    Camera {i} not found in target")


def _find_matching_images(
    recon1: SfmrReconstruction, recon2: SfmrReconstruction
) -> list[tuple[int, int]]:
    """Find matching images between two reconstructions."""
    name_to_idx1 = {Path(name).name: idx for idx, name in enumerate(recon1.image_names)}
    name_to_idx2 = {Path(name).name: idx for idx, name in enumerate(recon2.image_names)}

    matches = []
    for name, idx1 in name_to_idx1.items():
        if name in name_to_idx2:
            idx2 = name_to_idx2[name]
            matches.append((idx1, idx2))

    return matches


def _print_alignment_results(alignment_result) -> None:
    """Print alignment transformation and quality metrics."""
    transform_dict = alignment_result.transform.to_dict()

    print("\n  Alignment succeeded!")
    print(f"  Matched images: {len(alignment_result.matches)}")
    print(f"  RMS error: {alignment_result.total_rms_error:.4f}")

    print("\n  Transform to align target to reference:")
    print(f"    Scale: {transform_dict['scale']:.6f}")
    print(f"    Translation: {transform_dict['translation']}")
    rot = transform_dict["rotation"]
    print(
        f"    Rotation (quat): w={rot['w']:.4f}, x={rot['x']:.4f}, "
        f"y={rot['y']:.4f}, z={rot['z']:.4f}"
    )

    quat = RotQuaternion(rot["w"], rot["x"], rot["y"], rot["z"])
    rotation_angle_rad = 2 * np.arccos(np.clip(abs(rot["w"]), 0, 1))
    rotation_angle_deg = np.degrees(rotation_angle_rad)

    if rotation_angle_deg < 0.01:
        print("    Rotation (euler): yaw=~0, pitch=~0, roll=~0 (near-identity)")
    else:
        roll, pitch, yaw = quat.to_euler_angles()
        euler_deg = np.degrees([roll, pitch, yaw])
        print(
            f"    Rotation (euler): yaw={euler_deg[2]:.2f}, "
            f"pitch={euler_deg[1]:.2f}, roll={euler_deg[0]:.2f}"
        )


def _compare_image_extrinsics(
    recon1: SfmrReconstruction,
    recon2: SfmrReconstruction,
    matches: list[tuple[int, int]],
    alignment_result,
) -> None:
    """Compare camera extrinsics for matching images."""
    if alignment_result is None:
        print("  Skipping (no alignment available)")
        return

    transform = alignment_result.transform

    print(f"\n  Comparing {len(matches)} matching image(s):")

    position_errors = []
    rotation_errors = []

    for idx1, idx2 in matches:
        quat1_wxyz = recon1.quaternions_wxyz[idx1]
        trans1 = recon1.translations[idx1]
        quat2_wxyz = recon2.quaternions_wxyz[idx2]
        trans2 = recon2.translations[idx2]

        quat1 = RotQuaternion.from_wxyz_array(quat1_wxyz)
        center1 = quat1.camera_center(trans1)

        quat2 = RotQuaternion.from_wxyz_array(quat2_wxyz)
        center2 = quat2.camera_center(trans2)

        center2_transformed = transform @ center2

        pos_error = np.linalg.norm(center2_transformed - center1)
        position_errors.append(pos_error)

        quat2_transformed = quat2 * transform.rotation.conjugate()
        dot = abs(
            np.dot(
                np.asarray(quat2_transformed.to_wxyz_array()),
                np.asarray(quat1.to_wxyz_array()),
            )
        )
        dot = np.clip(dot, 0, 1)
        rot_error_rad = 2 * np.arccos(dot)
        rot_error_deg = np.degrees(rot_error_rad)
        rotation_errors.append(rot_error_deg)

    position_errors = np.array(position_errors)
    rotation_errors = np.array(rotation_errors)

    print("\n  Position errors (after alignment):")
    print_histogram(position_errors, "Distribution", num_buckets=60, min_val=0)

    print("\n  Rotation errors (after alignment):")
    print_histogram(
        rotation_errors, "Distribution (degrees)", num_buckets=60, min_val=0
    )

    if len(matches) > 0:
        worst_pos_idx = np.argmax(position_errors)
        worst_rot_idx = np.argmax(rotation_errors)

        idx1_pos, _ = matches[worst_pos_idx]
        idx1_rot, _ = matches[worst_rot_idx]

        print("\n  Worst position error:")
        print(f"    Image: {Path(recon1.image_names[idx1_pos]).name}")
        print(f"    Error: {position_errors[worst_pos_idx]:.4f}")

        print("\n  Worst rotation error:")
        print(f"    Image: {Path(recon1.image_names[idx1_rot]).name}")
        print(f"    Error: {rotation_errors[worst_rot_idx]:.4f}")


def _compare_feature_usage(
    recon1: SfmrReconstruction,
    recon2: SfmrReconstruction,
    matches: list[tuple[int, int]],
) -> None:
    """Compare SIFT feature usage for matching images."""
    print(f"\n  Analyzing feature usage for {len(matches)} matching image(s):")

    same_sift_count = 0
    different_sift_count = 0

    for idx1, idx2 in matches:
        sift_hash1 = recon1.sift_content_hashes[idx1]
        sift_hash2 = recon2.sift_content_hashes[idx2]

        if sift_hash1 == sift_hash2:
            same_sift_count += 1
        else:
            different_sift_count += 1

    print(f"    Same SIFT file: {same_sift_count}")
    print(f"    Different SIFT file: {different_sift_count}")

    if same_sift_count > 0:
        print("\n  Comparing feature index usage for images with same SIFT:")
        _compare_feature_indexes(recon1, recon2, matches)


def _compare_feature_indexes(
    recon1: SfmrReconstruction,
    recon2: SfmrReconstruction,
    matches: list[tuple[int, int]],
) -> None:
    """Compare which feature indexes are used for images with same SIFT file."""
    used_features1 = _build_used_features_map(recon1)
    used_features2 = _build_used_features_map(recon2)

    overlap_stats = []
    only_in_ref_stats = []
    only_in_target_stats = []

    for idx1, idx2 in matches:
        sift_hash1 = recon1.sift_content_hashes[idx1]
        sift_hash2 = recon2.sift_content_hashes[idx2]

        if sift_hash1 != sift_hash2:
            continue

        features1 = used_features1.get(idx1, set())
        features2 = used_features2.get(idx2, set())

        overlap = features1 & features2
        only_in_ref = features1 - features2
        only_in_target = features2 - features1

        overlap_stats.append(len(overlap))
        only_in_ref_stats.append(len(only_in_ref))
        only_in_target_stats.append(len(only_in_target))

    if len(overlap_stats) > 0:
        print("\n    Feature overlap statistics:")
        print(f"      Overlapping features (mean): {np.mean(overlap_stats):.1f}")
        print(f"      Overlapping features (median): {np.median(overlap_stats):.1f}")
        print(f"      Only in reference (mean): {np.mean(only_in_ref_stats):.1f}")
        print(f"      Only in target (mean): {np.mean(only_in_target_stats):.1f}")


def _build_used_features_map(recon: SfmrReconstruction) -> dict[int, set[int]]:
    """Build mapping from image index to set of used feature indexes."""
    used_features: dict[int, set[int]] = {}

    for img_idx, feat_idx in zip(
        recon.track_image_indexes, recon.track_feature_indexes
    ):
        img_idx = int(img_idx)
        if img_idx not in used_features:
            used_features[img_idx] = set()
        used_features[img_idx].add(int(feat_idx))

    return used_features


def _compare_3d_points(
    recon1: SfmrReconstruction,
    recon2: SfmrReconstruction,
    matches: list[tuple[int, int]],
    alignment_result,
) -> None:
    """Compare 3D points between reconstructions."""
    print("\n  Finding corresponding 3D points...")

    try:
        point_correspondences, positions1, positions2 = find_point_correspondences(
            source_recon=recon1,
            target_recon=recon2,
            shared_images=matches,
        )
    except ValueError:
        n_points1 = recon1.point_count
        n_points2 = recon2.point_count

        print("    Corresponding point pairs: 0")
        print(f"    Reference points: {n_points1} (0 matched, {n_points1} unique)")
        print(f"    Target points: {n_points2} (0 matched, {n_points2} unique)")
        print("    (No corresponding points found)")
        return

    n_ref_matched = len(point_correspondences)
    n_target_matched = len(set(point_correspondences.values()))
    n_points1 = recon1.point_count
    n_points2 = recon2.point_count

    print(f"    Corresponding point pairs: {n_ref_matched}")
    print(
        f"    Reference points: {n_points1} ({n_ref_matched} matched, {n_points1 - n_ref_matched} unique)"
    )
    print(
        f"    Target points: {n_points2} ({n_target_matched} matched, {n_points2 - n_target_matched} unique)"
    )

    if alignment_result is None:
        print("    (Skipping position comparison - no alignment available)")
        return

    print("\n  Comparing 3D positions of corresponding points...")

    positions2_transformed = alignment_result.transform @ positions2

    distances = np.linalg.norm(positions2_transformed - positions1, axis=1)

    print_histogram(
        distances,
        "Distance distribution (after alignment)",
        num_buckets=60,
        min_val=0,
        show_stats=True,
    )

    within_001 = np.sum(distances < 0.01)
    within_01 = np.sum(distances < 0.1)
    within_1 = np.sum(distances < 1.0)

    print("\n    Points within distance thresholds:")
    print(f"      < 0.01: {within_001} ({within_001 / n_ref_matched * 100:.1f}%)")
    print(f"      < 0.1:  {within_01} ({within_01 / n_ref_matched * 100:.1f}%)")
    print(f"      < 1.0:  {within_1} ({within_1 / n_ref_matched * 100:.1f}%)")

    if len(distances) > 0:
        worst_idx = np.argmax(distances)
        worst_point_id1 = list(point_correspondences.keys())[worst_idx]
        worst_point_id2 = point_correspondences[worst_point_id1]

        print("\n    Worst correspondence:")
        print(f"      Point IDs: {worst_point_id1} <-> {worst_point_id2}")
        print(f"      Distance: {distances[worst_idx]:.4f}")
