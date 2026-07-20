# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""RANSAC similarity decomposition of shared camera poses into rigid fragments.

A single least-squares similarity alignment between two reconstructions fits
the dominant rigid subset and hides structural failures: solves frequently
consist of several internally-rigid fragments at different scales and
orientations, plus isolated wrong frames. This module decomposes the shared
cameras into such fragments by repeatedly running RANSAC over minimal camera
subsets: two posed cameras determine a full similarity candidate (rotation
from the camera orientations, scale from the center distance), consensus is
measured with the same scale-relative position metric the comparison already
uses, and the largest consensus set is peeled off as a component until no
consensus of the minimum size remains. Whatever is left are individual
outlier frames.
"""

from dataclasses import dataclass, field

import numpy as np

from .align.core import ImageMatch, estimate_similarity_with_orientations
from ._sfmtool.geometry import Se3Transform


@dataclass
class FragmentComponent:
    """One internally-rigid subset of the shared cameras."""

    indices: np.ndarray
    """Indices into the shared-camera match list, ascending."""

    transform: Se3Transform
    """Similarity aligning this component's target poses to the reference."""

    pos_errors_pct: np.ndarray
    """Per-camera position error under ``transform``, % of reference scene scale."""

    rot_errors_deg: np.ndarray
    """Per-camera rotation error under ``transform``, degrees."""

    displacement_vs_first_pct: float | None = None
    """Mean distance between where component 1's transform and this component's
    transform place this component's cameras, % of scene scale (None for
    component 1). Unlike a raw translation, this measures the misplacement at
    the fragment's own location."""


@dataclass
class FragmentDecomposition:
    """Result of decomposing shared cameras into rigid fragments."""

    components: list[FragmentComponent] = field(default_factory=list)
    """Components in extraction order (largest consensus first)."""

    outlier_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    """Match indices belonging to no component."""

    outlier_pos_errors_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    """Outlier position errors under component 1's transform (% of scene scale)."""

    outlier_rot_errors_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    """Outlier rotation errors under component 1's transform (degrees)."""

    @property
    def is_single_rigid(self) -> bool:
        """True when every shared camera sits in one component."""
        return len(self.components) == 1 and len(self.outlier_indices) == 0


def _quat_multiply_single(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Row-wise Hamilton product of quaternions ``q`` (N,4 wxyz) with ``p`` (4,)."""
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    pw, px, py, pz = p
    return np.stack(
        [
            qw * pw - qx * px - qy * py - qz * pz,
            qw * px + qx * pw + qy * pz - qz * py,
            qw * py - qx * pz + qy * pw + qz * px,
            qw * pz + qx * py - qy * px + qz * pw,
        ],
        axis=1,
    )


class _PoseArrays:
    """Shared-camera poses unpacked into arrays for vectorized residuals."""

    def __init__(self, matches: list[ImageMatch]):
        self.source_centers = np.array([m.source_camera_center for m in matches])
        self.target_centers = np.array([m.target_camera_center for m in matches])
        self.source_quats = np.array(
            [np.asarray(m.source_quat.to_wxyz_array()) for m in matches]
        )
        self.target_quats = np.array(
            [np.asarray(m.target_quat.to_wxyz_array()) for m in matches]
        )

    def errors(
        self, transform: Se3Transform, indices: np.ndarray, scene_scale: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Position (% of scene scale) and rotation (deg) residuals at ``indices``.

        Uses the same residual conventions as the extrinsics comparison:
        centers are mapped through the similarity, orientations compose with
        the conjugate of the transform rotation.
        """
        rotation = transform.rotation
        rot_matrix = np.asarray(rotation.to_rotation_matrix())
        scale = transform.scale
        translation = np.asarray(transform.translation)

        mapped = scale * (self.source_centers[indices] @ rot_matrix.T) + translation
        pos = np.linalg.norm(mapped - self.target_centers[indices], axis=1)
        pos_pct = pos / scene_scale * 100.0

        conj = np.asarray(rotation.conjugate().to_wxyz_array())
        conj = conj / np.linalg.norm(conj)
        mapped_quats = _quat_multiply_single(self.source_quats[indices], conj)
        dots = np.abs(np.sum(mapped_quats * self.target_quats[indices], axis=1))
        rot_deg = np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))
        return pos_pct, rot_deg


def _candidate_pairs(
    indices: np.ndarray, max_candidates: int, rng: np.random.Generator
) -> np.ndarray:
    """Minimal-subset pairs to try: all pairs when few, else a random sample."""
    n = len(indices)
    n_pairs = n * (n - 1) // 2
    if n_pairs <= max_candidates:
        a, b = np.triu_indices(n, k=1)
    else:
        a = rng.integers(0, n, size=max_candidates)
        b = rng.integers(0, n - 1, size=max_candidates)
        b = np.where(b >= a, b + 1, b)
    return np.stack([indices[a], indices[b]], axis=1)


def _fit_pair(matches: list[ImageMatch], a: int, b: int) -> Se3Transform | None:
    """Similarity candidate from two posed cameras (rotation + center distance)."""
    transform = estimate_similarity_with_orientations([matches[a], matches[b]])
    if transform.scale <= 0:
        return None
    return transform


def _largest_consensus(
    matches: list[ImageMatch],
    poses: _PoseArrays,
    indices: np.ndarray,
    scene_scale: float,
    pos_threshold_pct: float,
    rot_threshold_deg: float,
    min_size: int,
    max_candidates: int,
    rng: np.random.Generator,
) -> FragmentComponent | None:
    """RANSAC one component out of ``indices``; None when consensus < ``min_size``."""
    baseline_floor = 1e-9 * max(scene_scale, 1e-12)

    best_inliers: np.ndarray | None = None
    best_transform: Se3Transform | None = None
    for a, b in _candidate_pairs(indices, max_candidates, rng):
        # Coincident centers pin no scale; skip the degenerate minimal subset.
        if (
            np.linalg.norm(poses.target_centers[a] - poses.target_centers[b])
            < baseline_floor
            or np.linalg.norm(poses.source_centers[a] - poses.source_centers[b])
            < baseline_floor
        ):
            continue
        transform = _fit_pair(matches, int(a), int(b))
        if transform is None:
            continue
        pos_pct, rot_deg = poses.errors(transform, indices, scene_scale)
        inliers = indices[(pos_pct < pos_threshold_pct) & (rot_deg < rot_threshold_deg)]
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_transform = transform

    if best_inliers is None or len(best_inliers) < min_size:
        return None

    # Refit on the consensus set and re-collect inliers until stable.
    for _ in range(10):
        transform = estimate_similarity_with_orientations(
            [matches[i] for i in best_inliers]
        )
        if transform.scale <= 0:
            break
        pos_pct, rot_deg = poses.errors(transform, indices, scene_scale)
        inliers = indices[(pos_pct < pos_threshold_pct) & (rot_deg < rot_threshold_deg)]
        if len(inliers) < min_size:
            break
        changed = len(inliers) != len(best_inliers) or not np.array_equal(
            inliers, best_inliers
        )
        best_inliers = inliers
        best_transform = transform
        if not changed:
            break

    pos_pct, rot_deg = poses.errors(best_transform, best_inliers, scene_scale)
    return FragmentComponent(
        indices=np.sort(best_inliers),
        transform=best_transform,
        pos_errors_pct=pos_pct[np.argsort(best_inliers)],
        rot_errors_deg=rot_deg[np.argsort(best_inliers)],
    )


def decompose_fragments(
    matches: list[ImageMatch],
    scene_scale: float,
    pos_threshold_pct: float = 3.5,
    rot_threshold_deg: float = 5.0,
    min_size: int = 5,
    max_candidates: int = 1000,
    seed: int = 0,
) -> FragmentDecomposition:
    """Decompose shared cameras into internally-rigid similarity components.

    Repeatedly RANSACs a similarity over the not-yet-assigned shared cameras
    (two posed cameras per candidate), takes the largest consensus set as the
    next component, and stops when no consensus reaches ``min_size``. Cameras
    in no component are individual outlier frames, reported with their errors
    under component 1's transform.

    Args:
        matches: Shared-camera pose pairs (target reconstruction as source,
            reference as target — the alignment direction the comparison uses).
        scene_scale: Reference scene scale; position errors are % of this.
        pos_threshold_pct: Consensus position threshold, % of scene scale.
        rot_threshold_deg: Consensus rotation threshold, degrees.
        min_size: Smallest camera count that still counts as a component.
        max_candidates: RANSAC candidate cap per component (all pairs when fewer).
        seed: RNG seed for candidate sampling (deterministic output).

    Returns:
        FragmentDecomposition with components largest-first plus outliers.
    """
    poses = _PoseArrays(matches)
    rng = np.random.default_rng(seed)

    decomposition = FragmentDecomposition()
    remaining = np.arange(len(matches))
    while len(remaining) >= max(2, min_size):
        component = _largest_consensus(
            matches,
            poses,
            remaining,
            scene_scale,
            pos_threshold_pct,
            rot_threshold_deg,
            min_size,
            max_candidates,
            rng,
        )
        if component is None:
            break
        decomposition.components.append(component)
        remaining = np.setdiff1d(remaining, component.indices)

    decomposition.outlier_indices = remaining
    if decomposition.components and len(remaining):
        pos_pct, rot_deg = poses.errors(
            decomposition.components[0].transform, remaining, scene_scale
        )
        decomposition.outlier_pos_errors_pct = pos_pct
        decomposition.outlier_rot_errors_deg = rot_deg

    # How far component 1's alignment displaces each later component's cameras
    # from their own alignment, evaluated at the fragment's location.
    if len(decomposition.components) > 1:
        first = decomposition.components[0].transform
        for component in decomposition.components[1:]:
            centers = poses.source_centers[component.indices]
            displaced = np.linalg.norm(
                np.asarray(first @ centers) - np.asarray(component.transform @ centers),
                axis=1,
            )
            component.displacement_vs_first_pct = float(
                np.mean(displaced) / scene_scale * 100.0
            )
    return decomposition


def _rotation_angle_deg(transform: Se3Transform) -> float:
    """Rotation angle of a similarity transform, degrees."""
    w = abs(transform.rotation.w)
    return float(np.degrees(2.0 * np.arccos(np.clip(w, 0.0, 1.0))))


def _print_error_stats(label: str, values: np.ndarray) -> None:
    print(
        f"      {label}: mean {np.mean(values):.2f}, "
        f"median {np.median(values):.2f}, max {np.max(values):.2f}"
    )


def print_fragment_decomposition(
    decomposition: FragmentDecomposition,
    image_names: list[str],
    pos_threshold_pct: float,
    rot_threshold_deg: float,
    min_size: int,
) -> None:
    """Print the fragment decomposition in the comparison's report style.

    ``image_names`` holds the reference-side image name for each shared-camera
    match index.
    """
    n_shared = sum(len(c.indices) for c in decomposition.components) + len(
        decomposition.outlier_indices
    )
    print(f"\n  Fragment decomposition (RANSAC similarity, {n_shared} shared cameras):")
    print(
        f"    Consensus: position < {pos_threshold_pct:g}% of scene scale, "
        f"rotation < {rot_threshold_deg:g} deg, min component size {min_size}"
    )

    if not decomposition.components:
        print(
            "    No consensus component found — no rigid subset of "
            f"{min_size}+ cameras agrees on a similarity."
        )
        return

    n_outliers = len(decomposition.outlier_indices)
    print(
        f"    Components: {len(decomposition.components)}, outlier frames: {n_outliers}"
    )

    reference = decomposition.components[0]
    for rank, component in enumerate(decomposition.components, start=1):
        names = [image_names[i] for i in component.indices]
        print(
            f"\n    Component {rank}: {len(names)} cameras ({names[0]} .. {names[-1]})"
        )
        _print_error_stats(
            "Position error (% of scene scale)", component.pos_errors_pct
        )
        _print_error_stats("Rotation error (deg)", component.rot_errors_deg)
        if rank > 1:
            # Reference-frame delta: undo component 1's alignment, apply this
            # component's. Scale ratio and rotation delta are frame-independent;
            # the translation part is reported as the displacement at the
            # fragment's own location (a raw translation about the origin
            # would be dominated by the scale mismatch).
            relative = component.transform @ reference.transform.inverse()
            print(
                f"      Vs component 1: scale x{relative.scale:.4f}, "
                f"rotation {_rotation_angle_deg(relative):.2f} deg, "
                f"displacement {component.displacement_vs_first_pct:.2f}% "
                "of scene scale"
            )

    if n_outliers:
        print("\n    Outlier frames (errors vs component 1):")
        order = np.argsort(-decomposition.outlier_pos_errors_pct)
        for k in order:
            idx = decomposition.outlier_indices[k]
            print(
                f"      {image_names[idx]}: position "
                f"{decomposition.outlier_pos_errors_pct[k]:.2f}% of scene scale, "
                f"rotation {decomposition.outlier_rot_errors_deg[k]:.2f} deg"
            )
