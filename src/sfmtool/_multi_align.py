# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Alignment of multiple SfM reconstructions to a common reference frame."""

from dataclasses import dataclass
from pathlib import Path

import click

from ._align import estimate_pairwise_alignment
from ._sfmtool import SfmrReconstruction


def _get_reconstruction_images(recon: SfmrReconstruction) -> dict[int, str]:
    """Get image names from a SfmrReconstruction."""
    return {i: name for i, name in enumerate(recon.image_names)}


def _find_shared_images(images_a: dict[int, str], images_b: dict[int, str]) -> set[str]:
    """Find image names that are shared between two reconstructions."""
    return set(images_a.values()) & set(images_b.values())


def _build_connectivity_graph(
    reconstructions: list[SfmrReconstruction],
) -> dict[int, dict[int, set[str]]]:
    """Build a connectivity graph based on shared images between reconstructions."""
    n = len(reconstructions)
    all_images = [_get_reconstruction_images(r) for r in reconstructions]

    graph: dict[int, dict[int, set[str]]] = {i: {} for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            shared = _find_shared_images(all_images[i], all_images[j])
            if shared:
                graph[i][j] = shared
                graph[j][i] = shared

    return graph


@dataclass
class AlignmentResult:
    """Result of aligning multiple reconstructions."""

    aligned: list[SfmrReconstruction | None]
    """Aligned reconstructions in same order as input. None if alignment failed."""

    total_shared_images: int
    """Total number of shared images used for alignment."""


def align_reconstructions(
    reference: SfmrReconstruction,
    to_align: list[SfmrReconstruction],
    method: str = "points",
    confidence: float = 0.7,
    use_ransac: bool = True,
    ransac_percentile: float = 95.0,
    ransac_iterations: int = 1000,
    verbose: bool = False,
) -> AlignmentResult:
    """Align reconstructions to a reference coordinate frame.

    Supports transitive alignment through a connectivity graph, so
    reconstructions that don't directly share images with the reference
    can still be aligned if they connect through other reconstructions.
    """
    if method == "points":
        return _align_with_points(
            reference=reference,
            to_align=to_align,
            use_ransac=use_ransac,
            ransac_percentile=ransac_percentile,
            ransac_iterations=ransac_iterations,
            verbose=verbose,
        )
    else:
        return _align_with_cameras(
            reference=reference,
            to_align=to_align,
            confidence=confidence,
            verbose=verbose,
        )


def _align_with_cameras(
    reference: SfmrReconstruction,
    to_align: list[SfmrReconstruction],
    confidence: float = 0.7,
    verbose: bool = False,
) -> AlignmentResult:
    """Align reconstructions using camera poses."""
    from ._align_by_cameras import build_image_matches, get_reconstruction_poses
    from .xform import SimilarityTransform, apply_transforms

    all_recons = [reference] + list(to_align)
    n = len(all_recons)

    graph = _build_connectivity_graph(all_recons)

    aligned_set = {0}
    current_recons = list(all_recons)
    results: list[SfmrReconstruction | None] = [None] * len(to_align)
    total_shared_images = 0

    aligned_data: dict[int, dict] = {}
    ref_quats, ref_centers = get_reconstruction_poses(current_recons[0])
    aligned_data[0] = {
        "images": _get_reconstruction_images(current_recons[0]),
        "quaternions": ref_quats,
        "camera_centers": ref_centers,
    }

    prev_count = 0
    while len(aligned_set) > prev_count:
        prev_count = len(aligned_set)

        for idx in range(1, n):
            if idx in aligned_set:
                continue

            aligned_neighbors = [nb for nb in graph[idx] if nb in aligned_set]
            if not aligned_neighbors:
                continue

            if verbose:
                click.echo(f"\nProcessing reconstruction {idx}...")
                click.echo(
                    f"  Connected to {len(aligned_neighbors)} aligned reconstruction(s)"
                )

            align_images = _get_reconstruction_images(current_recons[idx])
            align_quats, align_centers = get_reconstruction_poses(current_recons[idx])

            all_matches = []
            shared_count = 0

            for nb_idx in aligned_neighbors:
                shared_images = graph[idx][nb_idx]
                shared_count += len(shared_images)

                matches = build_image_matches(
                    shared_images=shared_images,
                    source_images=align_images,
                    source_quaternions=align_quats,
                    source_camera_centers=align_centers,
                    target_images=aligned_data[nb_idx]["images"],
                    target_quaternions=aligned_data[nb_idx]["quaternions"],
                    target_camera_centers=aligned_data[nb_idx]["camera_centers"],
                )
                all_matches.extend(matches)

            if not all_matches:
                if verbose:
                    click.echo("  Warning: No matches found, skipping")
                continue

            total_shared_images += shared_count

            try:
                result = estimate_pairwise_alignment(
                    all_matches,
                    confidence,
                    source_id=f"recon_{idx}",
                    target_id="reference_frame",
                )
                if verbose:
                    click.echo(f"  RMS error = {result.total_rms_error:.4f}")
            except Exception as e:
                if verbose:
                    click.echo(f"  Error: {e}")
                continue

            try:
                aligned_recon = apply_transforms(
                    recon=current_recons[idx],
                    transforms=[SimilarityTransform(result.transform)],
                )
                current_recons[idx] = aligned_recon
                results[idx - 1] = aligned_recon
                aligned_set.add(idx)

                new_quats, new_centers = get_reconstruction_poses(aligned_recon)
                aligned_data[idx] = {
                    "images": _get_reconstruction_images(aligned_recon),
                    "quaternions": new_quats,
                    "camera_centers": new_centers,
                }

                if verbose:
                    click.echo("  Successfully aligned")
            except Exception as e:
                if verbose:
                    click.echo(f"  Transform error: {e}")
                continue

    return AlignmentResult(aligned=results, total_shared_images=total_shared_images)


def _align_with_points(
    reference: SfmrReconstruction,
    to_align: list[SfmrReconstruction],
    use_ransac: bool = True,
    ransac_percentile: float = 95.0,
    ransac_iterations: int = 1000,
    verbose: bool = False,
) -> AlignmentResult:
    """Align reconstructions using 3D point correspondences."""
    from ._align_by_points import estimate_alignment_from_points_with_logging
    from .xform import SimilarityTransform, apply_transforms

    all_recons = [reference] + list(to_align)
    n = len(all_recons)

    graph = _build_connectivity_graph(all_recons)

    aligned_set = {0}
    current_recons = list(all_recons)
    results: list[SfmrReconstruction | None] = [None] * len(to_align)
    total_shared_images = 0

    prev_count = 0
    while len(aligned_set) > prev_count:
        prev_count = len(aligned_set)

        for idx in range(1, n):
            if idx in aligned_set:
                continue

            aligned_neighbors = [nb for nb in graph[idx] if nb in aligned_set]
            if not aligned_neighbors:
                continue

            if verbose:
                click.echo(f"\nProcessing reconstruction {idx}...")
                click.echo(
                    f"  Connected to {len(aligned_neighbors)} aligned reconstruction(s)"
                )

            align_images = _get_reconstruction_images(current_recons[idx])

            all_shared_pairs: list[tuple[int, int]] = []
            shared_count = 0

            for nb_idx in aligned_neighbors:
                shared_names = graph[idx][nb_idx]
                shared_count += len(shared_names)
                neighbor_images = _get_reconstruction_images(current_recons[nb_idx])

                for img_name in shared_names:
                    src_idx = next(
                        (i for i, n in align_images.items() if n == img_name), None
                    )
                    tgt_idx = next(
                        (i for i, n in neighbor_images.items() if n == img_name), None
                    )
                    if src_idx is not None and tgt_idx is not None:
                        all_shared_pairs.append((src_idx, tgt_idx))

            if not all_shared_pairs:
                if verbose:
                    click.echo("  Warning: No image pairs found, skipping")
                continue

            total_shared_images += shared_count

            first_neighbor = aligned_neighbors[0]

            try:
                result = estimate_alignment_from_points_with_logging(
                    source_recon=current_recons[idx],
                    target_recon=current_recons[first_neighbor],
                    shared_images=all_shared_pairs,
                    min_points=10,
                    use_ransac=use_ransac,
                    ransac_iterations=ransac_iterations,
                    ransac_percentile=ransac_percentile,
                    verbose=verbose,
                )
                if verbose:
                    click.echo(f"  RMS error = {result.total_rms_error:.4f}")
            except Exception as e:
                if verbose:
                    click.echo(f"  Error: {e}")
                continue

            try:
                aligned_recon = apply_transforms(
                    recon=current_recons[idx],
                    transforms=[SimilarityTransform(result.transform)],
                )
                current_recons[idx] = aligned_recon
                results[idx - 1] = aligned_recon
                aligned_set.add(idx)

                if verbose:
                    click.echo("  Successfully aligned")
            except Exception as e:
                if verbose:
                    click.echo(f"  Transform error: {e}")
                continue

    return AlignmentResult(aligned=results, total_shared_images=total_shared_images)


def align_command(
    reference_path: Path,
    align_paths: list[Path],
    output_dir: Path,
    method: str = "points",
    confidence: float = 0.7,
    max_error: float = 0.1,
    iterative: bool = False,
    visualize: bool = False,
    use_ransac: bool = True,
    ransac_percentile: float = 95.0,
    ransac_iterations: int = 1000,
) -> None:
    """CLI command for aligning reconstructions with file I/O."""
    import shutil

    output_dir = output_dir.resolve()
    all_input_paths = [reference_path.resolve()] + [p.resolve() for p in align_paths]

    for input_path in all_input_paths:
        try:
            input_path.relative_to(output_dir)
            raise click.ClickException(
                f"Input file {input_path.name} is inside output directory {output_dir}. "
                "Output directory must be different from input locations."
            )
        except ValueError:
            pass

    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Reference reconstruction: {reference_path.name}")
    click.echo(f"Aligning {len(align_paths)} reconstruction(s) to reference...")

    click.echo("\nLoading reconstructions...")
    click.echo(f"  Loading {reference_path.name}...")
    reference = SfmrReconstruction.load(str(reference_path))

    to_align = []
    for path in align_paths:
        click.echo(f"  Loading {path.name}...")
        to_align.append(SfmrReconstruction.load(str(path)))

    click.echo("\nAnalyzing connectivity...")
    graph = _build_connectivity_graph([reference] + to_align)
    for i, path in enumerate([reference_path] + list(align_paths)):
        neighbors = graph[i]
        if neighbors:
            neighbor_names = []
            for nb_idx, shared in neighbors.items():
                nb_path = ([reference_path] + list(align_paths))[nb_idx]
                neighbor_names.append(f"{nb_path.name} ({len(shared)} shared)")
            click.echo(f"  {path.name}: {', '.join(neighbor_names)}")
        else:
            click.echo(f"  {path.name}: (isolated)")

    for i, path in enumerate(align_paths, start=1):
        if not graph[i]:
            click.echo(
                f"\nWarning: No matching images found between reference and {path.name}"
            )

    reference_output = output_dir / reference_path.name
    click.echo(f"\nCopying reference to {reference_output}...")
    shutil.copy2(reference_path, reference_output)

    if method == "points":
        click.echo("\nUsing point-based alignment...")
        if use_ransac:
            click.echo(
                f"  RANSAC: percentile={ransac_percentile}, iterations={ransac_iterations}"
            )
    else:
        click.echo("\nUsing camera-based alignment...")

    result = align_reconstructions(
        reference=reference,
        to_align=to_align,
        method=method,
        confidence=confidence,
        use_ransac=use_ransac,
        ransac_percentile=ransac_percentile,
        ransac_iterations=ransac_iterations,
        verbose=True,
    )

    success_count = 0
    for i, aligned in enumerate(result.aligned):
        path = align_paths[i]
        if aligned is not None:
            output_path = output_dir / path.name
            aligned.save(str(output_path), operation="align")
            click.echo(f"  Saved {output_path}")
            success_count += 1

    click.echo("\n" + "=" * 50)
    click.echo("Alignment Summary")
    click.echo("=" * 50)
    click.echo(f"Reference: {reference_path.name}")
    click.echo(f"Successfully aligned: {success_count}/{len(align_paths)}")
    click.echo(f"Total matching images: {result.total_shared_images}")

    failed = [align_paths[i].name for i, a in enumerate(result.aligned) if a is None]
    if failed:
        click.echo(f"\nFailed to align: {', '.join(failed)}")

    click.echo(f"\nResults saved to: {output_dir}")
