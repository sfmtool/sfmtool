# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Alignment of multiple SfM reconstructions to a common reference frame."""

from dataclasses import dataclass
from pathlib import Path

import click

from .core import estimate_pairwise_alignment
from .._sfmtool.analysis import pair_images
from .._sfmtool.reconstruction import SfmrReconstruction
from .._workspace import require_version_2_workspace


def _build_connectivity_graph(
    reconstructions: list[SfmrReconstruction],
    file_hashes: list[list[bytes | None]] | None = None,
) -> dict[int, dict[int, dict]]:
    """Build a connectivity graph of the images reconstructions share.

    ``graph[i][j]`` is the ``pair_images`` result of reconstruction ``i``'s
    images (source) against ``j``'s (target), present only when it pairs at
    least one image. ``file_hashes`` holds each reconstruction's
    ``find_image_file_hashes()``, read here when not given.
    """
    n = len(reconstructions)
    if file_hashes is None:
        file_hashes = [r.find_image_file_hashes() for r in reconstructions]

    graph: dict[int, dict[int, dict]] = {i: {} for i in range(n)}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pairing = pair_images(
                reconstructions[i].image_names,
                file_hashes[i],
                reconstructions[j].image_names,
                file_hashes[j],
            )
            if len(pairing["pairs"]):
                graph[i][j] = pairing

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
    file_hashes: list[list[bytes | None]] | None = None,
) -> AlignmentResult:
    """Align reconstructions to a reference coordinate frame.

    Supports transitive alignment through a connectivity graph, so
    reconstructions that don't directly share images with the reference
    can still be aligned if they connect through other reconstructions.

    Images are paired by file hash, and by name where a hash is not known
    (``sfmtool.analysis.pair_images``). ``file_hashes`` holds the
    ``find_image_file_hashes()`` of ``reference`` and then of each of
    ``to_align``, read here when not given.
    """
    graph = _build_connectivity_graph([reference, *to_align], file_hashes)
    if method == "points":
        return _align_with_points(
            reference=reference,
            to_align=to_align,
            graph=graph,
            use_ransac=use_ransac,
            ransac_percentile=ransac_percentile,
            ransac_iterations=ransac_iterations,
            verbose=verbose,
        )
    else:
        return _align_with_cameras(
            reference=reference,
            to_align=to_align,
            graph=graph,
            confidence=confidence,
            verbose=verbose,
        )


def _align_with_cameras(
    reference: SfmrReconstruction,
    to_align: list[SfmrReconstruction],
    graph: dict[int, dict[int, dict]],
    confidence: float = 0.7,
    verbose: bool = False,
) -> AlignmentResult:
    """Align reconstructions using camera poses."""
    from .by_cameras import build_image_matches, get_reconstruction_poses
    from ..xform import SimilarityTransform, apply_transforms

    all_recons = [reference] + list(to_align)
    n = len(all_recons)

    aligned_set = {0}
    current_recons = list(all_recons)
    results: list[SfmrReconstruction | None] = [None] * len(to_align)
    total_shared_images = 0

    aligned_data: dict[int, dict] = {}
    ref_quats, ref_centers = get_reconstruction_poses(current_recons[0])
    aligned_data[0] = {
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

            align_names = current_recons[idx].image_names
            align_quats, align_centers = get_reconstruction_poses(current_recons[idx])

            all_matches = []
            shared_count = 0

            for nb_idx in aligned_neighbors:
                shared_pairs = graph[idx][nb_idx]["pairs"]
                shared_count += len(shared_pairs)

                matches = build_image_matches(
                    shared_pairs=shared_pairs,
                    source_names=align_names,
                    source_quaternions=align_quats,
                    source_camera_centers=align_centers,
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
    graph: dict[int, dict[int, dict]],
    use_ransac: bool = True,
    ransac_percentile: float = 95.0,
    ransac_iterations: int = 1000,
    verbose: bool = False,
) -> AlignmentResult:
    """Align reconstructions using 3D point correspondences."""
    from .by_points import estimate_alignment_from_points_with_logging
    from ..xform import SimilarityTransform, apply_transforms

    all_recons = [reference] + list(to_align)
    n = len(all_recons)

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

            all_shared_pairs: list[tuple[int, int]] = []
            shared_count = 0

            for nb_idx in aligned_neighbors:
                shared_pairs = graph[idx][nb_idx]["pairs"]
                shared_count += len(shared_pairs)
                all_shared_pairs.extend((int(s), int(t)) for s, t in shared_pairs)

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

    # Every input is written into output_dir under its basename, so two inputs
    # that share a basename (e.g. same-named files from different directories, or
    # the same file passed twice) would silently overwrite each other and the
    # user would get fewer outputs than inputs. Reject the collision up front.
    seen_names: dict[str, Path] = {}
    for input_path in all_input_paths:
        existing = seen_names.get(input_path.name)
        if existing is not None:
            raise click.ClickException(
                f"Inputs '{existing}' and '{input_path}' share the basename "
                f"'{input_path.name}' and would overwrite each other in the "
                "output directory. Give the inputs distinct file names."
            )
        seen_names[input_path.name] = input_path

    click.echo(f"Reference reconstruction: {reference_path.name}")
    click.echo(f"Aligning {len(align_paths)} reconstruction(s) to reference...")

    click.echo("\nLoading reconstructions...")
    click.echo(f"  Loading {reference_path.name}...")
    reference = SfmrReconstruction.load(reference_path)

    to_align = []
    for path in align_paths:
        click.echo(f"  Loading {path.name}...")
        to_align.append(SfmrReconstruction.load(path))

    # Aligning transforms and saves the reconstructions, as `sfm xform` does,
    # so a sift_files input must be in a version 2 workspace; checked before
    # any work.
    for recon in [reference, *to_align]:
        if recon.feature_source == "sift_files":
            require_version_2_workspace(recon.workspace_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("\nAnalyzing connectivity...")
    all_recons = [reference, *to_align]
    all_paths = [reference_path, *align_paths]
    file_hashes = [r.find_image_file_hashes() for r in all_recons]
    graph = _build_connectivity_graph(all_recons, file_hashes)
    for i, path in enumerate(all_paths):
        neighbors = graph[i]
        if neighbors:
            neighbor_names = []
            for nb_idx, pairing in neighbors.items():
                nb_path = all_paths[nb_idx]
                neighbor_names.append(
                    f"{nb_path.name} ({len(pairing['pairs'])} shared)"
                )
            click.echo(f"  {path.name}: {', '.join(neighbor_names)}")
        else:
            click.echo(f"  {path.name}: (isolated)")
    _echo_image_pairing(all_recons, all_paths, file_hashes)

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
        file_hashes=file_hashes,
    )

    success_count = 0
    for i, aligned in enumerate(result.aligned):
        path = align_paths[i]
        if aligned is not None:
            output_path = output_dir / path.name
            aligned.save(output_path, operation="align")
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


def _echo_image_pairing(
    recons: list[SfmrReconstruction],
    paths: list[Path],
    file_hashes: list[list[bytes | None]],
) -> None:
    """Print how each two inputs' images were paired: by file hash, by name,
    and the names that different files share, which are not paired."""
    for i in range(len(recons)):
        for j in range(i + 1, len(recons)):
            pairing = pair_images(
                recons[i].image_names,
                file_hashes[i],
                recons[j].image_names,
                file_hashes[j],
            )
            mismatched = pairing["same_name_different_file"]
            if not len(pairing["pairs"]) and not len(mismatched):
                continue
            click.echo(f"  {paths[i].name} and {paths[j].name}: {pairing['summary']}")
            if len(mismatched):
                names = recons[i].image_names
                listed = ", ".join(names[int(s)] for s, _ in mismatched[:3])
                more = f" and {len(mismatched) - 3} more" if len(mismatched) > 3 else ""
                click.echo(
                    f"    Not paired, same name but different file: {listed}{more}"
                )
