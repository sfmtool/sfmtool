# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Structure-free focal / camera-model estimate (`sfm estimate-intrinsics`).

The CLI face of the structure-free focal vote (see
`specs/cli/reconstruction/estimate-intrinsics-command.md` and
`specs/core/geometry/estimate-intrinsics.md`): cluster tracks in, a shared
focal and a camera-model verdict out, with no reconstruction in between.

The verdict semantics -- which column won, whether a fisheye verdict is
corroborated, which votes are the evidence behind it -- belong to the
`estimate_intrinsics` kernel, not to this module; the command reads them off
its result and spends its own code on I/O, the report and the `.camrig`.

Unlike its siblings the command is not wrapped in `timed_command` -- `--json`
puts the vote result on stdout, and a trailing timing line would stop that
output being parseable.
"""

from __future__ import annotations

import math
from pathlib import Path

import click


# `--model` -> the `focal_vote` column set it evaluates. The named forms run a
# single column, so the binding reports that column's own consensus and no
# arbitration takes place.
_MODEL_COLUMNS = {
    "auto": ("pinhole", "equidistant"),
    "pinhole": ("pinhole",),
    "fisheye": ("equidistant",),
}

# The binding's camera-model names.
_PINHOLE = "Pinhole"
_EQUIDISTANT = "EquidistantFisheye"

# The camera model each verdict is stored as in a `.camrig`. Both are
# centred-principal-point, distortion-free maps, which is exactly what the
# vote estimates.
_CAMRIG_MODEL = {_PINHOLE: "SIMPLE_PINHOLE", _EQUIDISTANT: "EQUIDISTANT_FISHEYE"}


def _resolve_dimensions(image_dims, image_names: list[str]) -> tuple[int, int]:
    """The one `(width, height)` every image shares, or a `UsageError`.

    The vote assumes a single shared camera with a centred principal point, so
    a file mixing resolutions cannot be answered as one estimate.
    """
    import numpy as np

    if image_dims is None:
        raise click.UsageError(
            "this .matches file stores no image dimensions (format version <= "
            "3); re-run `sfm match` to produce a file the vote can read"
        )
    dims = np.asarray(image_dims, dtype=np.int64)
    distinct, counts = np.unique(dims, axis=0, return_counts=True)
    if len(distinct) != 1:
        lines = []
        for (w, h), count in zip(distinct, counts):
            example = image_names[int(np.flatnonzero((dims == (w, h)).all(1))[0])]
            lines.append(f"  {w}x{h}  ({count} image(s))  e.g. {example}")
        raise click.UsageError(
            "the images in this .matches file have more than one resolution, "
            "and the vote estimates ONE shared camera:\n"
            + "\n".join(lines)
            + "\nSplit the images into per-camera .matches files and estimate "
            "each separately."
        )
    return int(distinct[0][0]), int(distinct[0][1])


def _load_observations(matches_path: Path) -> dict:
    """Flat cluster-contiguous observation arrays for the whole `.matches` file.

    Every cluster the file admits votes: the selection is the unrestricted one
    (all images, the format's own member admission), because the vote is a
    referee over the capture's full pair graph.
    """
    import numpy as np

    from .._sfmtool.io import MatchesFile

    mfile = MatchesFile(matches_path)
    if not mfile.has_clusters:
        raise click.UsageError(
            f"{matches_path} has no clusters section; run `sfm match --cluster` "
            "to produce a cluster-bearing .matches file"
        )
    image_names = list(mfile.image_names)
    width, height = _resolve_dimensions(mfile.image_dims, image_names)

    selection = mfile.select_clusters()
    starts = np.asarray(selection.cluster_starts, dtype=np.int64)
    cluster_indexes = np.repeat(
        np.arange(len(starts) - 1, dtype=np.uint32), np.diff(starts)
    )
    image_indexes = np.asarray(selection.member_images, dtype=np.uint32)
    # The backbone states its members' positions, whatever stage the file is
    # at: the detections of a matcher output, or the refinement's own answer
    # once `sfm cluster-patches` has run. The default selection has already
    # dropped the members the refinement excluded, by status -- which is the
    # only exclusion rule there is; the values themselves are always real.
    # Stored float32, widened here because the vote solves in float64.
    positions = np.ascontiguousarray(
        np.asarray(selection.member_positions(), dtype=np.float64)
    )
    if len(cluster_indexes) == 0:
        raise click.UsageError(
            f"{matches_path} holds no cluster observations to vote on"
        )
    return {
        "image_names": image_names,
        "width": width,
        "height": height,
        "cluster_indexes": cluster_indexes,
        "image_indexes": image_indexes,
        "positions": positions,
        "cluster_count": len(starts) - 1,
    }


def _diagonal_fov_deg(
    camera_model: str | None, focal_px: float | None, width: int, height: int
) -> float | None:
    """Diagonal field of view implied by a focal under the verdict's ray map.

    Both models are evaluated at the same corner radius `hypot(w, h) / 2`: the
    pinhole map opens as `2 atan(r / f)`, the equidistant one as `2 r / f`.
    """
    if focal_px is None or focal_px <= 0.0 or camera_model is None:
        return None
    r_corner = math.hypot(width, height) / 2.0
    if camera_model == _PINHOLE:
        return math.degrees(2.0 * math.atan(r_corner / focal_px))
    if camera_model == _EQUIDISTANT:
        return math.degrees(2.0 * r_corner / focal_px)
    return None


def _num(value, digits: int = 4) -> str:
    """A float for the report, or `n/a` when the vote produced none."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _report_lines(estimate: dict, data: dict) -> list[str]:
    """The human-readable report: the answer first, the diagnostics under it."""
    width, height = data["width"], data["height"]
    result = estimate["vote"]
    verdict = estimate["camera_model"]
    focal = estimate["focal_px"]
    confirmed = estimate["confirmed"]

    lines = [
        f"Images:        {len(data['image_names'])} @ {width}x{height}",
        f"Clusters:      {data['cluster_count']} "
        f"({len(data['cluster_indexes'])} observations)",
        "",
    ]

    if verdict is None:
        lines.append("Camera model:  no verdict (no column carried any vote)")
    else:
        marker = ""
        if confirmed is True:
            marker = "  CONFIRMED"
        elif confirmed is False:
            marker = "  UNCONFIRMED"
        stored = _CAMRIG_MODEL.get(verdict, verdict)
        lines.append(f"Camera model:  {verdict} ({stored}){marker}")

    if focal is None:
        lines.append(
            f"Focal length:  no consensus ({result['n_pool']} pooled vote(s); "
            "2 are needed)"
        )
    else:
        lines.append(f"Focal length:  {focal:.2f} px")
        fov = _diagonal_fov_deg(verdict, focal, width, height)
        if fov is not None:
            lines.append(f"Diagonal FOV:  {fov:.1f} deg")

    if confirmed is False:
        lines += [
            "",
            "The Fisheye verdict carries no certified rotation votes, so it is "
            "not corroborated;",
            "treat the capture as pinhole and re-run with `--model pinhole` "
            "for its focal.",
        ]

    lines += [
        "",
        "Votes:",
        f"  epipolar {result['n_epipolar']}, rotation {result['n_rotation']}, "
        f"pooled {result['n_pool']} (majority family: "
        f"{result['family'] or 'none'})",
        f"  pool spread (log IQR):  {_num(result['pool_spread'])}",
        f"  family disagreement:    {_num(result['family_disagreement'])}",
    ]

    if focal is None:
        lines += [
            "  rejections:  "
            f"h-dominated {result['n_h_dominated']}, "
            f"estimator failed {result['n_estimator_failed']}, "
            f"out of band {result['n_band_rejected']}",
        ]

    if result["columns"]:
        lines += ["", "Columns:"]
        for column in result["columns"]:
            lines.append(
                f"  {column['camera_model']:<20}"
                f"focal {_num(column['focal_px'], 2):>8} px  "
                f"spread {_num(column['pool_spread'])}  "
                f"certified epi/rot {column['n_certified_epipolar']}/"
                f"{column['n_certified_rotation']}"
            )
    return lines


def _derive_pattern(image_names: list[str]) -> str:
    """The matches' common image directory plus one `*` per level below it.

    The image table is POSIX workspace-relative, so the common prefix is taken
    on path segments -- never mid-name. A glob's `*` does not cross `/`, so
    names sitting a uniform depth below the common directory (a rig's
    per-sensor subdirectories, say) get one `*` segment per level. The final
    glob keeps the extension the matches' names actually carry, so a flat
    layout does not sweep up the workspace's own files; a mix of extensions
    falls back to a bare `*`.
    """
    import os.path

    parents = {Path(name).parent.as_posix() for name in image_names}
    if len(parents) == 1:
        common = parents.pop()
    else:
        common = Path(os.path.commonpath(sorted(parents))).as_posix()
    if common in ("", "."):
        common = ""
    exts = {Path(name).suffix for name in image_names}
    glob = f"*{exts.pop()}" if len(exts) == 1 and "" not in exts else "*"
    common_depth = len(Path(common).parts) if common else 0
    depths = {len(Path(name).parts) - common_depth for name in image_names}
    # One "*" per directory level between the common dir and the file names;
    # mixed depths cannot be expressed as one glob, so they keep the single
    # level and let validation hand the caller a message naming --pattern.
    levels = depths.pop() if len(depths) == 1 else 1
    segments = ["*"] * max(levels - 1, 0) + [glob]
    return "/".join(([common] if common else []) + segments)


def _write_camrig(
    output_path: Path,
    pattern: str | None,
    estimate: dict,
    data: dict,
    force: bool,
) -> str:
    """Commit the estimate as a one-sensor `.camrig`; returns a summary line.

    Raises `click.ClickException` for every refusal, so the caller can print
    the report first and still exit nonzero.
    """
    import numpy as np

    from ..camrig.create import CamrigCreateError, find_images, normalize_pattern
    from .._sfmtool.io import write_camrig

    verdict = estimate["camera_model"]
    focal = estimate["focal_px"]
    if focal is None or verdict not in _CAMRIG_MODEL:
        raise click.ClickException(
            "no focal consensus to write: the vote produced no shared focal "
            "for this capture."
        )
    if estimate["confirmed"] is False:
        raise click.ClickException(
            "refusing to write an UNCONFIRMED Fisheye estimate. Re-run with "
            "`--model pinhole` to commit the pinhole estimate instead."
        )
    if output_path.exists() and not force:
        raise click.ClickException(
            f"{output_path} already exists; pass --force to overwrite it."
        )

    # The `.camrig` pattern is resolved against the rig root, which the format
    # defines as the file's own directory.
    rig_root = output_path.parent
    try:
        stored_pattern = normalize_pattern(
            pattern if pattern else _derive_pattern(data["image_names"])
        )
        find_images(rig_root, stored_pattern)
    except CamrigCreateError as e:
        raise click.ClickException(
            f"{e}\nPass --pattern to name the images this rig describes, "
            f"relative to {rig_root}."
        ) from None

    camera = {
        "model": _CAMRIG_MODEL[verdict],
        "width": data["width"],
        "height": data["height"],
        "parameters": {
            "focal_length": float(focal),
            "principal_point_x": data["width"] / 2.0,
            "principal_point_y": data["height"] / 2.0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_camrig(
        path=str(output_path),
        name=output_path.stem,
        rig_type="generic",
        cameras=[camera],
        sensor_image_patterns=[stored_pattern],
        camera_indexes=[0],
        quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        translations_xyz=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
    )
    return (
        f"Wrote {output_path}\n"
        f"  camera:   {camera['model']} {data['width']}x{data['height']}\n"
        f"  focal:    {focal:.2f} px\n"
        f"  pattern:  {stored_pattern}"
    )


@click.command("estimate-intrinsics")
@click.help_option("--help", "-h")
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Cluster-bearing .matches file (from sfm match --cluster).",
)
@click.option(
    "--model",
    "model_option",
    type=click.Choice(["auto", "pinhole", "fisheye"], case_sensitive=False),
    default="auto",
    show_default=True,
    help=(
        "Camera-model columns to run: `auto` runs both and arbitrates between "
        "them; the named forms run one column and skip arbitration."
    ),
)
@click.option(
    "--write-camrig",
    "camrig_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write the estimate as a one-sensor .camrig at this path.",
)
@click.option(
    "--pattern",
    default=None,
    help=(
        "Image pattern stored in the .camrig; defaults to the matches' common "
        "image directory plus one '*' segment per directory level below it, "
        "ending '*<ext>' for the extension the image names share."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Allow --write-camrig to overwrite an existing file.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit the full vote result as JSON on stdout instead of the report.",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="RANSAC / pair-sampling seed; same inputs and seed give the same output.",
)
def estimate_intrinsics(
    input_path,
    model_option,
    camrig_path,
    pattern,
    force,
    as_json,
    seed,
):
    """Estimate a shared focal length and camera model from cluster matches.

    Image pairs drawn from the cluster tracks each vote through whichever
    estimator their geometry can observe, and the pooled log-median is the
    focal consensus. No structure is estimated, so the answer cannot be biased
    by the depth/focal compensation of structure-based estimation. Under
    `--model auto` the pinhole and equidistant-fisheye columns are both run and
    arbitrated on their certified mass of model-informative scan votes.

    The report is the product: a capture with no consensus still exits 0.
    Passing --write-camrig also commits the estimate as a one-sensor .camrig,
    which `sfm solve` picks up as the intrinsics prior for the images its
    pattern matches.

    \b
    Example:
        sfm estimate-intrinsics -i matches/clusters.matches
        sfm estimate-intrinsics -i matches/clusters.matches \\
            --write-camrig images.camrig
    """
    import json as json_module

    from .._sfmtool.geometry import estimate_intrinsics as estimate

    try:
        data = _load_observations(Path(input_path))
        result = estimate(
            data["cluster_indexes"],
            data["image_indexes"],
            data["positions"],
            data["width"],
            data["height"],
            seed=seed,
            columns=list(_MODEL_COLUMNS[model_option.lower()]),
        )
    except click.UsageError:
        raise
    except Exception as e:
        raise click.ClickException(str(e)) from None

    if as_json:
        # The vote's own dict stays at the top level, where it has always been;
        # the estimate's verdict fields are the same values the vote reports, so
        # nesting the vote under "vote" as well would duplicate every key. What
        # the estimate adds beyond the vote is what gets added here.
        vote = result["vote"]
        payload = dict(vote)
        payload["fisheye_confirmed"] = result["confirmed"]
        equidistant = next(
            (c for c in vote["columns"] if c["camera_model"] == _EQUIDISTANT), None
        )
        payload["certified_rotation_mass"] = (
            None if equidistant is None else int(equidistant["n_certified_rotation"])
        )
        payload["diagonal_fov_deg"] = _diagonal_fov_deg(
            result["camera_model"], result["focal_px"], data["width"], data["height"]
        )
        # The evidence behind THIS verdict, unlike the flat vote lists above,
        # which always describe the pinhole closed-form kernel.
        payload["verdict_votes"] = result["verdict_votes"]
        click.echo(json_module.dumps(payload, indent=2))
    else:
        click.echo("\n".join(_report_lines(result, data)))

    if camrig_path is not None:
        summary = _write_camrig(Path(camrig_path), pattern, result, data, force)
        click.echo("")
        click.echo(summary)
