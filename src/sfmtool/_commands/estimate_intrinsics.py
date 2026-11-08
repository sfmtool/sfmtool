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
from dataclasses import dataclass
from pathlib import Path

import click

from ._range_options import matches_image_filter, parse_image_selection, range_option

# The extensions a named image list's members must carry to be selected, the
# same test `sfm solve` applies to a list given with a `.matches` file.
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# `--model` -> the `columns` the kernel evaluates. The named forms run a single
# column, so the binding reports that column's own consensus and no arbitration
# takes place. `auto` hands the choice to the kernel: it votes pinhole-only and
# pays for the camera-model columns only when that vote comes back weak.
_MODEL_COLUMNS = {
    "auto": "auto",
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
            "the selected images of this .matches file have more than one "
            "resolution, and the vote estimates ONE shared camera:\n"
            + "\n".join(lines)
            + "\nSelect one camera's images with -r, or split the images into "
            "per-camera .matches files, and estimate each separately."
        )
    return int(distinct[0][0]), int(distinct[0][1])


def _open_matches(matches_path: Path, image_selection) -> tuple:
    """The opened `.matches` file and the images `-r` keeps of it.

    Returns `(mfile, file_image_names, kept, label, workspace_dir)`: `kept`
    is the names `matches_image_filter` keeps, in the file's order, and
    `label` says what selected them; both are `None` without `-r`.
    `workspace_dir` is the workspace the file resolves to, or `None`.
    """
    from .._sfmtool.fileio import MatchesFile

    mfile = MatchesFile(matches_path)
    if not mfile.has_clusters:
        raise click.UsageError(
            f"{matches_path} has no clusters section; run `sfm match --cluster` "
            "to produce a cluster-bearing .matches file"
        )
    file_image_names = list(mfile.image_names)
    workspace_dir = None if mfile.workspace_dir is None else Path(mfile.workspace_dir)
    image_filter = matches_image_filter(
        image_selection,
        matches_path,
        extensions=_IMAGE_EXTENSIONS,
        echo=lambda line: click.echo(line, err=True),
    )
    if image_filter is None:
        return mfile, file_image_names, None, None, workspace_dir
    if workspace_dir is None:
        raise click.ClickException(
            f"Cannot resolve workspace for {matches_path}. "
            "Ensure the workspace exists and contains .sfm-workspace.json."
        )
    kept = image_filter.keep(workspace_dir, file_image_names)
    return (
        mfile,
        file_image_names,
        [file_image_names[i] for i in kept],
        image_filter.label,
        workspace_dir,
    )


def _select(mfile, restrict: list[str] | None, data: dict) -> tuple:
    """The selection the vote runs on -- every cluster the file admits, or
    the clusters restricted to `restrict` -- and `data` completed with what
    the report and the `.camrig` need: the selected names, the shared image
    size and the cluster and observation counts."""
    import numpy as np

    if restrict is None:
        selection = mfile.select_clusters()
    else:
        selection = mfile.select_clusters(restrict_images=restrict)
    image_names = list(selection.image_names)
    # The kernel enforces the shared-camera rule too; resolving it here first
    # is what turns it into a message naming the offending resolutions.
    width, height = _resolve_dimensions(selection.image_dims, image_names)
    starts = np.asarray(selection.cluster_starts, dtype=np.int64)
    return selection, {
        **data,
        "image_names": image_names,
        "width": width,
        "height": height,
        "cluster_count": max(len(starts) - 1, 0),
        "observation_count": int(starts[-1]) if len(starts) else 0,
    }


def _load_selection(matches_path: Path, image_selection) -> tuple:
    """The selection the vote runs on, and what the report says about it.

    Without `-r` every cluster the file admits votes, on all its images, under
    the format's own member admission. With `-r` the images are those
    `matches_image_filter` keeps -- the same resolution `sfm solve` gives a
    range or a named image list on a `.matches` file -- and the selection is
    restricted to them, so only members on those images vote and a cluster
    left spanning fewer than two of them drops. The selection handle goes
    straight to the kernel, which reads the backbone's own CSR arrays and
    member positions off it -- so nothing is re-derived here beyond the
    counts and the image size the report and the `.camrig` need.
    """
    mfile, file_image_names, kept, label, workspace_dir = _open_matches(
        matches_path, image_selection
    )
    selection, data = _select(
        mfile,
        kept,
        {
            "matches_path": matches_path,
            "file_image_names": file_image_names,
            "selection_label": label,
            "workspace_dir": None if kept is None else workspace_dir,
        },
    )
    if kept is None and data["observation_count"] == 0:
        raise click.UsageError(
            f"{matches_path} holds no cluster observations to vote on"
        )
    return selection, data


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


def _json_payload(result: dict, data: dict) -> dict:
    """The `--json` payload of one estimate."""
    # The vote's own dict stays at the top level, where it has always been;
    # the estimate's verdict fields are the same values the vote reports, so
    # nesting the vote under "vote" as well would duplicate every key. What
    # the estimate adds beyond the vote is what gets added here.
    vote = result["vote"]
    payload = dict(vote)
    payload["image_count"] = len(data["file_image_names"])
    payload["selected_image_count"] = len(data["image_names"])
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
    # What `--model auto` decided: the weak-vote reasons that made the
    # camera-model columns worth running, `[]` when they were not run, and
    # null when the columns were named outright.
    payload["escalation"] = result["escalation"]
    # The pinhole-only vote that decision was read off, kept when the
    # escalation fired and null otherwise. The top-level vote keys describe
    # the escalated two-column run, so a caller's pinhole consensus, pool
    # and spread are read here.
    payload["screening_vote"] = result["screening_vote"]
    # The evidence behind THIS verdict, unlike the flat vote lists above,
    # which always describe the pinhole closed-form kernel.
    payload["verdict_votes"] = result["verdict_votes"]
    return payload


def _report_lines(estimate: dict, data: dict) -> list[str]:
    """The human-readable report: the answer first, the diagnostics under it."""
    width, height = data["width"], data["height"]
    result = estimate["vote"]
    verdict = estimate["camera_model"]
    focal = estimate["focal_px"]
    confirmed = estimate["confirmed"]

    n_images = len(data["image_names"])
    if data["selection_label"] is None:
        images = f"{n_images}"
    else:
        images = (
            f"{n_images} of {len(data['file_image_names'])} selected "
            f"({data['selection_label']})"
        )
    lines = [
        f"Images:        {images} @ {width}x{height}",
        f"Clusters:      {data['cluster_count']} "
        f"({data['observation_count']} observations)",
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

    # Whether the camera-model columns ran at all is the kernel's decision under
    # `--model auto`, so the report states it rather than leaving the reader to
    # infer it from a missing Columns block.
    escalation = estimate["escalation"]
    if escalation is not None:
        if escalation:
            lines += [
                "",
                "Arbitration:   the camera-model columns ran, on a weak pinhole "
                f"vote ({', '.join(escalation)})",
            ]
        else:
            lines += [
                "",
                "Arbitration:   not run -- the pinhole-only vote stood on its "
                "own, leaving the camera-model columns nothing to overturn",
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


def _derive_pattern(image_names: list[str], *, name_prefix: bool = False) -> str:
    """The images' common directory plus one `*` per level below it.

    The image table is POSIX workspace-relative, so the common prefix is taken
    on path segments -- never mid-name. A glob's `*` does not cross `/`, so
    names sitting a uniform depth below the common directory (a rig's
    per-sensor subdirectories, say) get one `*` segment per level. The final
    glob keeps the extension the names actually carry, so a flat layout does
    not sweep up the workspace's own files; a mix of extensions falls back to
    a bare `*`. With `name_prefix` the final glob also starts with the common
    prefix of the names' stems (their file names less the extension), cut
    before its first `*`, `?` or `[`, less any digits it then ends in, and
    with each `%` written `%%`: so `shopA_00001.jpg` ... `shopA_01089.jpg`
    give `shopA_*.jpg` rather than a pattern fixing part of the frame number.
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
    if name_prefix:
        prefix = os.path.commonprefix([Path(name).stem for name in image_names])
        # The prefix is literal text, so it stops before the first character
        # a pattern cannot hold literally: `*` is always a wildcard, and `?`
        # and `[` are read as wildcards by the filesystem walk that expands a
        # pattern (`camrig/pattern.py` globs it with pathlib).
        for i, c in enumerate(prefix):
            if c in "*?[":
                prefix = prefix[:i]
                break
        # A literal `%` is written `%%`, so it is not read as a frame field.
        glob = prefix.rstrip("0123456789").replace("%", "%%") + glob
    common_depth = len(Path(common).parts) if common else 0
    depths = {len(Path(name).parts) - common_depth for name in image_names}
    # One "*" per directory level between the common dir and the file names;
    # mixed depths cannot be expressed as one glob, so they keep the single
    # level and let validation hand the caller a message naming --pattern.
    levels = depths.pop() if len(depths) == 1 else 1
    segments = ["*"] * max(levels - 1, 0) + [glob]
    return "/".join(([common] if common else []) + segments)


def _covered_matches(
    rig_root: Path, found: list[str], names: list[str], workspace_dir: Path
) -> list[str]:
    """Those of `names`, workspace-relative images of the `.matches` file,
    that `found`, paths relative to `rig_root`, holds, as the file names
    them."""
    import os.path

    if not names:
        return []

    def key(path: Path) -> str:
        return os.path.normcase(str(path.resolve()))

    by_path = {key(Path(workspace_dir) / name): name for name in names}
    root = rig_root.resolve()
    return [by_path[k] for p in found if (k := key(root / p)) in by_path]


def _left_out_matches(rig_root: Path, found: list[str], data: dict) -> list[str]:
    """The `.matches` file's images that `-r` left out of the estimate and
    `found`, paths relative to `rig_root`, holds, as the file names them."""
    selected = set(data["image_names"])
    left_out = [n for n in data["file_image_names"] if n not in selected]
    if not left_out:
        return []
    return _covered_matches(rig_root, found, left_out, data["workspace_dir"])


def _unwritable(estimate: dict) -> str | None:
    """Why the estimate cannot be committed to a `.camrig`, or `None`."""
    if estimate["focal_px"] is None or estimate["camera_model"] not in _CAMRIG_MODEL:
        return (
            "no focal consensus to write: the vote produced no shared focal "
            "for these images."
        )
    if estimate["confirmed"] is False:
        return (
            "refusing to write an UNCONFIRMED Fisheye estimate. Re-run with "
            "`--model pinhole` to commit the pinhole estimate instead."
        )
    return None


def _store_camrig(
    output_path: Path, stored_pattern: str, estimate: dict, data: dict
) -> str:
    """Write the estimate as a one-sensor `.camrig` storing `stored_pattern`;
    returns a summary line."""
    import numpy as np

    from .._sfmtool.fileio import write_camrig

    focal = estimate["focal_px"]
    camera = {
        "model": _CAMRIG_MODEL[estimate["camera_model"]],
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
    from ..camrig.create import CamrigCreateError, find_images, normalize_pattern

    reason = _unwritable(estimate)
    if reason is not None:
        raise click.ClickException(reason)
    if output_path.exists() and not force:
        raise click.ClickException(
            f"{output_path} already exists; pass --force to overwrite it."
        )

    # The `.camrig` pattern is resolved against the rig root, which the format
    # defines as the file's own directory. A derived pattern that would also
    # cover images of the file that `-r` left out is narrowed to the selected
    # names' common prefix; a pattern that still covers one is refused, since
    # the `.camrig` would give those images an estimate they took no part in.
    rig_root = output_path.parent
    if pattern:
        candidates = [pattern]
    else:
        candidates = [_derive_pattern(data["image_names"])]
        narrowed = _derive_pattern(data["image_names"], name_prefix=True)
        if narrowed != candidates[0]:
            candidates.append(narrowed)
    for candidate in candidates:
        try:
            stored_pattern = normalize_pattern(candidate)
            found = find_images(rig_root, stored_pattern)
        except CamrigCreateError as e:
            raise click.ClickException(
                f"{e}\nPass --pattern to name the images this rig describes, "
                f"relative to {rig_root}."
            ) from None
        left_out = _left_out_matches(rig_root, found, data)
        if not left_out:
            break
    else:
        raise click.ClickException(
            f"pattern {stored_pattern!r} matches {len(left_out)} image(s) of "
            f"{data['matches_path']} that -r left out of the estimate, such as "
            f"{left_out[0]}, and the .camrig would give them this camera.\n"
            "Pass --pattern to name only the selected images, relative to "
            f"{rig_root}."
        )
    return _store_camrig(output_path, stored_pattern, estimate, data)


# The characters stripped from the ends of the parts of a sequence's file
# names that name its `.camrig`: the separators that commonly stand between a
# name and its file number (`shopA_00001`, `shopA-00001`, `shopA.00001`,
# `shopA 00001`, `shopA (1)`), and the `.` of the extension after it.
_SEPARATORS = " _-.()"


@dataclass
class _SequenceRun:
    """One sequence of the `.matches` file's images in a
    `--write-camrig-per-sequence` run: what it is, what the vote said and
    what became of its `.camrig`."""

    # The sequence's list pattern, workspace-relative, as `sfm pick-images`
    # writes it; a rig sequence's patterns joined by " and ".
    pattern: str
    # The file's images in the sequence, and those `-r` keeps.
    names: list[str]
    chosen: list[str]
    # The folder the `.camrig` goes in, and the parts of the file names
    # before and after the file number that name it, separators stripped
    # (the part after is the extension, behind a `)` for a number in
    # parentheses); `None` for a rig sequence.
    camrig_folder: Path | None = None
    name_before: str = ""
    name_after: str = ""
    # The `.camrig` path and the pattern it stores; `None` for a rig sequence.
    camrig_path: Path | None = None
    camrig_pattern: str | None = None
    data: dict | None = None
    estimate: dict | None = None
    # Why the sequence gets no `.camrig`, or the summary of the one written.
    reason: str | None = None
    summary: str | None = None


def _name_key(rel: str) -> tuple[str, str, str] | None:
    """The folder of a workspace-relative image name and the parts of its
    file name before and after the file number, which are what
    `collect_sequences` groups a sequence of single images by; `None` for a
    name with no file number."""
    from .._filenames import file_number_span

    folder, _slash, name = rel.rpartition("/")
    span = file_number_span(name)
    if span is None:
        return None
    return folder, name[: span[0]], name[span[1] :]


def _sequence_run(
    sequence, names: list[str], chosen: list[str], workspace_dir: Path
) -> _SequenceRun:
    """The planned run of a sequence of single images: the folder its
    `.camrig` goes in, the name parts the file is named from, and the
    pattern it stores, relative to that folder.

    The pattern is the sequence's list pattern less its folder: the names
    with the file number as a frame field, and `%%` for a `%`. The list
    pattern is the escaped folder and a `/`, then the escaped name pattern;
    a file name holds no `/`, so the name pattern is what follows the last
    one.
    """
    folder, before, after = _name_key(Path(sequence.images[0][0]).as_posix())
    return _SequenceRun(
        pattern=sequence.name,
        names=names,
        chosen=chosen,
        camrig_folder=workspace_dir / folder,
        name_before=before.rstrip(_SEPARATORS),
        name_after=after.strip(_SEPARATORS),
        camrig_pattern=sequence.name.rpartition("/")[2],
    )


def _plan_sequences(matches_path: Path, image_selection) -> tuple:
    """The sequences of the images `-r` keeps of the `.matches` file, and the
    opened file: `(mfile, runs, file_image_names, kept, label,
    workspace_dir)`.

    Only the kept images are grouped, as `sfm pick-images` groups them
    (`collect_sequences`, with the rigs `find_rigs` finds), so an image `-r`
    leaves out cannot stop the run. A sequence of a multi-sensor `.camrig` is
    planned with the reason it gets no `.camrig`; each other sequence with
    the name parts and the pattern of its `.camrig`, and with the file's
    images left out by `-r` whose names differ from its own only in the file
    number. An image of a rig of `rig_config.json` anywhere in the file is
    refused.
    """
    from .._pick_images import PickImagesInputError, collect_sequences, find_rigs
    from ..rig.config import _match_image_to_sensor

    mfile, file_image_names, kept, label, workspace_dir = _open_matches(
        matches_path, image_selection
    )
    if workspace_dir is None:
        raise click.ClickException(
            f"Cannot resolve workspace for {matches_path}. "
            "Ensure the workspace exists and contains .sfm-workspace.json; "
            "--write-camrig-per-sequence places each .camrig in its "
            "sequence's folder of that workspace."
        )
    chosen_names = file_image_names if kept is None else kept
    try:
        rig_configs, camrig_rigs, _notes = find_rigs(
            [workspace_dir / name for name in file_image_names], workspace_dir
        )
        if rig_configs is not None:
            for name in file_image_names:
                if (
                    _match_image_to_sensor(Path(name).as_posix(), rig_configs)
                    is not None
                ):
                    # `sfm solve` ignores rig_config.json once a `.camrig`
                    # covers any image it solves, so a written `.camrig`
                    # would undo the rig in a solve of the file, whatever
                    # `-r` selects here.
                    raise click.ClickException(
                        f"the image {name} belongs to a rig of rig_config.json, "
                        "and `sfm solve` ignores rig_config.json once a .camrig "
                        "covers any image it solves.\n"
                        "Give the rig a multi-sensor .camrig first."
                    )
        sequences = collect_sequences(
            [workspace_dir / name for name in chosen_names],
            workspace_dir,
            rig_configs,
            camrig_rigs,
        )
    except PickImagesInputError as e:
        raise click.ClickException(
            f"the images of {matches_path} cannot be grouped into sequences:\n{e}"
        ) from None

    by_rel = {Path(name).as_posix(): name for name in file_image_names}
    chosen_set = set(chosen_names)
    # The file's images `-r` left out, by the name parts that would put them
    # in a sequence of the kept images.
    left_out: dict[tuple, list[str]] = {}
    for name in file_image_names:
        if name not in chosen_set:
            key = _name_key(Path(name).as_posix())
            if key is not None:
                left_out.setdefault(key, []).append(name)

    runs: list[_SequenceRun] = []
    for sequence in sequences:
        chosen = [by_rel[p.as_posix()] for frame in sequence.images for p in frame]
        if not chosen:
            # A rig `.camrig` none of whose images `-r` keeps still gives
            # `collect_sequences` a rig sequence, of no frames.
            continue
        if sequence.sensors is not None:
            runs.append(
                _SequenceRun(
                    pattern=" and ".join(sequence.patterns),
                    names=chosen,
                    chosen=chosen,
                    reason=(
                        "a rig sequence: the multi-sensor .camrig that covers "
                        "it gives its images their cameras."
                    ),
                )
            )
            continue
        key = _name_key(Path(sequence.images[0][0]).as_posix())
        names = chosen + left_out.get(key, [])
        runs.append(_sequence_run(sequence, names, chosen, workspace_dir))
    return mfile, runs, file_image_names, kept, label, workspace_dir


def _shared_paths(runs: list[_SequenceRun]) -> list[list[_SequenceRun]]:
    """The groups of sequences that share a `.camrig` path. Paths are
    compared without regard to letter case, since the file systems of
    Windows and macOS do not tell `x.camrig` from `X.camrig`."""
    by_path: dict[str, list[_SequenceRun]] = {}
    for run in runs:
        if run.camrig_path is not None:
            key = str(run.camrig_path.resolve()).casefold()
            by_path.setdefault(key, []).append(run)
    return [shared for shared in by_path.values() if len(shared) > 1]


def _name_camrigs(runs: list[_SequenceRun]) -> None:
    """Give each sequence of single images its `.camrig` path, or the reason
    it gets none.

    A file is named from the part of the names before the file number, or
    the folder's name when that part is empty. Sequences that would share a
    file are named instead from that name, `_`, and the part of the names
    after the number, when that part is not empty. `number_from_filename`
    reads the digits at the end of a name, or in parentheses there, so the
    part after the number is the extension, behind a `)` for a number in
    parentheses: `x_01.jpg` and `x_01.png` give `x_jpg.camrig` and
    `x_png.camrig`. The sequences that still share a file get none.
    """

    def path(run: _SequenceRun, after: bool) -> Path:
        stem = run.name_before or run.camrig_folder.resolve().name
        if after and run.name_after:
            stem = f"{stem}_{run.name_after}"
        return run.camrig_folder / f"{stem}.camrig"

    for run in runs:
        if run.camrig_folder is not None:
            run.camrig_path = path(run, after=False)
    for shared in _shared_paths(runs):
        for run in shared:
            run.camrig_path = path(run, after=True)
    for shared in _shared_paths(runs):
        for run in shared:
            others = " and ".join(o.pattern for o in shared if o is not run)
            run.reason = (
                f"{run.camrig_path} would also be the .camrig of {others}, "
                "even named with the part of the file names after the number. "
                "Write each sequence's .camrig with -r and --write-camrig."
            )


def _finish_sequence(
    run: _SequenceRun,
    file_image_names: list[str],
    workspace_dir: Path,
    camrig_hits: list,
    matches_path: Path,
    force: bool,
) -> None:
    """Write the sequence's `.camrig`, or set the reason it gets none."""
    from ..camrig.create import CamrigCreateError, find_images, normalize_pattern

    if run.reason is None:
        run.reason = _unwritable(run.estimate)
    if run.reason is not None:
        return
    own = {(workspace_dir / name).resolve() for name in run.names}
    target = run.camrig_path.resolve()
    # A rig's `.camrig` counts too: its pattern can cover a frame that `-r`
    # left out of the rig's selection and that the name puts in this sequence.
    for camrig_file, _data, covered in camrig_hits:
        if camrig_file.resolve() != target and covered & own:
            run.reason = (
                f"{camrig_file} already covers {len(covered & own)} of its "
                "images, and `sfm solve` refuses an image two .camrig files "
                "match."
            )
            return
    if run.camrig_path.exists() and not force:
        run.reason = f"{run.camrig_path} already exists; pass --force to overwrite it."
        return
    rig_root = run.camrig_path.parent
    try:
        stored_pattern = normalize_pattern(run.camrig_pattern)
        found = find_images(rig_root, stored_pattern)
    except CamrigCreateError as e:
        run.reason = str(e)
        return
    names = set(run.names)
    others = [name for name in file_image_names if name not in names]
    covered = _covered_matches(rig_root, found, others, workspace_dir)
    if covered:
        run.reason = (
            f"pattern {stored_pattern!r} matches {len(covered)} image(s) of "
            f"{matches_path} in other sequences, such as {covered[0]}, and the "
            ".camrig would give them this camera."
        )
        return
    run.summary = _store_camrig(run.camrig_path, stored_pattern, run.estimate, run.data)


def _sequence_json(run: _SequenceRun, workspace_dir: Path) -> dict:
    """The `--json` entry of one sequence."""
    return {
        "pattern": run.pattern,
        "image_count": len(run.names),
        "selected_image_count": len(run.chosen),
        "camrig_path": (
            None
            if run.camrig_path is None
            else run.camrig_path.relative_to(workspace_dir).as_posix()
        ),
        "camrig_pattern": run.camrig_pattern,
        "written": run.summary is not None,
        "reason": run.reason,
        "estimate": (
            None if run.estimate is None else _json_payload(run.estimate, run.data)
        ),
    }


def _write_camrig_per_sequence(
    matches_path: Path,
    image_selection,
    vote_options: dict,
    force: bool,
    as_json: bool,
) -> None:
    """Estimate each sequence of the `.matches` file's images from its own
    images, write its `.camrig`, and print the report or the JSON.

    Raises `click.ClickException` after the output when no `.camrig` was
    written.
    """
    import json as json_module

    from .._sfmtool.geometry import estimate_intrinsics as estimate
    from ..camrig.resolver import CamrigSolveError, find_covering_camrigs

    mfile, runs, file_image_names, kept, label, workspace_dir = _plan_sequences(
        matches_path, image_selection
    )
    try:
        hits = find_covering_camrigs(
            [workspace_dir / name for name in file_image_names],
            workspace_dir,
            images=f"images of {matches_path}",
        )
    except CamrigSolveError as e:
        raise click.ClickException(str(e)) from None

    _name_camrigs(runs)
    for run in runs:
        if run.camrig_path is None:
            continue
        base = {
            "matches_path": matches_path,
            "file_image_names": run.names,
            "selection_label": label,
            "workspace_dir": workspace_dir,
        }
        try:
            selection, run.data = _select(mfile, run.chosen, base)
        except click.ClickException as e:
            # Images of more than one resolution: no vote, and no file.
            run.reason = e.message
            continue
        try:
            run.estimate = estimate(selection, **vote_options)
        except Exception as e:
            raise click.ClickException(str(e)) from None
        _finish_sequence(
            run,
            file_image_names,
            workspace_dir,
            hits,
            matches_path,
            force,
        )

    if as_json:
        payload = {
            "workspace_dir": str(workspace_dir),
            "image_count": len(file_image_names),
            "selected_image_count": len(file_image_names if kept is None else kept),
            "sequences": [_sequence_json(run, workspace_dir) for run in runs],
        }
        click.echo(json_module.dumps(payload, indent=2))

    noun = "sequence" if len(runs) == 1 else "sequences"
    lines = [f"{len(runs)} {noun} of {matches_path}:"]
    for run in runs:
        lines += ["", f"Sequence {run.pattern} ({len(run.names)} images)"]
        if not as_json and run.estimate is not None:
            report = _report_lines(run.estimate, run.data)
            lines += [f"  {line}" if line else "" for line in report]
            lines.append("")
        if run.summary is not None:
            lines += [f"  {line}" for line in run.summary.splitlines()]
        else:
            lines.append(f"  No .camrig: {run.reason}")
    # Under --json stdout holds the JSON alone, so the account of the files
    # goes to stderr.
    click.echo("\n".join(lines), err=as_json)
    if not any(run.summary is not None for run in runs):
        raise click.ClickException("no .camrig was written.")


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
@range_option(
    "A range expression of file numbers: estimate from only the .matches "
    "file's images whose file number it holds.",
    on="matches",
)
@click.option(
    "--model",
    "model_option",
    type=click.Choice(["auto", "pinhole", "fisheye"], case_sensitive=False),
    default="auto",
    show_default=True,
    help=(
        "Camera-model columns to run: `auto` votes pinhole-only and runs both "
        "columns, arbitrating between them, when that vote comes back weak; "
        "the named forms run one column and skip arbitration."
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
        "Image pattern stored in the .camrig; defaults to the selected images' "
        "common directory plus one '*' segment per directory level below it, "
        "ending '*<ext>' for the extension their names share, and narrowed "
        "to their names' common prefix ('<prefix>*<ext>') when it would "
        "otherwise match images of the file that -r left out."
    ),
)
@click.option(
    "--write-camrig-per-sequence",
    "per_sequence",
    is_flag=True,
    default=False,
    help=(
        "Write one one-sensor .camrig per image sequence of the file, each "
        "estimated from that sequence's images only, in the sequence's "
        "folder, named after the part of its file names before the number, "
        "or after the folder when that part is empty."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Allow --write-camrig or --write-camrig-per-sequence to overwrite "
        "existing files."
    ),
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
    help=(
        "RANSAC / pair-sampling seed; same inputs, seed and --draws give the "
        "same output."
    ),
)
@click.option(
    "--draws",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help=(
        "RANSAC draws per pair vote; a pair votes the log-space median of its "
        "draws when more than half of them vote."
    ),
)
def estimate_intrinsics(
    input_path,
    range_expr,
    named_image_list,
    model_option,
    camrig_path,
    pattern,
    per_sequence,
    force,
    as_json,
    seed,
    draws,
):
    """Estimate a shared focal length and camera model from cluster matches.

    Image pairs drawn from the cluster tracks each vote through whichever
    estimator their geometry can observe, and the pooled log-median is the
    focal consensus. No structure is estimated, so the answer cannot be biased
    by the depth/focal compensation of structure-based estimation. Under
    `--model auto` the pinhole vote runs first, and the equidistant-fisheye
    column is added -- and arbitrated against the pinhole one on their certified
    mass of model-informative scan votes -- only when that pinhole vote comes
    back weak. The report says which of the two happened, and why.

    -r selects the file's images to estimate from, as `sfm solve -r` selects
    them from a .matches file: only members on those images vote, so one
    .matches file over a whole capture can be estimated per camera.

    The report is the product: a capture with no consensus still exits 0.
    Passing --write-camrig also commits the estimate as a one-sensor .camrig,
    which `sfm solve` picks up as the intrinsics prior for the images its
    pattern matches.

    --write-camrig-per-sequence groups the file's images into sequences as
    `sfm pick-images` does (one folder's files whose names differ only in
    the file number), estimates each sequence from its own images, and
    writes its .camrig beside it: images/shopA_00001.jpg ... give
    images/shopA.camrig with the pattern shopA_%05d.jpg, and
    shopB/00001.jpg ... give shopB/shopB.camrig with the pattern %05d.jpg.
    A sequence the vote reaches no consensus on gets no file; the command
    fails only when no file is written.

    \b
    Example:
        sfm estimate-intrinsics -i matches/clusters.matches
        sfm estimate-intrinsics -i matches/clusters.matches \\
            --write-camrig images.camrig
        sfm estimate-intrinsics -i matches/clusters.matches -r video1 \\
            --write-camrig video1.camrig
        sfm estimate-intrinsics -i matches/clusters.matches \\
            --write-camrig-per-sequence
    """
    import json as json_module

    from .._sfmtool.geometry import estimate_intrinsics as estimate

    image_selection = parse_image_selection(range_expr, named_image_list)
    vote_options = {
        "seed": seed,
        "draws": draws,
        "columns": _MODEL_COLUMNS[model_option.lower()],
    }
    if per_sequence:
        for given, name in ((camrig_path, "--write-camrig"), (pattern, "--pattern")):
            if given is not None:
                raise click.UsageError(
                    f"--write-camrig-per-sequence cannot be combined with {name}."
                )
        _write_camrig_per_sequence(
            Path(input_path), image_selection, vote_options, force, as_json
        )
        return

    try:
        selection, data = _load_selection(Path(input_path), image_selection)
        result = estimate(selection, **vote_options)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e)) from None

    if as_json:
        click.echo(json_module.dumps(_json_payload(result, data), indent=2))
    else:
        click.echo("\n".join(_report_lines(result, data)))

    if camrig_path is not None:
        summary = _write_camrig(Path(camrig_path), pattern, result, data, force)
        # Under --json stdout holds the JSON alone, so the summary of the
        # write goes to stderr.
        click.echo("", err=as_json)
        click.echo(summary, err=as_json)
