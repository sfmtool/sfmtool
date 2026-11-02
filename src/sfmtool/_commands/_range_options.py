# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The `-r/--range` and `--named-image-list` options shared by eight commands.

`sfm sift`, `sfm match`, `sfm motion` and `sfm solve` pick images from image
directories; `sfm analyze --metrics`, `sfm panorama`, `sfm to-colmap-bin` and
`sfm to-nerfstudio` pick them from a reconstruction. Each takes `-r/--range`,
whose value is either a range expression of file numbers or the name of a
named image list of the workspace, and `--named-image-list NAME`, the long
form for a list. One name and one range together narrow the list. The options,
the rule that tells a name from a range, the errors and the resolution of a
list all live here so that the eight commands cannot drift apart.

A range on its own reaches each command as the string it always took
(`ImageSelection.range_expr`), so a command's handling of a range is its own
and is unchanged. A list is resolved here: against the workspace's files for
the image-directory commands (`resolve_list_images`), and against a
reconstruction's image names for the others
(`reconstruction_list_indices`).

The filter for `to-colmap-bin` and `to-nerfstudio` takes its own `echo`
callable rather than choosing one, because the two commands report through
different sinks (`print` and `click.echo`) and the point of this module is to
leave their output untouched.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import click

from .._filenames import has_extension, number_from_filename

RANGE_OR_NAME_SENTENCE = (
    "A --range value with a letter in it is read as a named image list; "
    "a range contains only digits, '-', ':' and ','."
)

_NAME_RULE = (
    "a name is one or more of the ASCII letters, the digits, '-', '_' and '.', "
    "with at least one letter"
)

_NAME_HELP = (
    " A value with a letter in it is instead the name of a named image list "
    "of the workspace (see --named-image-list); give one name and one range, "
    "as two -r options, to keep the list's members whose frame number is in "
    "the range."
)

_LIST_HELP = {
    "paths": (
        "Use the images of this named image list, defined in the workspace's "
        ".sfm-named-image-lists.json, in place of image paths. The workspace "
        "is found from the current directory. With -r RANGE, keep only the "
        "members whose frame number is in RANGE."
    ),
    "reconstruction": (
        "Use only the reconstruction's images that are members of this named "
        "image list, defined in the .sfm-named-image-lists.json of the "
        "reconstruction's workspace. With -r RANGE, keep only the members "
        "whose frame number is in RANGE."
    ),
}


@dataclass(frozen=True)
class ImageSelection:
    """What `-r/--range` and `--named-image-list` asked for.

    At most one range and at most one list name. With both, the range narrows
    the list rather than being applied on its own.
    """

    range_expr: str | None = None
    list_name: str | None = None

    @property
    def given(self) -> bool:
        return self.range_expr is not None or self.list_name is not None

    def describe_list(self) -> str:
        """The list, and the range narrowing it, as a message names them."""
        text = f"named image list '{self.list_name}'"
        if self.range_expr is not None:
            text += f" narrowed by the range '{self.range_expr}'"
        return text


def range_option(range_help: str, *, on: str) -> Callable[[Callable], Callable]:
    """Attach `-r/--range` and `--named-image-list` to a command.

    `range_help` is what the command's range does; the sentence about list
    names is added to it. `on` is `"paths"` for a command that picks images
    from image directories and `"reconstruction"` for one that picks them from
    a reconstruction. The command receives `range_expr`, a tuple of the `-r`
    values, and `named_image_list`, and turns them into an `ImageSelection`
    with `parse_image_selection`.
    """
    list_help = _LIST_HELP[on]

    def attach(f: Callable) -> Callable:
        f = click.option(
            "--named-image-list",
            "named_image_list",
            metavar="NAME",
            default=None,
            help=list_help,
        )(f)
        f = click.option(
            "--range",
            "-r",
            "range_expr",
            metavar="RANGE|NAME",
            multiple=True,
            help=range_help + _NAME_HELP,
        )(f)
        return f

    return attach


def _is_name(value: str) -> bool:
    return any(c.isalpha() for c in value)


def _check_name(value: str, option: str) -> str:
    from .._sfmtool.workspace import is_named_image_list_name

    if not is_named_image_list_name(value):
        raise click.UsageError(
            f"{option} '{value}' is not a valid named image list name: {_NAME_RULE}."
        )
    return value


def parse_image_selection(
    range_values: tuple[str, ...], named_image_list: str | None
) -> ImageSelection:
    """Sort the `-r` values and `--named-image-list` into a range and a name.

    A `-r` value with a letter in it is a list name, and any other value is a
    range. At most one of each may be given in all.
    """
    names: list[str] = []
    ranges: list[str] = []
    if named_image_list is not None:
        names.append(_check_name(named_image_list, "--named-image-list"))
    for value in range_values:
        if _is_name(value):
            names.append(_check_name(value, "--range"))
        else:
            ranges.append(value)
    if len(names) > 1:
        quoted = ", ".join(f"'{n}'" for n in names)
        raise click.UsageError(
            f"Give at most one named image list; got {quoted}. "
            "One name and one range may be given together, and the range then "
            "narrows the list. " + RANGE_OR_NAME_SENTENCE
        )
    if len(ranges) > 1:
        quoted = ", ".join(f"'{r}'" for r in ranges)
        raise click.UsageError(
            f"Give at most one range; got {quoted}. Write the frames as one "
            "range expression, such as '1-10,20-30'."
        )
    return ImageSelection(
        range_expr=ranges[0] if ranges else None,
        list_name=names[0] if names else None,
    )


def _no_workspace_error(where: str, name: str) -> click.ClickException:
    return click.ClickException(
        f"No workspace was found {where}, so no named image lists are defined "
        f"and none is called '{name}'. " + RANGE_OR_NAME_SENTENCE
    )


def _read_lists(workspace_dir: Path, name: str):
    """The workspace's lists, or the error for a workspace without a lists file."""
    from .._sfmtool.workspace import NAMED_IMAGE_LISTS_FILE_NAME, NamedImageLists

    try:
        lists = NamedImageLists.read(workspace_dir)
    except (ValueError, OSError) as e:
        raise click.ClickException(str(e))
    if lists is None:
        raise click.ClickException(
            f"The workspace {workspace_dir} has no named image lists, since it "
            f"has no {NAMED_IMAGE_LISTS_FILE_NAME}, so none is called '{name}'. "
            + RANGE_OR_NAME_SENTENCE
        )
    return lists


def _resolution_error(lists, name: str, e: Exception) -> click.ClickException:
    """`e` from a resolver, with the name-or-range sentence for an unknown name."""
    message = str(e)
    if name not in lists:
        message += ". " + RANGE_OR_NAME_SENTENCE
    return click.ClickException(message)


def _narrow(selection: ImageSelection):
    if selection.range_expr is None:
        return None
    from .._sfmtool.reconstruction import RangeExpr

    try:
        return RangeExpr(selection.range_expr)
    except ValueError as e:
        raise click.UsageError(f"--range '{selection.range_expr}': {e}")


@dataclass(frozen=True)
class ListImages:
    """The image files of a named image list, and the workspace it is in."""

    paths: list[Path]
    workspace_dir: Path


def resolve_list_images(
    selection: ImageSelection, *, extensions: tuple[str, ...]
) -> ListImages:
    """Resolve `selection`'s list against the files of the current workspace.

    The workspace is the one the upward search from the current directory
    finds. The images are the list's members, in the list's order, as
    absolute paths, keeping only the members with one of `extensions`, as the
    command's own directory expansion does; the others are skipped without a
    word, as a directory listing skips them.
    """
    from .._workspace import find_workspace_for_path

    name = selection.list_name
    assert name is not None
    cwd = Path.cwd()
    workspace_dir = find_workspace_for_path(cwd)
    if workspace_dir is None:
        raise _no_workspace_error(f"from the current directory {cwd}", name)
    lists = _read_lists(workspace_dir, name)
    narrow = _narrow(selection)
    try:
        members = lists.resolve(name, narrow=narrow)
    except (ValueError, OSError, InterruptedError) as e:
        raise _resolution_error(lists, name, e)

    root = Path(lists.workspace_dir).absolute()
    paths = [root / m.path for m in members]
    images = [p for p in paths if has_extension(p, extensions)]
    if not images:
        raise click.ClickException(
            f"{lists.path}: none of the {len(paths)} members of "
            f"{selection.describe_list()} is an image file "
            f"({', '.join(extensions)})."
        )
    return ListImages(paths=images, workspace_dir=root)


def reconstruction_list_indices(
    recon, selection: ImageSelection, sfmr_path: str | Path
) -> list[int]:
    """The indices of `recon`'s images that `selection`'s list holds.

    The list is resolved in the reconstruction's workspace against its image
    names, so the image files need not exist. The indices keep the
    reconstruction's order; members the reconstruction does not hold are
    ignored, and a list that leaves none of its images is an error naming the
    list.
    """
    name = selection.list_name
    assert name is not None
    workspace_dir = recon.workspace_dir
    if not workspace_dir:
        raise _no_workspace_error(f"for {sfmr_path}", name)
    lists = _read_lists(Path(workspace_dir), name)
    narrow = _narrow(selection)
    image_names = list(recon.image_names)
    try:
        members = lists.resolve_in_names(name, image_names, narrow=narrow)
    except ValueError as e:
        raise _resolution_error(lists, name, e)
    wanted = {m.path for m in members}
    return [i for i, image_name in enumerate(image_names) if image_name in wanted]


# ── The --range / --filter-points pair of the export commands ─────────────


def range_options(f: Callable) -> Callable:
    """Attach `-r/--range`, `--named-image-list` and `--filter-points`."""
    f = click.option(
        "--filter-points",
        "filter_points",
        is_flag=True,
        default=False,
        help="With --range or --named-image-list, also drop 3D points that have "
        "no remaining observations. Default is to keep all 3D points.",
    )(f)
    f = range_option(
        "Export only images whose file number matches this range expression "
        "(e.g. '10-50' or '0-9,20-29'). Observations on excluded images are "
        "dropped.",
        on="reconstruction",
    )(f)
    return f


def validate_range_options(selection: ImageSelection, filter_points: bool) -> None:
    """Reject `--filter-points` without a range or list to filter against."""
    if filter_points and not selection.given:
        raise click.UsageError("--filter-points requires --range or --named-image-list")


def apply_range_filter(
    recon,
    range_expr_str: str,
    filter_points: bool,
    echo: Callable[[str], None],
):
    """Subset the reconstruction by image file number range."""
    import numpy as np

    from .._sfmtool.reconstruction import RangeExpr

    range_numbers = set(RangeExpr(range_expr_str))

    image_names = recon.image_names
    keep_indices: list[int] = []
    for i, name in enumerate(image_names):
        file_number = number_from_filename(name)
        if file_number is not None and file_number in range_numbers:
            keep_indices.append(i)

    if not keep_indices:
        available_numbers = sorted(
            n
            for n in (number_from_filename(name) for name in image_names)
            if n is not None
        )
        raise ValueError(
            f"No images remain after applying range filter '{range_expr_str}'. "
            f"Available file numbers: {available_numbers}"
        )

    echo(
        f"  Applied range filter '{range_expr_str}': "
        f"keeping {len(keep_indices)} of {len(image_names)} images"
    )

    indices_arr = np.array(keep_indices, dtype=np.uint32)
    return recon.subset_by_image_indices(
        indices_arr, drop_orphaned_points=filter_points
    )


def apply_image_selection(
    recon,
    selection: ImageSelection,
    filter_points: bool,
    echo: Callable[[str], None],
    sfmr_path: str | Path,
):
    """Subset the reconstruction by `selection`: a list, or else a range."""
    import numpy as np

    if selection.list_name is None:
        assert selection.range_expr is not None
        return apply_range_filter(recon, selection.range_expr, filter_points, echo)
    keep_indices = reconstruction_list_indices(recon, selection, sfmr_path)
    echo(
        f"  Applied {selection.describe_list()}: "
        f"keeping {len(keep_indices)} of {recon.image_count} images"
    )
    indices_arr = np.array(keep_indices, dtype=np.uint32)
    return recon.subset_by_image_indices(
        indices_arr, drop_orphaned_points=filter_points
    )
