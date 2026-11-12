# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The `-r/--range` and `--named-image-list` options shared by ten commands.

`sfm sift`, `sfm match`, `sfm motion`, `sfm pick-images` and `sfm solve` pick
images from image directories; `sfm analyze --metrics`, `sfm panorama`,
`sfm to-colmap-bin` and `sfm to-nerfstudio` pick them from a reconstruction;
`sfm estimate-intrinsics` picks them from a `.matches` file. Each takes `-r/--range`,
whose value is either a range expression of file numbers or the name of a
named image list of the workspace, optionally followed by a slice such as
`[:30]`, and `--named-image-list NAME`, the long form for a list. One name and
one range together narrow the list, and a slice then takes part of the
narrowed list by position. The options, the range syntax their help gives,
the rule that tells a name from a range, the check that a range parses, the
errors and the resolution of a list all live here so that the ten commands
cannot drift apart.

A range on its own reaches each command as the string it always took
(`ImageSelection.range_expr`), so a command's handling of a range is its own
and is unchanged. A list is resolved here: against the workspace's files for
the image-directory commands (`resolve_list_images`), and against a
reconstruction's image names for the others
(`reconstruction_list_indices`). `sfm solve` with a `.matches` file, and
`sfm estimate-intrinsics`, filter the file's images by a range or a list
through `matches_image_filter`. The
`sfmtool.workspace` bindings split a value into a name and a slice and apply
the slice, and hold the rules and the messages for both.

The filter for `to-colmap-bin` and `to-nerfstudio` takes its own `echo`
callable rather than choosing one, because the two commands report through
different sinks (`print` and `click.echo`) and the point of this module is to
leave their output untouched.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click

from .._filenames import has_extension, number_from_filename

if TYPE_CHECKING:
    from .._sfmtool.workspace import NamedImageLists, NamedImageListSlice
    from ..colmap.db_setup import MatchesImageFilter

_RANGE_EXAMPLES = (
    "'7', '1-100', '1-10,20,30-40' or '1-100:10' (every 10th number from 1 to 100)"
)

RANGE_OR_NAME_SENTENCE = (
    "A --range value with a letter or '[' in it is read as a named image "
    "list; a range holds only digits, '-', ':' and ',', as in " + _RANGE_EXAMPLES + "."
)

# No word joined by a '-' to another word, such as START-END, since Click
# wraps help text at such a hyphen and would split the syntax across lines.
_RANGE_HELP = (
    " A range is file numbers and spans separated by commas, and a span may "
    "take a step after a ':', as in " + _RANGE_EXAMPLES + "."
)

# A span with its step written with a word or a '/' in place of ':', such as
# '1-100x10' or '1-100 step 10'. Only this exact shape gets a suggestion.
_MISTYPED_STEP = re.compile(
    r"\s*(\d+)\s*-\s*(\d+)\s*(?:x|by|step|every|/)\s*(\d+)\s*", re.IGNORECASE
)


def _step_suggestion(value: str) -> str | None:
    """`value` with its step written START-END:STEP, when `value` is a span
    with a mistyped step; otherwise None."""
    m = _MISTYPED_STEP.fullmatch(value)
    if m is None:
        return None
    start, end, step = m.groups()
    return f"{start}-{end}:{step}"


def _range_or_name_note(value: str) -> str:
    """Why `value`, which has a letter in it, was read as a list name, the
    range syntax, and a suggestion when `value` is a span with a mistyped
    step."""
    note = (
        f"'{value}' was read as a named image list, not a range, because it "
        "has a letter in it; a range holds only digits, '-', ':' and ',', as "
        f"in {_RANGE_EXAMPLES}."
    )
    suggestion = _step_suggestion(value)
    if suggestion is not None:
        note += (
            " A span with a step is written START-END:STEP; did you mean "
            f"'{suggestion}'?"
        )
    return note


_NAME_HELP = (
    " A value with a letter in it is instead the name of a named image list "
    "of the workspace (see --named-image-list); give one name and one range, "
    "as two -r options, to keep the list's members whose frame number is in "
    "the range. A slice after the name, quoted, takes part of the list by "
    "position, as for a Python list: 'NAME[:30]' is its first 30 images, "
    "'NAME[-30:]' its last 30 and 'NAME[::3]' every third one."
)

_LIST_HELP = {
    "paths": (
        "Use the images of this named image list, defined in the workspace's "
        ".sfm-named-image-lists.json, in place of image paths. The workspace "
        "is found from the current directory. With -r RANGE, keep only the "
        "members whose frame number is in RANGE. A slice after the name, "
        "such as 'NAME[:30]', then takes part of the list's images by position."
    ),
    "reconstruction": (
        "Use only the reconstruction's images that are members of this named "
        "image list, defined in the .sfm-named-image-lists.json of the "
        "reconstruction's workspace. With -r RANGE, keep only the members "
        "whose frame number is in RANGE. A slice after the name, such as "
        "'NAME[:30]', then takes part of the members the reconstruction holds "
        "by position."
    ),
    "matches": (
        "Use only the .matches file's images that are among this named image "
        "list's images, with the list read from the "
        ".sfm-named-image-lists.json of the file's workspace. With -r RANGE, "
        "keep only the members whose frame number is in RANGE. A slice after "
        "the name, such as 'NAME[:30]', takes part of the list's images by "
        "position."
    ),
}


@dataclass(frozen=True)
class ImageSelection:
    """What `-r/--range` and `--named-image-list` asked for.

    At most one range and at most one list name. With both, the range narrows
    the list rather than being applied on its own. `list_slice` is the slice
    given after the list's name, which takes part of the narrowed list by
    position. `list_range_value` is the `-r` value that gave the list, slice
    included, or None when `--named-image-list` gave it.
    """

    range_expr: str | None = None
    list_name: str | None = None
    list_slice: "NamedImageListSlice | None" = None
    list_range_value: str | None = None

    @property
    def given(self) -> bool:
        return self.range_expr is not None or self.list_name is not None

    def describe_list(self) -> str:
        """The list, the range narrowing it and its slice, as a message names
        them."""
        text = f"named image list '{self.list_name}'"
        if self.range_expr is not None:
            text += f" narrowed by the range '{self.range_expr}'"
            if self.list_slice is not None:
                text += " and"
        if self.list_slice is not None:
            text += f" sliced by {self.list_slice}"
        return text

    def list_note(self) -> str:
        """For an error about the list: when a `-r` value gave it, a space and
        `_range_or_name_note` of that value; otherwise ''."""
        if self.list_range_value is None:
            return ""
        return " " + _range_or_name_note(self.list_range_value)


def range_option(
    range_help: str, *, on: str, list_help_extra: str = ""
) -> Callable[[Callable], Callable]:
    """Attach `-r/--range` and `--named-image-list` to a command.

    `range_help` is what the command's range does; the range syntax and the
    sentence about list names are added to it. `on` is `"paths"` for a
    command that picks images from image directories, `"reconstruction"` for
    one that picks them from a reconstruction and `"matches"` for one that
    picks them from a `.matches` file. `list_help_extra` is added to the help of
    `--named-image-list`, for what a list does in the command beyond that. The
    command receives `range_expr`, a tuple of the `-r` values, and
    `named_image_list`, and turns them into an `ImageSelection` with
    `parse_image_selection`.
    """
    list_help = _LIST_HELP[on] + list_help_extra

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
            help=range_help + _RANGE_HELP + _NAME_HELP,
        )(f)
        return f

    return attach


def _is_list_value(value: str) -> bool:
    """Whether a `-r` value names a list: it has a letter, or a slice's `[`."""
    return "[" in value or any(c.isalpha() for c in value)


def _split_list_value(value: str, option: str) -> tuple[str, str, object]:
    """`value`, its list name and its slice, or the usage error naming `option`."""
    from .._sfmtool.workspace import split_named_image_list_value

    try:
        name, list_slice = split_named_image_list_value(value)
    except ValueError as e:
        message = f"{option} {e}."
        # A value with no '[' that is not a valid name has a letter in it, and
        # may be a mistyped range; one with a slice was meant as a list.
        if option == "--range" and "[" not in value:
            message += " " + _range_or_name_note(value)
        raise click.UsageError(message)
    return value, name, list_slice


def _check_range(value: str) -> None:
    """The usage error for a `-r` value that is not a valid range expression."""
    from .._sfmtool.reconstruction import RangeExpr

    try:
        RangeExpr(value)
    except ValueError as e:
        message = f"--range '{value}': {e}." + _RANGE_HELP
        suggestion = _step_suggestion(value)
        if suggestion is not None:
            message += f" Did you mean '{suggestion}'?"
        raise click.UsageError(message)


def parse_image_selection(
    range_values: tuple[str, ...], named_image_list: str | None
) -> ImageSelection:
    """Sort the `-r` values and `--named-image-list` into a range and a name.

    A `-r` value with a letter or a `[` in it is a list name, optionally
    followed by a slice, and any other value is a range, which must parse as
    a range expression. At most one of each may be given in all.
    """
    names: list[tuple[str, str, object]] = []
    ranges: list[str] = []
    if named_image_list is not None:
        names.append(_split_list_value(named_image_list, "--named-image-list"))
    for value in range_values:
        if _is_list_value(value):
            names.append(_split_list_value(value, "--range"))
        else:
            _check_range(value)
            ranges.append(value)
    if len(names) > 1:
        quoted = ", ".join(f"'{value}'" for value, _, _ in names)
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
    list_value, list_name, list_slice = names[0] if names else (None, None, None)
    return ImageSelection(
        range_expr=ranges[0] if ranges else None,
        list_name=list_name,
        list_slice=list_slice,
        list_range_value=list_value if named_image_list is None else None,
    )


def _no_workspace_error(where: str, selection: ImageSelection) -> click.ClickException:
    return click.ClickException(
        f"No workspace was found {where}, so no named image lists are defined "
        f"and none is called '{selection.list_name}'." + selection.list_note()
    )


def _read_lists(workspace_dir: Path, selection: ImageSelection):
    """The workspace's lists, or the error for a workspace without a lists file."""
    from .._sfmtool.workspace import NAMED_IMAGE_LISTS_FILE_NAME, NamedImageLists

    try:
        lists = NamedImageLists.read(workspace_dir)
    except (ValueError, OSError) as e:
        raise click.ClickException(str(e))
    if lists is None:
        raise click.ClickException(
            f"The workspace {workspace_dir} has no named image lists, since it "
            f"has no {NAMED_IMAGE_LISTS_FILE_NAME}, so none is called "
            f"'{selection.list_name}'." + selection.list_note()
        )
    return lists


def _resolution_error(
    lists, selection: ImageSelection, e: Exception
) -> click.ClickException:
    """`e` from a resolver, with `ImageSelection.list_note` for an unknown name."""
    message = str(e)
    if selection.list_name not in lists:
        message += "." + selection.list_note()
    return click.ClickException(message)


def _slice(lists, selection: ImageSelection, narrow, items: list) -> list:
    """The `items` the selection's slice keeps, or all of them without one."""
    if selection.list_slice is None:
        return items
    try:
        return lists.slice_members(
            selection.list_name, selection.list_slice, items, narrow=narrow
        )
    except ValueError as e:
        raise click.ClickException(str(e))


def _narrow(selection: ImageSelection):
    if selection.range_expr is None:
        return None
    from .._sfmtool.reconstruction import RangeExpr

    return RangeExpr(selection.range_expr)


@dataclass(frozen=True)
class ListImages:
    """The image files of a named image list, and the workspace it is in."""

    paths: list[Path]
    workspace_dir: Path


def _list_image_members(
    selection: ImageSelection, workspace_dir: Path, extensions: tuple[str, ...]
) -> tuple["NamedImageLists", list[str]]:
    """The workspace's lists, and the workspace-relative paths of the images
    `selection` takes from its list there.

    The images are the list's members, narrowed by the selection's range,
    that have one of `extensions`, in the list's order; the selection's slice
    then indexes them and keeps them in the slice's order.
    """
    name = selection.list_name
    assert name is not None
    lists = _read_lists(workspace_dir, selection)
    narrow = _narrow(selection)
    try:
        members = lists.resolve(name, narrow=narrow)
    except (ValueError, OSError, InterruptedError) as e:
        raise _resolution_error(lists, selection, e)

    images = [m.path for m in members if has_extension(Path(m.path), extensions)]
    if not images:
        raise click.ClickException(
            f"{lists.path}: none of the {len(members)} members of "
            f"{selection.describe_list()} is an image file "
            f"({', '.join(extensions)})."
        )
    return lists, _slice(lists, selection, narrow, images)


def resolve_list_images(
    selection: ImageSelection, *, extensions: tuple[str, ...]
) -> ListImages:
    """Resolve `selection`'s list against the files of the current workspace.

    The workspace is the one the upward search from the current directory
    finds. The images are the list's members, in the list's order, as
    absolute paths, keeping only the members with one of `extensions`, as the
    command's own directory expansion does; the others are skipped without a
    word, as a directory listing skips them. A slice indexes the images that
    test leaves, and keeps them in the slice's order.
    """
    from ..workspace import find_workspace_for_path

    assert selection.list_name is not None
    cwd = Path.cwd()
    workspace_dir = find_workspace_for_path(cwd)
    if workspace_dir is None:
        raise _no_workspace_error(f"from the current directory {cwd}", selection)
    lists, images = _list_image_members(selection, workspace_dir, extensions)
    root = Path(lists.workspace_dir).absolute()
    return ListImages(paths=[root / p for p in images], workspace_dir=root)


def matches_image_filter(
    selection: ImageSelection,
    matches_file: str | Path,
    *,
    extensions: tuple[str, ...],
    echo: Callable[[str], None] = print,
) -> "MatchesImageFilter | None":
    """The filter `selection` puts on the images of a `.matches` file, or
    None when it asks for none.

    A range alone keeps the file's images whose file number it holds. A list
    keeps the file's images that are among the images `resolve_list_images`
    would take from it, with the same range, slice and `extensions`, but
    resolved in the `.matches` file's workspace: so a slice counts positions
    in the list, and the list's images the file does not hold are then
    ignored. Either way the kept images stay in the file's order, and keeping
    none is an error naming the range or the list and the file. `echo`
    prints the count of the list's images the file does not hold.
    """
    from ..colmap.db_setup import MatchesImageFilter

    if not selection.given:
        return None
    if selection.list_name is None:
        range_expr = selection.range_expr
        numbers = _narrow(selection)

        def keep_range(workspace_dir: Path, image_names: list[str]) -> list[int]:
            kept = [
                i
                for i, name in enumerate(image_names)
                if (n := number_from_filename(Path(name).name)) is not None
                and n in numbers
            ]
            if not kept:
                raise click.ClickException(
                    f"Range '{range_expr}' excludes all {len(image_names)} images "
                    f"in {matches_file}"
                )
            return kept

        return MatchesImageFilter(label=f"--range {range_expr}", keep=keep_range)

    def keep_list(workspace_dir: Path, image_names: list[str]) -> list[int]:
        lists, images = _list_image_members(selection, workspace_dir, extensions)
        wanted = set(images)
        kept = [i for i, name in enumerate(image_names) if name in wanted]
        if not kept:
            raise click.ClickException(
                f"{lists.path}: none of the {len(images)} images of "
                f"{selection.describe_list()} is among the {len(image_names)} "
                f"images of {matches_file}."
            )
        if len(kept) < len(images):
            echo(
                f"{len(images) - len(kept)} of the {len(images)} images of "
                f"{selection.describe_list()} are not in {matches_file} and "
                "are ignored."
            )
        return kept

    return MatchesImageFilter(label=selection.describe_list(), keep=keep_list)


def reconstruction_list_indices(
    recon, selection: ImageSelection, sfmr_path: str | Path
) -> list[int]:
    """The indices of `recon`'s images that `selection`'s list holds.

    The list is resolved in the reconstruction's workspace against its image
    names, so the image files need not exist. The indices keep the
    reconstruction's order; members the reconstruction does not hold are
    ignored, and a list that leaves none of its images is an error naming the
    list. A slice indexes the members the reconstruction holds, in the list's
    order, before the indices are put in the reconstruction's order.
    """
    name = selection.list_name
    assert name is not None
    workspace_dir = recon.workspace_dir
    if not workspace_dir:
        raise _no_workspace_error(f"for {sfmr_path}", selection)
    lists = _read_lists(Path(workspace_dir), selection)
    narrow = _narrow(selection)
    image_names = list(recon.image_names)
    try:
        members = lists.resolve_in_names(name, image_names, narrow=narrow)
    except ValueError as e:
        raise _resolution_error(lists, selection, e)
    members = _slice(lists, selection, narrow, members)
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
        "Export only images whose file number matches this range expression. "
        "Observations on excluded images are dropped.",
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
