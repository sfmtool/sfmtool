# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep source citations and local links in specs pointed at existing files."""

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "specs"
SOURCE_SUFFIXES = {".py", ".rs", ".sh"}
SOURCE_SPEC_PATH = re.compile(
    r"(?<![\w./:-])(?:(?:\.\./|\./)+)?specs/(?:[\w.-]+/)*[\w.-]+\.md(?![\w/-]|\.[A-Za-z0-9])"
)
FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
INLINE_CODE = re.compile(r"(`+).*?\1")
INLINE_LINK = re.compile(r"\]\(\s*(?:<([^>]+)>|([^\s)]+))")
REFERENCE_LINK = re.compile(r"^[ \t]{0,3}\[[^]]+\]:\s*(?:<([^>]+)>|([^\s]+))")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _source_citations():
    for directory in ("crates", "src", "scripts", "tests"):
        for path in (ROOT / directory).rglob("*"):
            if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                for match in SOURCE_SPEC_PATH.finditer(line):
                    yield path, line_number, match.group()


def _markdown_links(path: Path):
    text = HTML_COMMENT.sub(
        lambda match: "\n" * match.group().count("\n"),
        path.read_text(encoding="utf-8"),
    )
    fence_char = ""
    fence_length = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        marker = FENCE.match(line)
        if marker:
            run = marker.group(1)
            if not fence_char:
                fence_char, fence_length = run[0], len(run)
                continue
            if (
                run[0] == fence_char
                and len(run) >= fence_length
                and not line[marker.end() :].strip()
            ):
                fence_char = ""
                continue
        if fence_char:
            continue
        line = INLINE_CODE.sub("", line)
        for match in INLINE_LINK.finditer(line):
            yield line_number, match.group(1) or match.group(2)
        definition = REFERENCE_LINK.match(line)
        if definition:
            yield line_number, definition.group(1) or definition.group(2)


def _local_path(source: Path, target: str) -> Path | None:
    if target.startswith(("#", "//")) or SCHEME.match(target):
        return None
    path = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not path:
        return None
    return ROOT / path.lstrip("/") if path.startswith("/") else source.parent / path


def test_markdown_links_skip_code_examples(tmp_path):
    markdown = tmp_path / "example.md"
    markdown.write_text(
        "`[example](planned.md)`\n"
        "```markdown\n"
        "[example](planned.md)\n"
        "```\n"
        "[real](existing.md)\n"
        "[reference]: existing.md\n",
        encoding="utf-8",
    )
    assert list(_markdown_links(markdown)) == [
        (5, "existing.md"),
        (6, "existing.md"),
    ]


def test_source_citation_paths_include_relative_paths_and_sentence_punctuation():
    line = (
        "See ../specs/formats/sfmr-file-format.md. "
        "Also specs/formats/matches-file-format.md."
    )
    assert [match.group() for match in SOURCE_SPEC_PATH.finditer(line)] == [
        "../specs/formats/sfmr-file-format.md",
        "specs/formats/matches-file-format.md",
    ]


def test_source_citations_and_spec_links_resolve():
    missing = []
    citations = 0
    links = 0
    for source, line, target in _source_citations():
        citations += 1
        paths = [ROOT / target]
        if target.startswith("."):
            paths = [source.parent / target]
            crate = next(
                (
                    parent
                    for parent in source.parents
                    if (parent / "Cargo.toml").is_file()
                ),
                None,
            )
            if crate is not None:
                paths.append(crate / target)
        if not any(path.is_file() for path in paths):
            missing.append(f"{source.relative_to(ROOT)}:{line}: {target}")
    for source in SPECS.rglob("*.md"):
        for line, target in _markdown_links(source):
            path = _local_path(source, target)
            if path is None:
                continue
            links += 1
            if not path.exists():
                missing.append(f"{source.relative_to(ROOT)}:{line}: {target}")
    assert citations > 0 and links > 0, "expected source citations and spec links"
    assert not missing, "broken source citations or spec links:\n" + "\n".join(
        sorted(missing)
    )
