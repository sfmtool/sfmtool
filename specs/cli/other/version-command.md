# `sfm version` Command

`sfm version` prints the version of the `sfmtool` package and, when the code
running is not the release that version names, which commit it comes from.
The package version in `pyproject.toml` and `crates/sfmtool-py/Cargo.toml`
changes only at a release, so every build of a later commit carries the
version of the last release; the commit is what tells such a build from the
release. The version itself is unchanged: this is display only, and
`importlib.metadata.version("sfmtool")` still reads the package metadata.

## Command Syntax

```bash
sfm version
```

It takes no options and prints one line.

## Output

The line starts `sfmtool <version>`, the package metadata version (`unknown`
when the package metadata is not installed). What follows it depends on two
`git describe` strings, each the output of
`git describe --tags --long --always` with `-dirty` added when
`git status --porcelain --untracked-files=no` lists a change:

- **the checkout's**, run when the command runs, in the git checkout the
  `sfmtool` package is run from (see below);
- **the extension's**, recorded when the compiled extension
  `sfmtool._sfmtool` was built, in the source tree it was built from, and read
  through `sfmtool.build_git_describe()`.

Either may be unknown. Its form is `<tag>-<n>-g<hash>`: the nearest tag, the
number of commits since it and the commit's abbreviated hash, with `-dirty`
added when a tracked file differed from that commit, staged or not
(untracked files do not count). This is what `git describe --dirty` prints;
it is computed with `git status` because `git describe --dirty` rewrites the
git index when the index's file times are out of date, and the index is a path
whose change makes Cargo run `build.rs` again (below), while `git status` run
with `GIT_OPTIONAL_LOCKS=0` leaves it unchanged. With no tag
reachable, as in a shallow clone, it is the abbreviated hash alone (that is
what `--always` adds).

| Checkout | Extension | Printed |
|----------|-----------|---------|
| unknown | unknown | `sfmtool 0.3` |
| `v0.3-0-g276e8d986`, or unknown | `v0.3-0-g276e8d986` | `sfmtool 0.3` |
| `v0.3-385-g276e8d986` | `v0.3-385-g276e8d986`, or unknown | `sfmtool 0.3 (v0.3-385-g276e8d986)` |
| `v0.3-385-g276e8d986-dirty` | `v0.3-385-g276e8d986-dirty` | `sfmtool 0.3 (v0.3-385-g276e8d986-dirty)` |
| `v0.3-386-g1a2b3c4d5` | `v0.3-385-g276e8d986` | `sfmtool 0.3 (checkout v0.3-386-g1a2b3c4d5, extension built from v0.3-385-g276e8d986)` |

In words:

- **Both known and different**: both are printed in parentheses, labelled
  `checkout` and `extension built from`. This is an extension that was not
  rebuilt after the checkout moved to another commit or after its Rust sources
  changed, a stale `maturin develop`; the line shows it whether or not either
  string is a release.
- **Otherwise**, the one string known (the two being equal when both are) is
  printed in parentheses after the version, unless it names the release: a
  clean tree exactly at the tag `v<version>` or `<version>`, that is
  `v0.3-0-g<hash>` for version `0.3`. A tag that is not the package version,
  such as `v0.4-0-g<hash>` while the package says `0.3`, is printed.
- **Neither known**: the version alone.

So a wheel built at its release tag prints `sfmtool 0.3`, and so does an
editable checkout exactly at that tag, clean, with an extension built there.

## Where the checkout's describe comes from

The package is in a checkout when its directory is `<root>/src/sfmtool` and
`<root>` holds `.git` (a directory, or the file of a linked worktree), as in
the editable install `pixi` makes. The command then runs
`git -C <root> describe --tags --long --always` and
`git -C <root> status --porcelain --untracked-files=no`, each with a
two-second timeout. It skips the describe, with no message, when `<root>` has no `.git`
(an install into site-packages, including one in a virtual environment inside
some other repository, whose `<root>` is not that repository's root), when git
is not installed, when either git command exits with an error, and when one
runs past the timeout. Git runs with `GIT_OPTIONAL_LOCKS=0`, and with `GIT_DIR` and
`GIT_WORK_TREE` removed from its environment so that `-C <root>` decides the
repository. This is [`checkout_describe`](../../../src/sfmtool/_version.py),
and the line is formatted by `version_text` in the same file.

## Where the extension's describe comes from

[`build.rs`](../../../crates/sfmtool-py/build.rs) of the `sfmtool-py` crate
runs the same two commands in the repository root two directories above the crate,
when that root holds `.git`, and records its output in the compile-time
variable `SFMTOOL_BUILD_GIT_DESCRIBE`, which `build_git_describe()` in
[`lib.rs`](../../../crates/sfmtool-py/src/lib.rs) returns (`None` for the
empty string). It records the empty string, and the build goes on, when there
is no `.git` (a build from the sdist, which carries none), when git is not
installed, when git fails, and when git runs past a ten-second timeout. It
uses no crate beyond the standard library.

Cargo runs the script again, and so recompiles `sfmtool-py` with the new
string, when one of these changes:

- the git `HEAD` file, the ref of the branch it names, `packed-refs` and the
  `refs/tags` directory, so a commit, a checkout or a new tag is seen;
- the git index, so staging a change is seen;
- each crate's `Cargo.toml`, `build.rs` and `src/` directory, and the
  workspace's `Cargo.toml` and `Cargo.lock`, so `-dirty` follows an edit to
  any source the extension is compiled from, staged or not.

An edit to any other file, such as a Python module, does not run it again by
itself, so the recorded string marks such an edit `-dirty` only from the next
time the script runs. Only paths that exist are watched, since Cargo runs a
build script on every build while a path it watches is missing.
