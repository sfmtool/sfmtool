<!--
Thanks for contributing to sfmtool! A few notes before you fill this out:

- For non-trivial behavior changes, the relevant spec under specs/ should be
  updated in the same PR (or a follow-up tracked in this PR description).
- If you touched anything in crates/ that is re-exported through sfmtool-py,
  rerun `pixi run maturin develop --release` before running Python tests.
-->

## Summary

<!-- What does this PR do, and why? One or two sentences is fine. -->

## Changes

<!-- Bullet list of the notable changes. Skip this for trivial PRs. -->

## Testing

<!--
How did you verify this works? Commands run, datasets used, manual steps.

If you added a test or changed code that a test runs, give the time of each
test you added, and the before and after times of each existing test whose
time you changed by more than a second, from the durations list at the end of
`pixi run test` (it lists phases of 10 seconds or more; add
`--durations-min=0` to list every phase). Repeat a comparison before
reporting it, since whole-suite times vary between runs. For a test that
takes more than 5 seconds, say why it needs its input (see `specs/STANDARDS.md#tests-cost-what-their-assertions-need`).
-->

## Screenshots and Renders

<!--
Screenshots and rendered images are required when the change alters something
you can see, in the SfM Explorer viewer or on a docs page, or any image the
code renders or writes: the output image of a CLI command, or the output of a
helper or library function that renders or encodes an image (a renderer, a
texture, a thumbnail or patch bitmap builder, an image encoder). Small changes
count, such as a new menu entry, a changed label or tooltip. When the change
alters an image that already existed, show a before/after pair. Give each
rendered image a caption saying what was rendered with which options. Write
"n/a" only when nothing visible and no rendered image changes, and say what
you checked, for example "n/a: Rust core only, no viewer, docs or rendered
image change".

Check that a screenshot or render shows no sensitive data. Prefer pictures of the
checked-in test data under `test-data/images/`, such as
`seoul_bull_sculpture_ground_truth_v01.sfmr`,
`seoul_bull_sculpture_ground_truth_v02.sfmr`,
`kerry_park_ground_truth_v01.sfmr`, `kerry_park_ground_truth_v02.sfmr` or
`dino_dog_toy_ground_truth_v01.sfmr`. Make sure no local paths, usernames,
private images or other personal data are visible, for example in a window
title or a file dialog. Screenshot one panel rather than the whole desktop.
The viewer's MCP endpoint (`pixi run gui-mcp`, see `specs/gui/mcp-server.md`)
can screenshot a single panel.
-->

## Checklist

- [ ] Python changes: `pixi run fmt && pixi run check`
- [ ] Rust changes: `pixi run cargo fmt && pixi run cargo clippy --workspace --all-targets --features sfm-explorer/ui-tests -- -D warnings && pixi run doc`
- [ ] Rust changes touching `sfmtool-py` re-exports: ran `pixi run maturin develop --release`
- [ ] Tests pass locally: the full suites that apply (`pixi run test`, `pixi run cargo test --workspace`, `pixi run -e ui ui-test`), not a subset; if one did not run, say which and why. The one allowed failure is the Windows real-mouse `ui_basic` test failing with "could not aim at the viewer's own window" on a locked desktop, when the rest of `ui_basic` passes (see `AGENTS.md` § "Task completion checks")
- [ ] Updated the relevant spec under `specs/` if behavior changed, or added a new spec if this introduces new behavior
- [ ] Each test added or changed takes only the time its assertions need, with no case or assertion dropped to save time; test times reported in Testing (see `specs/STANDARDS.md#tests-cost-what-their-assertions-need`)

## License

By submitting this pull request, I confirm that my contribution is made
under the terms of the project's [Apache License 2.0](/LICENSE), and I
certify the [Developer Certificate of Origin](/CONTRIBUTING.md#sign-off-your-commits-dco)
by signing off on all of my commits.
