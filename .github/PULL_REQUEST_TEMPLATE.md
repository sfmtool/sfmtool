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

<!-- How did you verify this works? Commands run, datasets used, manual steps. -->

## Screenshots

<!--
Screenshots are required when the change is something you can see: in the
SfM Explorer viewer, in an image a CLI command writes, or on a docs page. Small
changes count, such as a new menu entry, a changed label or tooltip. When the
change alters something that already existed, show a before/after pair. Write
"n/a" only when there is nothing to see, and say what you checked, for example
"n/a: Rust core only, no viewer, CLI image or docs change".

Check that a screenshot shows no sensitive data. Prefer screenshots of the
checked-in test data under `test-data/images/`, such as
`seoul_bull_sculpture_ground_truth_v01.sfmr`,
`seoul_bull_sculpture_ground_truth_v02.sfmr` or
`kerry_park_ground_truth_v01.sfmr`. Make sure no local paths, usernames,
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

## License

By submitting this pull request, I confirm that my contribution is made
under the terms of the project's [Apache License 2.0](/LICENSE), and I
certify the [Developer Certificate of Origin](/CONTRIBUTING.md#sign-off-your-commits-dco)
by signing off on all of my commits.
