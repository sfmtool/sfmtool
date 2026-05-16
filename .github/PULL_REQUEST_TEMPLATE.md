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

## Checklist

- [ ] Python changes: `pixi run fmt && pixi run check`
- [ ] Rust changes: `pixi run cargo fmt && pixi run cargo clippy --workspace`
- [ ] Rust changes touching `sfmtool-py` re-exports: ran `pixi run maturin develop --release`
- [ ] Tests pass locally (`pixi run test`, `pixi run cargo test --workspace` as applicable)
- [ ] Updated the relevant spec under `specs/` if behavior changed, or added a new spec if this introduces new behavior

## License

By submitting this pull request, I confirm that my contribution is made
under the terms of the project's [Apache License 2.0](/LICENSE), and I
certify the [Developer Certificate of Origin](/CONTRIBUTING.md#sign-off-your-commits-dco)
by signing off on all of my commits.
