# Security policy

## Supported versions

sfmtool is in active development and only the `main` branch and the most
recent published release receive security fixes.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security reports.

Use [GitHub's private vulnerability reporting][gh-pvr] on this repository
to send a report directly to the maintainers. Include:

- A description of the issue and the affected component (CLI, Python
  pipeline, Rust crate, PyO3 bindings, GUI, build/packaging).
- Reproduction steps or a proof-of-concept.
- The version (`pixi run sfm --version`) or commit SHA you tested.
- Any suggested mitigation.

We will acknowledge the report and work with you on a fix and a coordinated
disclosure timeline.

[gh-pvr]: https://github.com/sfmtool/sfmtool/security/advisories/new
