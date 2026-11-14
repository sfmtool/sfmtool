# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for CLI commands."""

import functools
import time

import click


def timed_command(f=None, *, stderr_when=None):
    """Decorator that measures and prints the execution time of a CLI command.

    The line reads ``Completed in 12.34s`` (or ``Completed in 2m 5.00s`` from a
    minute on) and is printed on stdout. ``stderr_when`` names a boolean
    parameter of the command, such as ``as_json`` for a ``--json`` flag that
    puts machine-readable output on stdout: when the command is called with
    that parameter true, the line goes to stderr instead, so stdout holds only
    that output. Use it as ``@timed_command`` or
    ``@timed_command(stderr_when="as_json")``.
    """

    def decorate(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            err = bool(stderr_when and kwargs.get(stderr_when))
            start_time = time.perf_counter()
            try:
                return f(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start_time
                if elapsed >= 60:
                    minutes = int(elapsed // 60)
                    seconds = elapsed % 60
                    click.echo(f"Completed in {minutes}m {seconds:.2f}s", err=err)
                else:
                    click.echo(f"Completed in {elapsed:.2f}s", err=err)

        return wrapper

    if f is None:
        return decorate
    return decorate(f)
