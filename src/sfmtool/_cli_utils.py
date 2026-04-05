# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for CLI commands."""

import functools
import time

import click


def timed_command(f):
    """Decorator that measures and prints the execution time of a CLI command."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            return f(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start_time
            if elapsed >= 60:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                click.echo(f"Completed in {minutes}m {seconds:.2f}s")
            else:
                click.echo(f"Completed in {elapsed:.2f}s")

    return wrapper
