# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import click


@click.group()
def main():
    """SfM Tool - Fun with Structure from Motion."""


@main.command()
def version():
    """Print the version."""
    click.echo("sfmtool 0.1")
