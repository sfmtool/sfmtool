# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Custom Click Group for organizing commands into categories."""

from importlib import import_module

import click
from click.utils import make_default_short_help

# The order the categories are listed in by `--help`.
CATEGORY_ORDER = (
    "Workspace",
    "Image Feature",
    "Reconstruction",
    "Visualization",
    "Image Processing",
    "COLMAP Interop",
    "Other",
)


class CategoryGroup(click.Group):
    """A Click Group that lists its commands by category in ``--help`` and
    imports the module of a lazily added command only when that command is
    looked up.

    A command added with ``add_lazy_command`` is known by its name, the module
    and attribute that define it, and its one-line help. ``--help`` lists it from
    those, so neither ``sfm --help`` nor running one command imports the modules
    of the others, which between them import numpy, OpenCV and pycolmap.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command_categories = {}
        # name -> (module, attribute, one-line help)
        self.lazy_commands = {}

    def add_lazy_command(self, name, module, attribute, short_help, category="Other"):
        """Add the command ``module.attribute`` under ``name``, importing
        ``module`` when the command is first looked up.

        ``short_help`` is the first sentence of the command's help, which
        ``--help`` shows for it without importing ``module``.
        """
        self.lazy_commands[name] = (module, attribute, short_help)
        self.command_categories[name] = category

    def list_commands(self, ctx):
        return sorted(set(self.commands) | set(self.lazy_commands))

    def get_command(self, ctx, cmd_name):
        cmd = self.commands.get(cmd_name)
        if cmd is None and cmd_name in self.lazy_commands:
            module, attribute, _ = self.lazy_commands[cmd_name]
            cmd = getattr(import_module(module), attribute)
            self.add_command(cmd, name=cmd_name)
        return cmd

    def command_short_help(self, ctx, name, limit):
        """The one-line help ``--help`` lists for the command ``name``."""
        if name in self.lazy_commands:
            return make_default_short_help(self.lazy_commands[name][2], limit)
        return self.get_command(ctx, name).get_short_help_str(limit=limit)

    def format_commands(self, ctx, formatter):
        """Format commands organized by category."""
        categories = {}
        for name in self.list_commands(ctx):
            category = self.command_categories.get(name, "Other")
            categories.setdefault(category, []).append(name)

        limit = formatter.width - 30
        for category in CATEGORY_ORDER:
            names = categories.get(category)
            if not names:
                continue
            with formatter.section(f"{category} Commands"):
                formatter.write_dl(
                    [
                        (name, self.command_short_help(ctx, name, limit))
                        for name in names
                    ]
                )
