# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Custom Click Group for organizing commands into categories."""

import click


class CategoryGroup(click.Group):
    """A Click Group that organizes commands into categories for help display."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command_categories = {}

    def add_command_with_category(self, cmd, name=None, category="Other"):
        """Add a command to the group with a category label."""
        command_name = name or cmd.name
        self.add_command(cmd, name=command_name)
        self.command_categories[command_name] = category

    def format_commands(self, ctx, formatter):
        """Format commands organized by category."""
        categories = {}
        for name in self.list_commands(ctx):
            category = self.command_categories.get(name, "Other")
            if category not in categories:
                categories[category] = []
            cmd = self.get_command(ctx, name)
            if cmd is not None:
                categories[category].append((name, cmd))

        category_order = [
            "Workspace",
            "Image Feature",
            "Reconstruction",
            "Visualization",
            "Image Processing",
            "COLMAP Interop",
            "Other",
        ]

        for category in category_order:
            if category not in categories:
                continue

            commands = categories[category]
            if not commands:
                continue

            commands.sort(key=lambda x: x[0])

            with formatter.section(f"{category} Commands"):
                rows = []
                for name, cmd in commands:
                    help_text = cmd.get_short_help_str(limit=formatter.width - 30)
                    rows.append((name, help_text))
                formatter.write_dl(rows)
