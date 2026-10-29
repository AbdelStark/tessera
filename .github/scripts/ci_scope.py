# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Route documentation changes without suppressing checks for unknown source paths."""

import sys


def needs_code_checks(paths: list[str]) -> bool:
    """Unknown paths and an unavailable diff require the complete bootstrap pipeline."""
    return not paths or any(
        path == "docs/integration.md" or not (
            ("/" not in path and path.endswith(".md"))
            or path.startswith(("docs/", "research/"))
        )
        for path in paths
    )


if __name__ == "__main__":
    changed = sys.stdin.buffer.read().decode("utf-8", errors="replace").split("\0")
    print(f"code={str(needs_code_checks([path for path in changed if path])).lower()}")
