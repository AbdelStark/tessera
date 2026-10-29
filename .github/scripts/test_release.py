# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Adversarial release tag and independent-builder checks for issue #178."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from release import TARGETS, changelog_section, compare, metadata


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        (self.root / "verifier").mkdir()
        for directory, version in (
            (self.root, "0.2.0"),
            (self.root / "verifier", "0.1.0"),
        ):
            (directory / "Cargo.toml").write_text(
                f'[workspace.package]\nversion = "{version}"\n'
            )

    def test_tags_select_separate_trains_and_require_exact_version(self):
        self.assertFalse(metadata(self.root, "v0.2.0", False)["verifier"])
        self.assertTrue(metadata(self.root, "verify-v0.1.0", False)["verifier"])
        for tag in ("v0.1.0", "verify-v0.2.0", "verify-v0.1.0-rc", "verify-v0.0.1-rc"):
            with (
                self.subTest(tag=tag),
                self.assertRaisesRegex(ValueError, "differs from manifest"),
            ):
                metadata(self.root, tag, False)
        self.assertTrue(metadata(self.root, "verify-v0.0.1-rc", True)["dry_run"])
        with self.assertRaisesRegex(ValueError, "differs from manifest"):
            metadata(self.root, "verify-v0.9.0", True)

    def test_malformed_tags_cannot_enter_outputs_or_shell_commands(self):
        for tag in (
            "v01.2.0",
            "verify-v0.1.0\n",
            "../v0.1.0",
            "v0.1",
            "v0.1.0;id",
            "v0.1.0-rc.0",
            "v0.1.0+x",
        ):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, "expected"):
                metadata(self.root, tag, True)

    def test_changelog_preserves_selected_section_without_other_releases(self):
        text = "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- Next\n\n## [0.1.0] - 2026-10-28\n\n### Fixed\n\n- Released\n\n## [0.0.1]\n- Old\n"
        self.assertEqual(changelog_section(text, "0.1.0"), "### Fixed\n\n- Released")
        self.assertEqual(changelog_section(text, "0.2.0"), "### Added\n\n- Next")
        with self.assertRaisesRegex(ValueError, "required release section"):
            changelog_section("# empty", "0.1.0")
        with self.assertRaisesRegex(ValueError, "required release section"):
            changelog_section(text, "0.2.0", dry_run=False)

    def builds(self):
        out = self.root / "out"
        out.mkdir()
        (out / "release.json").write_text(json.dumps({"commit": "a" * 40}))
        (out / "notes.md").write_text("release notes\n")
        for target in TARGETS:
            for builder in (1, 2):
                directory = self.root / f"verifier-{target}-{builder}"
                directory.mkdir()
                (directory / f"tessera-verify-{target}").write_bytes(target.encode())
                (directory / "build.json").write_text(
                    json.dumps({"commit": "a" * 40, "target": target})
                )
        return out

    def test_equal_builds_include_both_architectures_and_correct_hashes(self):
        out = self.builds()
        compare(self.root, out)
        expected = "".join(
            f"{hashlib.sha256(t.encode()).hexdigest()}  tessera-verify-{t}\n"
            for t in TARGETS
        )
        self.assertEqual((out / "SHA256SUMS").read_text(), expected)
        self.assertIn(expected, (out / "notes.md").read_text())
        for target in TARGETS:
            self.assertEqual(
                (out / f"tessera-verify-{target}").read_bytes(), target.encode()
            )

    def test_one_differing_byte_refuses_release(self):
        out = self.builds()
        target = TARGETS[0]
        (self.root / f"verifier-{target}-2" / f"tessera-verify-{target}").write_bytes(
            target.encode() + b"!"
        )
        with self.assertRaisesRegex(ValueError, "binaries differ"):
            compare(self.root, out)
        self.assertFalse((out / "SHA256SUMS").exists())

    def test_missing_builder_refuses_release(self):
        out = self.builds()
        (
            self.root / f"verifier-{TARGETS[1]}-2" / f"tessera-verify-{TARGETS[1]}"
        ).unlink()
        with self.assertRaises(FileNotFoundError):
            compare(self.root, out)
        self.assertFalse((out / "SHA256SUMS").exists())

    def test_empty_binaries_refuse_release(self):
        out = self.builds()
        target = TARGETS[0]
        for builder in (1, 2):
            (
                self.root / f"verifier-{target}-{builder}" / f"tessera-verify-{target}"
            ).write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "empty"):
            compare(self.root, out)

    def test_different_provenance_refuses_release(self):
        out = self.builds()
        (self.root / f"verifier-{TARGETS[0]}-2" / "build.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            compare(self.root, out)

    def test_equal_but_wrong_commit_or_target_refuses_release(self):
        out = self.builds()
        target = TARGETS[0]
        for bad in (
            {"commit": "b" * 40, "target": target},
            {"commit": "a" * 40, "target": TARGETS[1]},
        ):
            for builder in (1, 2):
                (self.root / f"verifier-{target}-{builder}" / "build.json").write_text(
                    json.dumps(bad)
                )
            with (
                self.subTest(bad=bad),
                self.assertRaisesRegex(ValueError, "another commit or target"),
            ):
                compare(self.root, out)


if __name__ == "__main__":
    unittest.main()
