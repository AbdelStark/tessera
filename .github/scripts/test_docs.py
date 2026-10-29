# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Exercise rendered-link failures and the documentation artifact boundary (#176)."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from docs import ROOT_PAGES, Page, Rewrite, assemble, check, prepare, repair_api


class SiteFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.site = base / "site"
        self.site.mkdir()
        (base / ".env").write_text("PRIVATE_CANARY=never-export\n")
        (self.site / "sub").mkdir()
        (self.site / "index.html").write_text(
            '<a href="sub/page.html?search=x#%CE%B2">next</a><img src="image.svg"><a href="https://example.invalid/">external</a>'
        )
        (self.site / "sub/page.html").write_text(
            '<h1 id="β">Title</h1><a href="../index.html">home</a>'
        )
        (self.site / "image.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        )


class LinkTests(SiteFixture):
    def test_upstream_relative_method_and_dispatcher_links_use_real_targets(self):
        api = self.site / "api/prover/tracing_core"
        (api / "span").mkdir(parents=True)
        owner = api / "span/struct.Current.html"
        owner.write_text('<h2 id="method.new">new</h2>')
        (api / "trait.Subscriber.html").write_text(
            '<a href="span/struct.Current.html#tymethod.new">new</a>'
        )
        dispatcher = api / "dispatcher/index.html"
        dispatcher.parent.mkdir()
        dispatcher.write_text('<a href="#fn.with_default">default</a>')
        (dispatcher.parent / "fn.with_default.html").write_text("default")
        repair_api(self.site, self.site)
        check(self.site)
        self.assertIn("#method.new", (api / "trait.Subscriber.html").read_text())
        self.assertEqual(Page(dispatcher.read_text()).urls, ["fn.with_default.html"])
        owner.write_text("no method")
        with self.assertRaisesRegex(ValueError, "missing anchor #method.new"):
            check(self.site)

    def test_unresolved_rust_paths_are_not_external_url_schemes(self):
        for url in ("crate::Span", "super::Subscriber", "Visit::record_value"):
            (self.site / "index.html").write_text(f'<a href="{url}">broken</a>')
            with (
                self.subTest(url=url),
                self.assertRaisesRegex(ValueError, "unsupported URL scheme"),
            ):
                check(self.site)

    def test_disabled_tracing_features_link_to_pinned_upstream_docs(self):
        page = self.site / "api/prover/tracing_core/field/index.html"
        page.parent.mkdir(parents=True)
        page.write_text(
            '<a href="Visit::record_value">visit</a>'
            '<a href="valuable::Valuable::as_value">valuable</a>'
        )
        reexport = self.site / "api/prover/tracing/field/index.html"
        reexport.parent.mkdir(parents=True)
        reexport.write_text(page.read_text())
        root = Path(self.temp.name)
        (root / "Cargo.lock").write_text(
            '[[package]]\nname="tracing-core"\nversion="0.1.36"\n'
        )
        repair_api(root, self.site)
        self.assertEqual(
            Page(page.read_text()).urls,
            [
                "https://docs.rs/tracing-core/0.1.36/tracing_core/field/trait.Visit.html#method.record_value",
                "https://docs.rs/valuable/0.1.1/valuable/trait.Valuable.html#tymethod.as_value",
            ],
        )
        self.assertEqual(Page(reexport.read_text()).urls, Page(page.read_text()).urls)
        check(self.site)

    def test_nested_queries_unicode_fragments_images_and_external_urls(self):
        check(self.site)
        self.assertEqual(Page('<a name="a&amp;b">').ids, {"a&b"})

    def test_missing_page_is_an_error_with_source_and_target(self):
        (self.site / "sub/page.html").unlink()
        with self.assertRaisesRegex(
            ValueError, r"index.html: sub/page.html.*missing file"
        ):
            check(self.site)

    def test_missing_anchor_is_an_error_with_source_and_target(self):
        (self.site / "sub/page.html").write_text('<h1 id="other">Title</h1>')
        with self.assertRaisesRegex(
            ValueError, r"index.html: sub/page.html.*missing anchor #%CE%B2"
        ):
            check(self.site)

    def test_missing_image_is_not_ignored(self):
        (self.site / "image.svg").unlink()
        with self.assertRaisesRegex(ValueError, r"index.html: image.svg: missing file"):
            check(self.site)

    def test_literal_and_encoded_path_traversal_are_refused(self):
        for url in ("../.env", "%2e%2e/.env"):
            (self.site / "index.html").write_text(f'<a href="{url}">bad</a>')
            with (
                self.subTest(url=url),
                self.assertRaisesRegex(ValueError, "escapes the site"),
            ):
                check(self.site)

    def test_no_rendered_output_is_an_error(self):
        with (
            tempfile.TemporaryDirectory() as empty,
            self.assertRaisesRegex(ValueError, "no HTML"),
        ):
            check(Path(empty))

    def test_redirects_preserve_and_validate_the_destination_fragment(self):
        (self.site / "redirect.html").write_text(
            '<meta http-equiv="refresh" content="0;URL=sub/page.html">'
        )
        (self.site / "index.html").write_text(
            '<a href="redirect.html#%CE%B2">target</a>'
        )
        check(self.site)
        (self.site / "sub/page.html").write_text(
            '<h1 id="other">wrong destination</h1>'
        )
        with self.assertRaisesRegex(ValueError, "missing anchor"):
            check(self.site)

    def test_redirect_cycles_are_an_error(self):
        (self.site / "index.html").write_text(
            '<meta http-equiv="refresh" content="0;URL=index.html">'
        )
        with self.assertRaisesRegex(ValueError, "redirect cycle"):
            check(self.site)

    def test_encoded_rustdoc_ids_are_validated_without_discarding_the_fragment(self):
        path = self.site / "index.html"
        path.write_text(
            '<a href="#impl-T%3CU%3E">impl</a><h2 id="impl-T%3CU%3E">generic</h2>'
        )
        check(self.site)
        path.write_text('<a href="#impl-T%3CU%3E">impl</a>')
        with self.assertRaisesRegex(ValueError, "missing anchor"):
            check(self.site)

    def test_source_ranges_require_the_rustdoc_script_and_every_line(self):
        path = self.site / "module.rs.html"
        (self.site / "src-script-test.js").write_text("// fixture")
        path.write_text(
            '<script src="src-script-test.js"></script><a id="1"></a><a id="2"></a><a id="3"></a>'
        )
        (self.site / "index.html").write_text('<a href="module.rs.html#1-3">source</a>')
        check(self.site)
        for text in (
            '<a id="1"></a><a id="2"></a><a id="3"></a>',
            '<script src="src-script-test.js"></script><a id="1"></a><a id="3"></a>',
            '<script src="src-script-test.js"></script><a id="1"></a><a id="3"></a><a id="4"></a>',
        ):
            path.write_text(text)
            with (
                self.subTest(text=text),
                self.assertRaisesRegex(ValueError, "missing anchor #1-3"),
            ):
                check(self.site)

    def test_rewriting_preserves_svg_attribute_case_and_code_and_script_text(self):
        text = '<svg viewBox="0 0 2 2"><a href="old">link</a></svg><code>&lt;a href="old"&gt;</code><script>const text = \'href="old"\';</script>'
        result = text
        for start, end, replacement in reversed(
            Rewrite(text, lambda url: "new").changes
        ):
            result = result[:start] + replacement + result[end:]
        self.assertEqual(
            result, text.replace('<a href="old">link', '<a href="new">link')
        )


class RepairTests(SiteFixture):
    def test_associated_rustdoc_item_uses_an_existing_unique_anchor(self):
        api = self.site / "api/prover"
        index = api / "tessera_ir/index.html"
        index.parent.mkdir(parents=True)
        index.write_text('<a class="enum" href="enum.Op.html">Op</a>')
        target = index.parent / "enum.Op.html"
        target.write_text('<h2 id="variant.InverseCdf">InverseCdf</h2>')
        page = api / "tessera_compiler/index.html"
        page.parent.mkdir()
        page.write_text('<a href="tessera_ir::Op::InverseCdf">variant</a>')
        repair_api(self.site, self.site)
        self.assertIn("../tessera_ir/enum.Op.html#variant.InverseCdf", page.read_text())
        check(self.site)
        for ids, symbol in [
            ('<h2 id="variant.InverseCdf">present</h2>', "Missing"),
            ('<h2 id="method.new">one</h2><h2 id="method.new-1">two</h2>', "new"),
        ]:
            target.write_text(ids)
            page.write_text(
                f'<a href="tessera_ir::Op::{symbol}">unknown or ambiguous</a>'
            )
            repair_api(self.site, self.site)
            with (
                self.subTest(symbol=symbol),
                self.assertRaisesRegex(ValueError, "missing file"),
            ):
                check(self.site)

    def test_qualified_rustdoc_symbol_uses_its_real_export_target(self):
        api = self.site / "api/prover"
        index = api / "tessera_compiler/index.html"
        index.parent.mkdir(parents=True)
        index.write_text('<a class="enum" href="select/enum.Gadget.html">Gadget</a>')
        target = index.parent / "select/enum.Gadget.html"
        target.parent.mkdir()
        target.write_text("<h1>Gadget</h1>")
        page = api / "tessera_gadgets/index.html"
        page.parent.mkdir()
        page.write_text('<a href="tessera_compiler::Gadget">Gadget</a>')
        repair_api(self.site, self.site)
        self.assertIn("../tessera_compiler/select/enum.Gadget.html", page.read_text())
        check(self.site)
        page.write_text('<a href="tessera_compiler::Missing">Missing</a>')
        repair_api(self.site, self.site)
        with self.assertRaisesRegex(
            ValueError, "tessera_compiler::Missing: missing file"
        ):
            check(self.site)

        conflict = target.parent / "fn.Gadget.html"
        conflict.write_text("<h1>Conflicting export</h1>")
        index.write_text(
            '<a class="enum" href="select/enum.Gadget.html">Gadget</a><a class="fn" href="select/fn.Gadget.html">Gadget</a>'
        )
        page.write_text('<a href="tessera_compiler::Gadget">ambiguous</a>')
        repair_api(self.site, self.site)
        with self.assertRaisesRegex(
            ValueError, "tessera_compiler::Gadget: missing file"
        ):
            check(self.site)

    def test_inherited_tracing_link_still_checks_its_destination_anchor(self):
        api = self.site / "api/prover"
        target = api / "tracing/dispatcher/index.html"
        target.parent.mkdir(parents=True)
        target.write_text('<h1 id="setting-the-default-subscriber">Dispatcher</h1>')
        user = api / "crate/struct.User.html"
        user.parent.mkdir()
        user.write_text(
            '<a href="dispatcher#setting-the-default-subscriber">dispatcher</a>'
        )
        repair_api(self.site, self.site)
        self.assertIn(
            "../tracing/dispatcher/index.html#setting-the-default-subscriber",
            user.read_text(),
        )
        check(self.site)
        target.write_text('<h1 id="different">wrong anchor</h1>')
        with self.assertRaisesRegex(ValueError, "missing anchor"):
            check(self.site)

    def test_missing_implementor_script_is_only_filled_for_an_empty_trait(self):
        api = self.site / "api/prover"
        (api / "crate").mkdir(parents=True)
        page = api / "crate/trait.Empty.html"
        page.write_text('<script src="../trait.impl/crate/trait.Empty.js"></script>')
        repair_api(self.site, self.site)
        self.assertTrue((api / "trait.impl/crate/trait.Empty.js").is_file())
        check(self.site)
        page = api / "crate/trait.Nonempty.html"
        page.write_text(
            '<h2 id="impl-Nonempty-for-T">Implementation</h2><script src="../trait.impl/crate/trait.Nonempty.js"></script>'
        )
        repair_api(self.site, self.site)
        self.assertFalse((api / "trait.impl/crate/trait.Nonempty.js").exists())
        with self.assertRaisesRegex(ValueError, "trait.Nonempty.js: missing file"):
            check(self.site)

    def test_hidden_trait_method_link_requires_the_destination_method(self):
        page = self.site / "api/prover/crate/struct.Example.html"
        page.parent.mkdir(parents=True)
        page.write_text(
            '<a href="#tymethod.show">show</a><h2 id="method.show">Implementation</h2>'
        )
        repair_api(self.site, self.site)
        check(self.site)
        self.assertIn('href="#method.show"', page.read_text())
        page.write_text('<a href="#tymethod.show">show</a>')
        repair_api(self.site, self.site)
        with self.assertRaisesRegex(ValueError, "missing anchor #tymethod.show"):
            check(self.site)


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        for name in (
            *ROOT_PAGES,
            "docs/book.toml",
            "spec/schemas/README.md",
            "docs/spec/02-api.md",
            "docs/rfcs/RFC-0001-example.md",
            "research/note.md",
            "src/example.rs",
        ):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Example\n\nStatus: Draft\n")
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-q",
                "-m",
                "Fixture",
            ],
            cwd=self.root,
            check=True,
        )
        (self.root / ".env").write_text("PRIVATE_CANARY=never-export\n")
        self.work = self.base / "build"

    def test_only_tracked_scope_is_staged_and_draft_status_is_preserved(self):
        prepare(self.root, self.work)
        self.assertEqual(
            (self.work / "src/docs/rfcs/RFC-0001-example.md").read_bytes(),
            (self.root / "docs/rfcs/RFC-0001-example.md").read_bytes(),
        )
        self.assertFalse((self.work / "src/.env").exists())
        self.assertFalse((self.work / "src/research/note.md").exists())
        self.assertNotIn(
            "never-export",
            "".join(p.read_text() for p in self.work.rglob("*") if p.is_file()),
        )
        summary = (self.work / "src/SUMMARY.md").read_text()
        self.assertIn("docs/spec/02-api.md", summary)
        self.assertIn("docs/rfcs/RFC-0001-example.md", summary)

    def test_existing_staging_is_not_overwritten(self):
        self.work.mkdir()
        (self.work / "keep").write_text("original")
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare(self.root, self.work)
        self.assertEqual((self.work / "keep").read_text(), "original")

    def test_symlink_chapter_cannot_export_untracked_data(self):
        path = self.root / "docs/secret.md"
        path.symlink_to(self.root / ".env")
        subprocess.run(["git", "add", "docs/secret.md"], cwd=self.root, check=True)
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            prepare(self.root, self.work)

    def rendered(self):
        prepare(self.root, self.work)
        manifest = json.loads((self.work / "manifest.json").read_text())
        site = self.work / "site"
        for path in manifest["chapters"]:
            page = site / Path(path).with_suffix(".html")
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text('<h1 id="example">Example</h1>')
        for name, entry in (("prover", "tessera_prover"), ("verifier", "tv_core")):
            path = self.base / name / entry / "index.html"
            path.parent.mkdir(parents=True)
            path.write_text('<a href="../index.html">crates</a>')
        return site

    def test_assembly_copies_tracked_assets_and_uses_commit_links_for_research(self):
        site = self.rendered()
        (site / "SPEC.html").write_text(
            '<a href="src/example.rs">source</a><a href="research/note.html#example">research</a>'
        )
        assemble(self.root, self.work, self.base / "prover", self.base / "verifier")
        self.assertEqual(
            (site / "src/example.rs").read_bytes(),
            (self.root / "src/example.rs").read_bytes(),
        )
        self.assertIn(
            "github.com/AbdelStark/tessera/blob/", (site / "SPEC.html").read_text()
        )
        self.assertFalse((site / ".env").exists())

    def test_missing_in_scope_chapter_cannot_be_hidden_by_a_source_link(self):
        site = self.rendered()
        (site / "docs/rfcs/RFC-0001-example.html").unlink()
        with self.assertRaisesRegex(
            ValueError, "missing rendered chapters.*RFC-0001-example"
        ):
            assemble(self.root, self.work, self.base / "prover", self.base / "verifier")

    def test_missing_api_tree_is_an_error(self):
        self.rendered()
        with self.assertRaisesRegex(ValueError, "missing verifier rustdoc"):
            assemble(self.root, self.work, self.base / "prover", self.base / "absent")

    def test_untracked_resource_is_not_copied(self):
        site = self.rendered()
        (site / "SPEC.html").write_text('<a href=".env">private</a>')
        with self.assertRaisesRegex(ValueError, "unresolved documentation link: .env"):
            assemble(self.root, self.work, self.base / "prover", self.base / "verifier")


if __name__ == "__main__":
    unittest.main()
