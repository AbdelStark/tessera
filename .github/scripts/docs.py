# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Stage canonical Markdown, assemble separate rustdoc trees, and check the site (#176)."""

import argparse
import html
import json
import re
import shutil
import subprocess
import tomllib
from html.parser import HTMLParser
from os.path import relpath
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
ROOT_PAGES = (
    "README.md",
    "SPEC.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "CHANGELOG.md",
    "verifier/CHANGELOG.md",
    "python/README.md",
)
ATTR = re.compile(r"""\b(href|src)\s*=\s*(["'])(.*?)\2""", re.IGNORECASE)


def tracked(root):
    """Use the Git index, never ignored artifacts, credentials or directory symlinks."""
    result = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return {Path(p.decode()) for p in result.split(b"\0") if p}


def title(path):
    """Use a chapter's first heading, with a filename fallback."""
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            return line[2:].replace("[", r"\[").replace("]", r"\]")
    return path.stem


def prepare(root, work):
    """Create a fresh staging tree from tracked in-scope Markdown only."""
    if work.exists():
        raise ValueError(
            f"{work}: staging directory already exists; choose a fresh path"
        )
    files = tracked(root)
    pages = {p for p in files if p.suffix == ".md" and p.parts[0] == "docs"}
    pages.update(Path(p) for p in ROOT_PAGES)
    pages.update(
        p for p in files if p.suffix == ".md" and p.parts[:2] == ("spec", "schemas")
    )
    if not pages <= files:
        raise ValueError(
            f"missing tracked chapters: {sorted(str(p) for p in pages - files)}"
        )
    work.mkdir(parents=True)
    src = work / "src"
    for page in sorted(pages):
        if (root / page).is_symlink():
            raise ValueError(f"{page}: a documentation chapter must not be a symlink")
        target = src / page
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / page, target)
    shutil.copyfile(root / "docs/book.toml", work / "book.toml")

    # Landing pages make directory links and the chapter hierarchy agree.
    directories = {p.parent for p in pages if p.parent != Path(".")}
    for directory in list(directories):
        directories.update(p for p in directory.parents if p != Path("."))
    landings = {}
    for directory in sorted(directories):
        readme = directory / "README.md"
        if readme in pages:
            landings[directory] = readme
        else:
            landing = directory / "index.md"
            if landing not in pages:
                (src / directory).mkdir(parents=True, exist_ok=True)
                label = {
                    Path("docs"): "Guides and reports",
                    Path("docs/spec"): "Specification",
                    Path("docs/rfcs"): "RFCs",
                }.get(directory, directory.name.replace("-", " ").title())
                (src / landing).write_text(f"# {label}\n\n")
                pages.add(landing)
            landings[directory] = landing
    for directory, landing in landings.items():
        if landing in files:
            continue
        entries = [p for p in pages if p.parent == directory and p != landing]
        links = [f"- [{title(src / p)}]({p.name})" for p in sorted(entries)]
        links += [
            f"- [{d.name}]({p.relative_to(directory).as_posix()})"
            for d, p in sorted(landings.items())
            if d.parent == directory
        ]
        with (src / landing).open("a") as out:
            out.write("\n".join(links) + "\n")

    api = Path("api.md")
    if api in pages:
        raise ValueError("api.md collides with the generated API index")
    (src / api).write_text(
        "# Rust API reference\n\n"
        "[02 §1](docs/spec/02-public-api.md#1-stability-tiers) defines tier B: the "
        "Rust items re-exported by `tessera-prover` and `tv-core`, alongside the CLI "
        "and Python interfaces. Other crate APIs are internal (tier C).\n\n"
        "- [Prover API: tessera-prover](api/prover/tessera_prover/index.html)\n"
        "- [Independent verifier API: tv-core](api/verifier/tv_core/index.html)\n"
        "- [Prover API index](api/prover/index.html)\n"
        "- [Verifier API index](api/verifier/index.html)\n"
        "- [CLI and Python interfaces](docs/spec/02-public-api.md#2-command-line-interface)\n\n"
        "The two workspaces build separately. Their API examples are compiled as "
        "doctests; `no_run` examples require caller-owned model or proof inputs. "
        "Neither compiling an example nor a component result certifies a proof.\n"
    )
    pages.add(api)
    summary = [
        "# Summary",
        "",
        "[Tessera](README.md)",
        "",
        "- [Specification entry](SPEC.md)",
        "- [Rust API reference](api.md)",
    ]

    def tree(directory, depth=0, excluded=()):
        landing = landings[directory]
        summary.append(
            f"{'  ' * depth}- [{title(src / landing)}]({landing.as_posix()})"
        )
        for page in sorted(p for p in pages if p.parent == directory and p != landing):
            summary.append(
                f"{'  ' * (depth + 1)}- [{title(src / page)}]({page.as_posix()})"
            )
        for child in sorted(
            d for d in landings if d.parent == directory and d not in excluded
        ):
            tree(child, depth + 1)

    tree(Path("docs/spec"))
    tree(Path("docs/rfcs"))
    tree(Path("docs"), excluded=(Path("docs/spec"), Path("docs/rfcs")))
    tree(Path("spec/schemas"))
    summary += [
        f"- [{title(src / p)}]({p.as_posix()})" for p in map(Path, ROOT_PAGES[2:])
    ]
    listed = {Path(p) for p in re.findall(r"\]\(([^)]+)\)", "\n".join(summary))}
    summary += [f"- [{title(src / p)}]({p.as_posix()})" for p in sorted(pages - listed)]
    (src / "SUMMARY.md").write_text("\n".join(summary) + "\n")
    # Keep the manifest outside the uploaded site. All paths come from the Git index.
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    manifest = {
        "schema": "tessera.docs-build/1",
        "commit": commit,
        "chapters": sorted(p.as_posix() for p in pages),
        "directories": {
            str(d): str(p.with_suffix(".html")) for d, p in landings.items()
        },
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"staged {len(pages)} chapters in {work}")


class Page(HTMLParser):
    """Collect IDs and URL attributes using an HTML parser, not Markdown heuristics."""

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.urls = []
        self.redirect = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "meta" and (values.get("http-equiv") or "").lower() == "refresh":
            match = re.search(
                r"url\s*=\s*(.*)", values.get("content") or "", re.IGNORECASE
            )
            if match:
                self.redirect = match[1].strip(" \"'")
        for key, value in attrs:
            if value is None:
                continue
            if key == "id" or (tag == "a" and key == "name"):
                self.ids.add(value)
            if key in ("href", "src"):
                self.urls.append(value)

    handle_startendtag = handle_starttag


def local_target(site, page, url):
    """Resolve a URL from canonical site/page paths and reject artifact escapes."""
    parts = urlsplit(url)
    if parts.scheme and parts.scheme not in {
        "https",
        "http",
        "mailto",
        "tel",
        "javascript",
        "data",
    }:
        raise ValueError(
            f"{page.relative_to(site)}: unsupported URL scheme (unresolved Rust path?): {url}"
        )
    if parts.scheme or parts.netloc:
        return None
    path = unquote(parts.path)
    target = (
        (
            site / path.lstrip("/") if path.startswith("/") else page.parent / path
        ).resolve()
        if path
        else page
    )
    if not target.is_relative_to(site):
        raise ValueError(f"{page.relative_to(site)}: link escapes the site: {url}")
    if target.is_dir():
        target /= "index.html"
    return target, parts.fragment


class Rewrite(HTMLParser):
    """Rewrite URL attributes only inside actual HTML tags, preserving SVG/case/code."""

    def __init__(self, text, convert):
        super().__init__(convert_charrefs=False)
        self.lines = text.splitlines(keepends=True)
        self.offsets = [0]
        for line in self.lines:
            self.offsets.append(self.offsets[-1] + len(line))
        self.changes = []
        self.convert = convert
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        original = self.get_starttag_text()
        converted = ATTR.sub(
            lambda m: (
                f"{m[1]}={m[2]}{html.escape(self.convert(html.unescape(m[3])), quote=True)}{m[2]}"
            ),
            original,
        )
        if converted != original:
            line, col = self.getpos()
            start = self.offsets[line - 1] + col
            self.changes.append((start, start + len(original), converted))

    handle_startendtag = handle_starttag


class ExportLinks(HTMLParser):
    """Read named exports from generated module indexes, including aliases."""

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.exports = []
        self.current = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            values = dict(attrs)
            kinds = {
                "enum",
                "struct",
                "trait",
                "fn",
                "type",
                "macro",
                "mod",
                "constant",
                "static",
                "union",
            }
            if kinds.intersection((values.get("class") or "").split()) and values.get(
                "href"
            ):
                self.current = (values["href"], [])
            else:
                self.current = None

    def handle_data(self, text):
        if self.current is not None:
            self.current[1].append(text)

    def handle_endtag(self, tag):
        if tag == "a" and self.current is not None:
            href, text = self.current
            name = "".join(text).strip()
            if re.fullmatch(r"[A-Za-z_]\w*", name):
                self.exports.append((name, href))
            self.current = None


def assemble(root, work, prover, verifier):
    """Add independent API trees and resolve links to tracked source resources."""
    manifest = json.loads((work / "manifest.json").read_text())
    if manifest["schema"] != "tessera.docs-build/1":
        raise ValueError("unsupported documentation manifest")
    site = (work / "site").resolve()
    chapters = {Path(p).with_suffix(".html") for p in manifest["chapters"]}
    missing = sorted(str(p) for p in chapters if not (site / p).is_file())
    if missing:
        raise ValueError(f"missing rendered chapters: {', '.join(missing)}")
    files = tracked(root)
    book_pages = list(site.rglob("*.html"))
    for source, name, entry in (
        (prover, "prover", "tessera_prover"),
        (verifier, "verifier", "tv_core"),
    ):
        if not (source / entry / "index.html").is_file():
            raise ValueError(f"missing {name} rustdoc: {source / entry / 'index.html'}")
        shutil.copytree(source, site / "api" / name)
        # cargo doc may omit a workspace index unless unstable index generation is enabled.
        index = site / "api" / name / "index.html"
        entries = sorted(
            p.parent.name
            for p in (site / "api" / name).glob("*/index.html")
            if p.parent.name != "src"
        )
        index.write_text(
            '<!doctype html><meta charset="utf-8"><title>API index</title><h1>API index</h1><ul>'
            + "".join(f'<li><a href="{p}/index.html">{p}</a></li>' for p in entries)
            + "</ul>"
        )

    def convert(page, url):
        resolved = local_target(site, page, url)
        if resolved is None:
            return url
        target, fragment = resolved
        relative = target.relative_to(site)
        if relative in chapters and not target.is_file():
            raise ValueError(
                f"{page.relative_to(site)}: missing rendered chapter: {url}"
            )
        if target.is_file():
            return url
        # A directory URL refers to its canonical Markdown landing page, not a raw listing.
        parent = relative.parent if relative.name == "index.html" else relative
        if str(parent) in manifest["directories"]:
            mapped = site / manifest["directories"][str(parent)]
            return quote(relpath(mapped, page.parent)) + (
                "#" + quote(fragment) if fragment else ""
            )
        source = relative.with_suffix(".md") if relative.suffix == ".html" else relative
        if source in files:
            if source.suffix == ".md":
                return (
                    f"https://github.com/AbdelStark/tessera/blob/{manifest['commit']}/{quote(source.as_posix())}"
                    + ("#" + quote(fragment) if fragment else "")
                )
            if (root / source).is_symlink():
                raise ValueError(f"{source}: linked resources must not be symlinks")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / source, target)
            return url
        directory = relative.parent if relative.name == "index.html" else relative
        if any(p.is_relative_to(directory) for p in files):
            return f"https://github.com/AbdelStark/tessera/tree/{manifest['commit']}/{quote(directory.as_posix())}"
        raise ValueError(
            f"{page.relative_to(site)}: unresolved documentation link: {url}"
        )

    for page in book_pages:
        text = page.read_text()
        rewrite = Rewrite(text, lambda url, current=page: convert(current, url))
        for start, end, replacement in reversed(rewrite.changes):
            text = text[:start] + replacement + text[end:]
        page.write_text(text)
    repair_api(root, site)
    check(site)


def resolve_symbol(url, symbols, site, pages):
    """Resolve only unique exports or associated items with real rendered anchors."""
    if url in symbols:
        return next(iter(symbols[url])) if len(symbols[url]) == 1 else None
    parent, separator, member = url.rpartition("::")
    if not separator or not re.fullmatch(r"[A-Za-z_]\w*", member):
        return None
    base = resolve_symbol(parent, symbols, site, pages)
    if base is None:
        return None
    target, fragment = base
    visited = set()
    while True:
        if target in visited:
            raise ValueError(f"Rustdoc symbol redirect cycle: {url}")
        visited.add(target)
        if target not in pages:
            pages[target] = Page(target.read_text())
        parsed = pages[target]
        if not parsed.redirect:
            break
        redirected = local_target(site, target, parsed.redirect)
        if redirected is None:
            return None
        target, next_fragment = redirected
        fragment = next_fragment or fragment
    candidates = set()
    kinds = {
        "variant",
        "method",
        "tymethod",
        "associatedconstant",
        "associatedtype",
        "structfield",
    }
    for anchor in parsed.ids:
        if fragment:
            prefix = fragment + ".field."
            if not anchor.startswith(prefix):
                continue
            actual = anchor[len(prefix) :]
        else:
            kind, dot, actual = anchor.partition(".")
            if not dot or kind not in kinds:
                continue
        # Rustdoc suffixes duplicate method names. Keep both candidates so a
        # bare path cannot silently select one of several trait implementations.
        actual = re.sub(r"-[0-9]+$", "", actual)
        if actual == member:
            candidates.add((target, anchor))
    symbols[url] = candidates
    return next(iter(candidates)) if len(candidates) == 1 else None


def repair_api(root, site):
    """Repair rustdoc's inherited relative links; all destinations are checked next."""
    site = site.resolve()
    for name in ("prover", "verifier"):
        api = site / "api" / name
        symbols = {}
        associated_pages = {}
        for index in api.rglob("index.html"):
            namespace = "::".join(index.parent.relative_to(api).parts)
            for exported, href in ExportLinks(index.read_text()).exports:
                resolved = local_target(site, index, href)
                if resolved is not None and resolved[0].is_file():
                    symbols.setdefault(namespace + "::" + exported, set()).add(resolved)
        for page in api.rglob("*.html"):
            text = page.read_text()
            parsed = Page(text)

            def fix(
                url,
                page=page,
                api=api,
                parsed=parsed,
                symbols=symbols,
                associated_pages=associated_pages,
            ):
                # These references describe cfg-disabled optional dependency
                # features. Their upstream pages include those features; keep
                # local API builds on the workspace's actual default features.
                optional = {
                    "Visit::record_value": "tracing_core/field/trait.Visit.html#method.record_value",
                    "valuable::Valuable::as_value": "valuable/trait.Valuable.html#tymethod.as_value",
                }
                if (
                    page
                    in {
                        api / "tracing_core/field/index.html",
                        api / "tracing/field/index.html",
                    }
                    and url in optional
                ):
                    if url == "Visit::record_value":
                        lock = tomllib.loads((root / "Cargo.lock").read_text())
                        version = next(
                            p["version"]
                            for p in lock["package"]
                            if p["name"] == "tracing-core"
                        )
                        package = f"tracing-core/{version}"
                    else:
                        # Documentation-only target, not an enabled dependency.
                        package = "valuable/0.1.1"
                    return f"https://docs.rs/{package}/{optional[url]}"
                # Some rustdoc builds retain an explicit intra-doc Rust path as
                # the URL. Only resolve uniquely indexed exports; never guess.
                resolved = resolve_symbol(url, symbols, site, associated_pages)
                if resolved is not None:
                    target, fragment = resolved
                    return relpath(target, page.parent) + (
                        "#" + fragment if fragment else ""
                    )
                inherited = {
                    "dispatcher#setting-the-default-subscriber": "tracing/dispatcher/index.html#setting-the-default-subscriber",
                    "../struct.EnteredSpan.html": "tracing/span/struct.EnteredSpan.html",
                    "../struct.Metadata.html": "tracing/struct.Metadata.html",
                    "tracing/#recording-fields": "tracing/index.html#recording-fields",
                }
                if url in inherited:
                    path, separator, fragment = inherited[url].partition("#")
                    target = api / path
                    if not target.is_file():
                        raise ValueError(
                            f"missing tracing API target: {target}; build cargo doc -p tracing --no-deps"
                        )
                    return relpath(target, page.parent) + (
                        separator + fragment if separator else ""
                    )
                if url == "macro@crate::instrument":
                    lock = tomllib.loads((root / "Cargo.lock").read_text())
                    version = next(
                        p["version"] for p in lock["package"] if p["name"] == "tracing"
                    )
                    return f"https://docs.rs/tracing/{version}/tracing/attr.instrument.html"
                if (
                    page == api / "tracing_core/struct.Once.html"
                    and url == "struct.Mutex.html#poisoning"
                ):
                    toolchain = tomllib.loads(
                        (root / "rust-toolchain.toml").read_text()
                    )
                    channel = toolchain["toolchain"]["channel"]
                    return f"https://doc.rust-lang.org/{channel}/std/sync/struct.Mutex.html#poisoning"
                if page == api / "tracing_core/dispatcher/index.html" and url in {
                    "#fn.with_default",
                    "#fn.set_global_default",
                }:
                    return url[1:] + ".html"
                if url.startswith("#tymethod."):
                    actual = url[1:].replace("tymethod.", "method.", 1)
                    if url[1:] not in parsed.ids and actual in parsed.ids:
                        return "#" + actual
                # rustdoc sometimes requests an empty out-of-line implementation
                # registry without emitting the file. Never synthesize data for a
                # trait whose page lists an implementation.
                if urlsplit(url).scheme:
                    # Leave unresolved schemes for check(), which reports every
                    # broken reference with its source rather than stopping here.
                    return url
                resolved = local_target(site, page, url)
                if resolved is not None:
                    target, fragment = resolved
                    if (
                        fragment.startswith("tymethod.")
                        and target.is_file()
                        and target.suffix == ".html"
                    ):
                        if target not in associated_pages:
                            associated_pages[target] = Page(target.read_text())
                        ids = associated_pages[target].ids
                        actual = fragment.replace("tymethod.", "method.", 1)
                        if fragment not in ids and actual in ids:
                            return relpath(target, page.parent) + "#" + actual
                    empty_registry = (
                        page.name.startswith("trait.")
                        and target.name == page.with_suffix(".js").name
                        and target.is_relative_to(api / "trait.impl")
                        and not any(i.startswith("impl-") for i in parsed.ids)
                    )
                    if empty_registry and not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(
                            "// rustdoc emitted no out-of-line implementors.\n(function(){const empty={};if(window.register_implementors){window.register_implementors(empty);}else{window.pending_implementors=empty;}})();\n"
                        )
                return url

            rewrite = Rewrite(text, fix)
            for start, end, replacement in reversed(rewrite.changes):
                text = text[:start] + replacement + text[end:]
            page.write_text(text)


def anchor_exists(target, fragment, pages):
    """Validate normal IDs and rustdoc's JavaScript source line selections."""
    page = pages[target]
    if fragment in page.ids or unquote(fragment) in page.ids:
        return True
    source = target.name.endswith(".rs.html") and any(
        "src-script-" in u for u in page.urls
    )
    match = re.fullmatch(r"([0-9]+)-([0-9]+)", fragment) if source else None
    if match:
        start, end = map(int, match.groups())
        return (
            start <= end
            and end - start < len(page.ids)
            and all(str(i) in page.ids for i in range(start, end + 1))
        )
    return False


def check(site):
    """Check every rendered HTML URL, including API links, fragments and images, offline."""
    site = site.resolve()
    pages = {p: Page(p.read_text()) for p in site.rglob("*.html")}
    if not pages:
        raise ValueError(f"{site}: no HTML pages to check")
    errors = []
    for path, page in pages.items():
        for url in page.urls + ([page.redirect] if page.redirect else []):
            try:
                resolved = local_target(site, path, url)
                if resolved is None:
                    continue
                target, fragment = resolved
                visited = set()
                while target in pages and pages[target].redirect:
                    if target in visited:
                        raise ValueError("redirect cycle")
                    visited.add(target)
                    next_target = local_target(site, target, pages[target].redirect)
                    if next_target is None:
                        target = None
                        break
                    target, next_fragment = next_target
                    fragment = next_fragment or fragment
                if target is None:
                    continue
                if not target.is_file():
                    raise ValueError("missing file")
                if (
                    fragment
                    and target in pages
                    and not anchor_exists(target, fragment, pages)
                ):
                    raise ValueError(f"missing anchor #{fragment}")
            except ValueError as exc:
                errors.append(f"{path.relative_to(site)}: {url}: {exc}")
    if errors:
        raise ValueError("broken rendered links:\n" + "\n".join(sorted(set(errors))))
    print(f"checked {len(pages)} HTML pages: no broken local links")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "assemble", "check"))
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".docs-build")
    parser.add_argument("--prover-doc", type=Path)
    parser.add_argument("--verifier-doc", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(ROOT, args.work_dir)
        elif args.command == "assemble":
            assemble(
                ROOT,
                args.work_dir,
                args.prover_doc or args.work_dir / "target/prover/doc",
                args.verifier_doc or args.work_dir / "target/verifier/doc",
            )
        else:
            check(args.work_dir / "site")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"documentation: {exc}\n")


if __name__ == "__main__":
    main()
