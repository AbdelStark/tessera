// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Agent context policy: `.agents/` is the single source of truth for skills and workflows,
//! `AGENTS.md` indexes every skill and workflow, `CLAUDE.md` delegates to `@AGENTS.md`, and
//! `.claude/` and `.codex/` mirror `.agents/` through relative symlinks.

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const MIRRORS: &[&str] = &[".claude", ".codex"];

/// Checks the agent context layout under `repo` and returns one message per violation.
pub(crate) fn check(repo: &Path) -> io::Result<Vec<String>> {
    let mut violations = Vec::new();

    let agents_md_path = repo.join("AGENTS.md");
    let agents_md = match fs::read_to_string(&agents_md_path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            violations.push("AGENTS.md: missing".to_owned());
            String::new()
        }
        Err(e) => return Err(e),
    };

    let claude_md_path = repo.join("CLAUDE.md");
    match fs::read_to_string(&claude_md_path) {
        Ok(text) => {
            if !text.lines().any(|line| line.trim() == "@AGENTS.md") {
                violations.push("CLAUDE.md: must reference `@AGENTS.md`".to_owned());
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            violations.push("CLAUDE.md: missing".to_owned());
        }
        Err(e) => return Err(e),
    }

    let skills = check_skills(repo, &agents_md, &mut violations)?;
    let workflows = check_workflows(repo, &agents_md, &mut violations)?;

    for mirror in MIRRORS {
        check_mirror(repo, mirror, "skills", &skills, &mut violations)?;
        check_mirror(repo, mirror, "workflows", &workflows, &mut violations)?;
    }

    Ok(violations)
}

fn check_skills(
    repo: &Path,
    agents_md: &str,
    violations: &mut Vec<String>,
) -> io::Result<BTreeSet<String>> {
    let mut skills = BTreeSet::new();
    let skills_dir = repo.join(".agents/skills");
    let entries = match fs::read_dir(&skills_dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            violations.push(".agents/skills: missing directory".to_owned());
            return Ok(skills);
        }
        Err(e) => return Err(e),
    };

    for entry in entries {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let rel = format!(".agents/skills/{name}");
        let ftype = entry.file_type()?;
        if !ftype.is_dir() {
            violations.push(format!("{rel}: must be a directory"));
            continue;
        }
        skills.insert(name.clone());

        let skill_md = entry.path().join("SKILL.md");
        let skill_md_rel = format!("{rel}/SKILL.md");
        match fs::read_to_string(&skill_md) {
            Ok(text) => check_skill_frontmatter(&skill_md_rel, &name, &text, violations),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                violations.push(format!("{skill_md_rel}: missing"));
            }
            Err(e) => return Err(e),
        }

        if !agents_md.is_empty() && !agents_md.contains(&name) {
            violations.push(format!(
                "AGENTS.md: must index skill `{name}` ({skill_md_rel})"
            ));
        }
    }

    if skills.is_empty() {
        violations.push(".agents/skills: must contain at least one skill".to_owned());
    }

    Ok(skills)
}

fn check_skill_frontmatter(
    rel: &str,
    expected_name: &str,
    text: &str,
    violations: &mut Vec<String>,
) {
    let Some(rest) = text.strip_prefix("---\n") else {
        violations.push(format!("{rel}: must start with YAML frontmatter `---`"));
        return;
    };
    let Some((frontmatter, _body)) = rest.split_once("\n---\n") else {
        violations.push(format!("{rel}: unclosed YAML frontmatter `---`"));
        return;
    };
    let mut found_name = None;
    let mut found_description = false;
    for line in frontmatter.lines() {
        if let Some(val) = line.strip_prefix("name:") {
            found_name = Some(val.trim());
        } else if let Some(val) = line.strip_prefix("description:")
            && !val.trim().is_empty()
        {
            found_description = true;
        }
    }
    match found_name {
        Some(actual) if actual == expected_name => {}
        Some(actual) => violations.push(format!(
            "{rel}: frontmatter `name: {actual}` does not match directory `{expected_name}`"
        )),
        None => violations.push(format!(
            "{rel}: frontmatter missing `name: {expected_name}`"
        )),
    }
    if !found_description {
        violations.push(format!(
            "{rel}: frontmatter missing non-empty `description:`"
        ));
    }
}

fn check_workflows(
    repo: &Path,
    agents_md: &str,
    violations: &mut Vec<String>,
) -> io::Result<BTreeSet<String>> {
    let mut workflows = BTreeSet::new();
    let wf_dir = repo.join(".agents/workflows");
    let entries = match fs::read_dir(&wf_dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            violations.push(".agents/workflows: missing directory".to_owned());
            return Ok(workflows);
        }
        Err(e) => return Err(e),
    };

    for entry in entries {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let rel = format!(".agents/workflows/{name}");
        let ftype = entry.file_type()?;
        if !ftype.is_file() || Path::new(&name).extension().is_none_or(|ext| ext != "js") {
            violations.push(format!("{rel}: must be a `.js` workflow file"));
            continue;
        }
        workflows.insert(name.clone());
        if !agents_md.is_empty() && !agents_md.contains(&name) {
            violations.push(format!("AGENTS.md: must index workflow `{name}` ({rel})"));
        }
    }

    Ok(workflows)
}

fn check_mirror(
    repo: &Path,
    mirror: &str,
    kind: &str,
    expected: &BTreeSet<String>,
    violations: &mut Vec<String>,
) -> io::Result<()> {
    let dir = repo.join(mirror).join(kind);
    let entries = match fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            violations.push(format!("{mirror}/{kind}: missing directory"));
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    let mut seen = BTreeSet::new();
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        seen.insert(name.clone());
        let rel = format!("{mirror}/{kind}/{name}");
        if !expected.contains(&name) {
            violations.push(format!(
                "{rel}: unexpected entry not present in `.agents/{kind}/`"
            ));
            continue;
        }
        let ftype = entry.file_type()?;
        if !ftype.is_symlink() {
            violations.push(format!(
                "{rel}: must be a symlink to `../../.agents/{kind}/{name}` (found regular file or directory)"
            ));
            continue;
        }
        let target = fs::read_link(entry.path())?;
        let expected_target = PathBuf::from(format!("../../.agents/{kind}/{name}"));
        if target != expected_target {
            violations.push(format!(
                "{rel}: symlink points to `{}`, expected `{}`",
                target.display(),
                expected_target.display()
            ));
        }
    }

    for name in expected {
        if !seen.contains(name) {
            violations.push(format!(
                "{mirror}/{kind}/{name}: missing symlink to `../../.agents/{kind}/{name}`"
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Fixture;

    #[cfg(unix)]
    fn symlink(target: &str, link: &Path) {
        std::os::unix::fs::symlink(target, link).expect("create symlink");
    }

    #[test]
    #[cfg(unix)]
    fn valid_agent_context_passes() {
        let fixture = Fixture::new("agent-context-ok");
        fixture.write(
            "AGENTS.md",
            "# AGENTS\nUses `alpha-skill` and `sweep.js`.\n",
        );
        fixture.write("CLAUDE.md", "@AGENTS.md\n");
        fixture.write(
            ".agents/skills/alpha-skill/SKILL.md",
            "---\nname: alpha-skill\ndescription: Test skill.\n---\n\n# Alpha\n",
        );
        fixture.write(".agents/workflows/sweep.js", "export default {};\n");

        for mirror in MIRRORS {
            fs::create_dir_all(fixture.path().join(mirror).join("skills")).expect("mkdir");
            fs::create_dir_all(fixture.path().join(mirror).join("workflows")).expect("mkdir");
            symlink(
                "../../.agents/skills/alpha-skill",
                &fixture.path().join(mirror).join("skills/alpha-skill"),
            );
            symlink(
                "../../.agents/workflows/sweep.js",
                &fixture.path().join(mirror).join("workflows/sweep.js"),
            );
        }

        let violations = check(fixture.path()).expect("readable");
        assert!(
            violations.is_empty(),
            "unexpected violations: {violations:?}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn drift_and_non_symlinks_are_reported() {
        let fixture = Fixture::new("agent-context-bad");
        fixture.write("AGENTS.md", "# AGENTS\n");
        fixture.write("CLAUDE.md", "# Not referencing agents\n");
        fixture.write(
            ".agents/skills/alpha-skill/SKILL.md",
            "---\nname: wrong-name\ndescription: \n---\n",
        );
        fixture.write(".agents/workflows/sweep.js", "export default {};\n");
        // Regular file in .claude/skills instead of a symlink
        fixture.write(".claude/skills/alpha-skill/SKILL.md", "copied file\n");
        fixture.write(".claude/skills/stray/SKILL.md", "stray\n");

        let violations = check(fixture.path()).expect("readable");
        assert!(
            violations.iter().any(|v| v.contains("CLAUDE.md")),
            "missing CLAUDE.md check: {violations:?}"
        );
        assert!(
            violations
                .iter()
                .any(|v| v.contains("does not match directory `alpha-skill`")),
            "missing frontmatter name check: {violations:?}"
        );
        assert!(
            violations
                .iter()
                .any(|v| v.contains("missing non-empty `description:`")),
            "missing frontmatter description check: {violations:?}"
        );
        assert!(
            violations
                .iter()
                .any(|v| v.contains("AGENTS.md: must index skill `alpha-skill`")),
            "missing AGENTS.md index check: {violations:?}"
        );
        assert!(
            violations
                .iter()
                .any(|v| v.contains(".claude/skills/alpha-skill: must be a symlink")),
            "missing non-symlink check: {violations:?}"
        );
        assert!(
            violations
                .iter()
                .any(|v| v.contains(".claude/skills/stray: unexpected entry")),
            "missing stray entry check: {violations:?}"
        );
        assert!(
            violations
                .iter()
                .any(|v| v.contains(".codex/skills: missing directory")),
            "missing .codex/skills check: {violations:?}"
        );
    }
}
