#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 OR MIT
#
# One unit of the Tessera autoresearch loop, run headlessly, so that a scheduler outside any
# interactive session (cron, launchd, a CI timer) can keep the loop going. Each run does what
# `/autoresearch` does inside a session: take the first feasible queue item of research/LEDGER.md,
# work it, consolidate, and leave research/ consistent (research/tools/check_kb.py and
# `cargo xtask lint` pass). Runs compose because each is bounded and ends with the ledger consistent.
#
# Usage, from the repository root (a worktree on the research branch):
#     research/tools/loop.sh                 # one unit of work
#     LOOP_BUDGET_USD=40 research/tools/loop.sh
#     LOOP_MODE=sweep research/tools/loop.sh # the parallel sweep instead of one unit (hours, millions of tokens)
#
# Scheduling it is the maintainer's choice; two examples. Every 6 hours with cron:
#     13 */6 * * * cd /path/to/worktree && research/tools/loop.sh >> "$HOME/.cache/tessera-autoresearch/cron.log" 2>&1
# or the launchd agent in launchd/com.tessera.autoresearch.plist (StartInterval 21600), whose header
# gives the install and remove commands.
#
# The run never commits, checks out, stashes, rebases, merges or pushes, and deletes nothing; the
# maintainer reviews `git status` and commits through the issue workflow. Logs go outside the
# repository.

set -eu

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

LOG_DIR=${LOOP_LOG_DIR:-"$HOME/.cache/tessera-autoresearch"}
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%dT%H%M%S)
LOG="$LOG_DIR/run-$STAMP.log"
BUDGET=${LOOP_BUDGET_USD:-25}
MODE=${LOOP_MODE:-unit}

if [ "$MODE" = "sweep" ]; then
  TASK='Run the saved workflow autoresearch-sweep (.claude/workflows/autoresearch-sweep.js) with its defaults, then make sure research/tools/check_kb.py and cargo xtask lint pass.'
else
  TASK='Run one unit of the Tessera autoresearch loop: invoke the autoresearch skill with no arguments (read research/LEDGER.md, take the first feasible queue item, do that unit of work, consolidate), then run python3 -I research/tools/check_kb.py and cargo xtask lint until both pass.'
fi

PROMPT="$TASK Work only inside this worktree. Make no git commit, checkout, stash, rebase, merge or push and delete nothing; move with /bin/mv -f and copy with /bin/cp -f. Finish with a short report: what the unit produced with evidence levels, what changed in the ledger, and the queue's first item now."

# Tools the headless run may use without a prompt, one argument each (a pattern such as
# "Bash(python3 *)" contains a space and must stay one argument). Narrow this list to taste; a
# tool outside it is refused and the run reports what it could not do.
set -- Read Glob Grep Edit Write WebFetch WebSearch Skill Agent Workflow \
  "Bash(python3 *)" "Bash(cargo *)" "Bash(ls *)" "Bash(cat *)" "Bash(sed *)" "Bash(grep *)" \
  "Bash(awk *)" "Bash(find *)" "Bash(wc *)" "Bash(date *)" "Bash(mkdir *)" "Bash(/bin/cp *)" \
  "Bash(/bin/mv *)" "Bash(gh api *)" "Bash(gh issue view *)" "Bash(gh issue list *)" \
  "Bash(git status *)" "Bash(git diff *)" "Bash(git log *)" "Bash(git fetch *)"

echo "autoresearch loop: $MODE, $(date -u +%FT%TZ), log $LOG"
claude -p "$PROMPT" \
  --permission-mode acceptEdits \
  --allowedTools "$@" \
  --max-budget-usd "$BUDGET" \
  --output-format text \
  2>&1 | tee "$LOG"

echo "autoresearch loop: done, $(date -u +%FT%TZ); review with: git -C $ROOT status --short"
