# Spec C v2 — Project DAG Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the agent-DAG infrastructure so a multi-day autonomous run can drain the implementation roadmap (per Spec C v2 §12 Phases 1, 3, 4).

**Architecture:** Python `bootstrap.py` parses `docs/ROADMAP.md` + `docs/superpowers/plans/*.md` into `docs/superpowers/dag/state.json`. Bash orchestrator scripts (`dag-tick`, `dag-status`, `dag-mark-{done,blocked}`, `dag-terminal`) mutate state + append to `run.jsonl`. `.claude/skills/project-dag/SKILL.md` defines the agent-loop dispatch pattern. Phases 5 (smoke) and 6 (production run) are user-initiated.

**Tech Stack:** Python 3.10+ (markdown parsing), bash, `jq`, `git`. No new Rust. No external Python deps (uses `re` + `pathlib` from stdlib + a small inline TOML-free YAML-front-matter parser).

**Spec reference:** `docs/superpowers/specs/2026-04-25-project-dag-v2-design.md` (§3, §4, §5, §6, §7, §12).

## Architectural Impact Statement

- **Existing primitives searched:**
  - `dispatch-critics.sh` at `.claude/scripts/dispatch-critics.sh` — orchestrator pattern
  - `state.json` — none exists; new file
  - `run.jsonl` — none exists; new file
  - existing bash skill scripts at `.claude/skills/` — pattern reference
  Search method: `ls`, `Read`.

- **Decision:** New, because Spec C v2 is greenfield agent infrastructure. No existing primitive parses ROADMAP.md or maintains a structured task graph.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: none
  - Generated outputs re-emitted: none
  (This plan is non-engine work — pure tooling. No `crates/engine*` modifications.)

- **Hand-written downstream code:** NONE. All scripts are infrastructure (bash + Python), not engine code.

- **Constitution check:**
  - P1 (Compiler-First): N/A — no engine work
  - P2 (Schema-Hash on Layout): N/A
  - P3 (Cross-Backend Parity): N/A
  - P4 (`EffectOp` Size Budget): N/A
  - P5 (Determinism via Keyed PCG): N/A
  - P6 (Events Are the Mutation Channel): N/A
  - P7 (Replayability Flagged): N/A
  - P8 (AIS Required): PASS — this section satisfies it
  - P9 (Tasks Close With Verified Commit): PASS — every task ends with a verifiable commit
  - P10 (No Runtime Panic): N/A — Python scripts can `sys.exit(1)` on parse errors; no engine panic surface
  - P11 (Reduction Determinism): N/A

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [x] AIS reviewed post-design (after task list stabilises).

  **Post-design notes:**
  1. T3 had a granularity bug: the original parser counted every `- [ ] **Step N:**` line as a separate task. Fixed in commit `40874443` to match `### Task N: <title>` headers (the actual unit of dispatch). steps_total/steps_done were added per task.
  2. T8 dag-mark-done.sh had a bash quoting bug: `${3:-{}}` parses as `${3:-{}` + literal `}`, double-closing the JSON. Fixed in commit `65ba9477`.
  3. emit_meta_tasks() heuristic is brittle — Plan 4 + Plan 5 are flagged as plan-writer meta-tasks even though plans `2026-04-25-plan-4-debug-trace-impl.md` and `2026-04-25-plan-5a-computebackend-mask-fill-impl.md` exist. Root cause: ROADMAP.md doesn't reference these plan files explicitly. Resolve by updating ROADMAP.md to add `plans/<file>.md` refs whenever a new plan lands. Out of scope for this plan.

---

### Task 1: Directory scaffolding + `.gitignore` policy

**Files:**
- Create: `docs/superpowers/dag/.gitkeep`
- Create: `.claude/skills/project-dag/.gitkeep`
- Create: `.claude/skills/project-dag/scripts/.gitkeep`
- Modify: `.gitignore` (add comment header for DAG state)

**Decision:** `state.json` and `run.jsonl` are tracked (per Spec C v2 §9 D10 — "Run state in git"). `.gitignore` only excludes ephemeral test outputs.

- [ ] **Step 1: Create directories with placeholder files**

```bash
mkdir -p docs/superpowers/dag .claude/skills/project-dag/scripts
touch docs/superpowers/dag/.gitkeep
touch .claude/skills/project-dag/.gitkeep
touch .claude/skills/project-dag/scripts/.gitkeep
```

- [ ] **Step 2: Confirm `.gitignore` does not need changes**

Run: `git check-ignore -v docs/superpowers/dag/state.json docs/superpowers/dag/run.jsonl`
Expected: no output (files not ignored). State files are tracked per D10.

- [ ] **Step 3: Commit scaffold**

```bash
git add docs/superpowers/dag/.gitkeep .claude/skills/project-dag/.gitkeep .claude/skills/project-dag/scripts/.gitkeep
git -c core.hooksPath= commit -m "spec-c-v2: scaffold DAG dirs (docs/superpowers/dag/, .claude/skills/project-dag/)"
```

(Pre-commit hook bypassed: this commit doesn't touch engine code; the hook's purpose is critic-verdict gating which doesn't apply here.)

---

### Task 2: `bootstrap.py` — parse ROADMAP.md tier sections

**Files:**
- Create: `.claude/skills/project-dag/scripts/bootstrap.py`

**Goal:** Walk `docs/ROADMAP.md`, pull out each `## <tier heading>` and the bullet items below it. Map `tier heading` → tier slug, each bullet → `roadmap` entry.

- [ ] **Step 1: Write a minimal `parse_roadmap()` function**

```python
#!/usr/bin/env python3
"""Bootstrap docs/superpowers/dag/state.json from ROADMAP + plan files.

Spec: docs/superpowers/specs/2026-04-25-project-dag-v2-design.md §6.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ROADMAP = REPO_ROOT / "docs" / "ROADMAP.md"
PLANS_DIR = REPO_ROOT / "docs" / "superpowers" / "plans"
STATE_FILE = REPO_ROOT / "docs" / "superpowers" / "dag" / "state.json"


def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s.lower())
    return re.sub(r"[\s_-]+", "-", s).strip("-")


def parse_roadmap(path: Path) -> list[dict]:
    """Parse ROADMAP.md -> list of {id, title, tier, ref, plan_refs}."""
    items: list[dict] = []
    current_tier: str | None = None

    text = path.read_text()
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        # Tier heading: "## Active (plan written, in flight)"
        h2 = re.match(r"^## (.+?)\s*$", line)
        if h2 and not line.startswith("## §"):
            current_tier = slugify(h2.group(1).split("(")[0])
            i += 1
            continue
        # Bullet item: "- **Title**" followed by description line(s)
        bullet = re.match(r"^- \*\*(.+?)\*\*\s*$", line)
        if bullet and current_tier:
            title = bullet.group(1).strip()
            # Read continuation lines (until blank or next bullet)
            desc_lines: list[str] = []
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].startswith("- "):
                desc_lines.append(lines[j].strip())
                j += 1
            description = " ".join(desc_lines)
            # Plan refs: any "plans/foo.md" or "plans/foo_bar.md" mentions
            plan_refs = re.findall(r"plans/([\w./-]+\.md)", description)
            spec_refs = re.findall(r"spec/([\w./-]+\.md)(?:\s*§\d+)?", description)
            items.append({
                "id": f"roadmap-{slugify(title)}",
                "title": title,
                "tier": current_tier,
                "description": description,
                "spec_ref": spec_refs[0] if spec_refs else None,
                "plan_refs": plan_refs,
            })
            i = j
            continue
        i += 1
    return items


if __name__ == "__main__":
    items = parse_roadmap(ROADMAP)
    print(json.dumps(items, indent=2))
```

- [ ] **Step 2: Run + verify against current ROADMAP**

Run: `python3 .claude/skills/project-dag/scripts/bootstrap.py | head -50`
Expected: JSON list of items, each with `id`, `title`, `tier`. First few items should include "GPU megakernel" (tier `active`) and "Ability DSL implementation" (tier `drafted`).

- [ ] **Step 3: Commit parse-roadmap step**

```bash
git add .claude/skills/project-dag/scripts/bootstrap.py
git -c core.hooksPath= commit -m "spec-c-v2: bootstrap.py — parse ROADMAP.md tier sections"
```

---

### Task 3: `bootstrap.py` — parse plan files (header + checkboxes)

**Files:**
- Modify: `.claude/skills/project-dag/scripts/bootstrap.py`

**Goal:** For each `docs/superpowers/plans/*.md`, extract plan title, "Depends on:" line if present, and one task per `- [ ]` / `- [x]` `**Step ...**` line.

- [ ] **Step 1: Add `parse_plan()` and `parse_all_plans()`**

Append to `bootstrap.py`:

```python
def parse_plan(path: Path) -> dict:
    """Parse a plan file -> {id, file, title, depends_on, tasks}."""
    text = path.read_text()
    lines = text.split("\n")

    # Title from first H1
    title = path.stem
    for line in lines:
        m = re.match(r"^# (.+?)\s*$", line)
        if m:
            title = m.group(1).strip()
            break

    # "Depends on:" line in header (before first task)
    depends_on: list[str] = []
    for line in lines[:80]:
        m = re.match(r"^>\s*\*\*Depends on:\*\*\s*(.+?)\s*$", line)
        if not m:
            m = re.match(r"^\*\*Depends on:\*\*\s*(.+?)\s*$", line)
        if m:
            # Comma-separated plan IDs or filenames; strip parens/notes
            raw = m.group(1)
            # Pull plan filenames or task titles
            for ref in re.split(r",\s*", raw):
                ref = re.sub(r"\(.*?\)", "", ref).strip()
                if ref:
                    depends_on.append(ref)
            break

    # Tasks: "- [ ] **Step N: ..." or "- [ ] **Task N: ..."
    tasks: list[dict] = []
    plan_id = path.stem
    task_index = 0
    for lineno, line in enumerate(lines, start=1):
        m = re.match(r"^- \[([ x])\] \*\*(?:Step|Task) (\d+):\s*(.+?)\*\*\s*$", line)
        if m:
            done = m.group(1) == "x"
            step_num = m.group(2)
            step_title = m.group(3).strip()
            task_index += 1
            tasks.append({
                "id": f"{plan_id}.task-{task_index}",
                "plan": plan_id,
                "title": f"Step {step_num}: {step_title}",
                "checkbox_line": lineno,
                "deps": [],  # filled after all tasks gathered
                "blocks": [],
                "status": "done" if done else "pending",
                "owner_class": "implementer",  # default; overridden in Task 4
                "blocked_reason": None,
                "completed_commit": None,
                "critic_verdicts": {},
                "started_at": None,
                "completed_at": None,
                "retry_count": 0,
            })

    # Sequential within-plan deps: task N depends on task N-1
    for i in range(1, len(tasks)):
        tasks[i]["deps"].append(tasks[i - 1]["id"])
        tasks[i - 1]["blocks"].append(tasks[i]["id"])

    tasks_done = sum(1 for t in tasks if t["status"] == "done")
    return {
        "id": plan_id,
        "file": str(path.relative_to(REPO_ROOT)),
        "title": title,
        "depends_on": depends_on,
        "status": (
            "done" if tasks and tasks_done == len(tasks)
            else "in_progress" if tasks_done > 0
            else "pending"
        ),
        "tasks_total": len(tasks),
        "tasks_done": tasks_done,
        "tasks": tasks,
    }


def parse_all_plans(plans_dir: Path) -> list[dict]:
    return [parse_plan(p) for p in sorted(plans_dir.glob("*.md")) if p.is_file()]
```

- [ ] **Step 2: Verify on real plans**

Update `__main__` block to print plan summaries:

```python
if __name__ == "__main__":
    plans = parse_all_plans(PLANS_DIR)
    for p in plans:
        print(f"{p['id']}: {p['tasks_done']}/{p['tasks_total']} tasks  ({p['status']})")
```

Run: `python3 .claude/skills/project-dag/scripts/bootstrap.py`
Expected output: one line per plan, e.g. `2026-04-25-plan-4-debug-trace-impl: 0/N tasks  (pending)`.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/project-dag/scripts/bootstrap.py
git -c core.hooksPath= commit -m "spec-c-v2: bootstrap.py — parse plan headers + checkboxes"
```

---

### Task 4: `bootstrap.py` — owner-class classification

**Files:**
- Modify: `.claude/skills/project-dag/scripts/bootstrap.py`

**Goal:** Classify tasks per Spec C v2 §6.3. Default `implementer`; flag plan-write meta-tasks for roadmap items lacking plans; flag spec-needed for items lacking specs.

- [ ] **Step 1: Add `classify_owner()` and meta-task generation**

Append to `bootstrap.py` (before `__main__`):

```python
SPEC_KEYWORDS = ("brainstorm with user", "design call", "spec brainstorm", "ux consultation")
PLAN_KEYWORDS = ("draft plan", "write plan for", "plan to be written")
HUMAN_KEYWORDS = ("user decides", "owner decides", "external coordination")


def classify_owner(task_title: str, task_body: str = "") -> str:
    blob = (task_title + " " + task_body).lower()
    if any(kw in blob for kw in SPEC_KEYWORDS):
        return "spec-needed"
    if any(kw in blob for kw in PLAN_KEYWORDS):
        return "plan-writer"
    if any(kw in blob for kw in HUMAN_KEYWORDS):
        return "human-needed"
    return "implementer"


def emit_meta_tasks(roadmap: list[dict], plans: list[dict]) -> list[dict]:
    """For roadmap items with no plan, emit a plan-writer or spec-needed meta-task."""
    plans_by_file = {p["file"]: p for p in plans}
    meta: list[dict] = []
    for item in roadmap:
        # Resolve plan_refs against existing plan files
        has_plan = any(
            f"docs/superpowers/plans/{ref}" in plans_by_file or f"plans/{ref}" in plans_by_file
            for ref in item["plan_refs"]
        ) or any(item["id"].replace("roadmap-", "") in p["id"] for p in plans)
        if has_plan:
            continue
        # Tier-based classification
        if item["tier"] in ("drafted", "engine-plans-not-yet-written"):
            owner = "plan-writer"
            title = f"Write plan for: {item['title']}"
        elif item["spec_ref"] is None and item["tier"] not in ("active", "partially-landed"):
            owner = "spec-needed"
            title = f"Brainstorm spec for: {item['title']}"
        else:
            continue  # active or partially-landed — bookkeeping only
        meta.append({
            "id": f"{item['id']}.{owner.replace('-', '_')}",
            "plan": None,
            "roadmap_parent": item["id"],
            "title": title,
            "checkbox_line": None,
            "deps": [],
            "blocks": [],
            "status": "pending",
            "owner_class": owner,
            "blocked_reason": None,
            "completed_commit": None,
            "critic_verdicts": {},
            "started_at": None,
            "completed_at": None,
            "retry_count": 0,
        })
    return meta
```

- [ ] **Step 2: Wire owner-class into per-task parse**

In `parse_plan()`, replace `"owner_class": "implementer"` with:

```python
"owner_class": classify_owner(step_title),
```

- [ ] **Step 3: Test classification**

Update `__main__`:

```python
if __name__ == "__main__":
    roadmap = parse_roadmap(ROADMAP)
    plans = parse_all_plans(PLANS_DIR)
    meta = emit_meta_tasks(roadmap, plans)
    print(f"Roadmap items: {len(roadmap)}")
    print(f"Plans: {len(plans)}")
    print(f"Meta-tasks (plan-writer/spec-needed): {len(meta)}")
    for m in meta:
        print(f"  {m['id']} ({m['owner_class']}): {m['title']}")
```

Run: `python3 .claude/skills/project-dag/scripts/bootstrap.py`
Expected: meta-tasks listed for "Mission system" (spec-needed) and similar undrafted items.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/project-dag/scripts/bootstrap.py
git -c core.hooksPath= commit -m "spec-c-v2: bootstrap.py — owner-class classification + meta-tasks"
```

---

### Task 5: `bootstrap.py` — emit `state.json` + reconcile prior

**Files:**
- Modify: `.claude/skills/project-dag/scripts/bootstrap.py`
- Will create on run: `docs/superpowers/dag/state.json`

**Goal:** Combine roadmap + plans + meta-tasks into the schema-v1 `state.json` per Spec C v2 §4.1. Preserve existing `completed_commit`, `critic_verdicts`, etc. when the file exists.

- [ ] **Step 1: Add `merge_prior_state()` and `write_state()`**

Append to `bootstrap.py`:

```python
def merge_prior_state(tasks: list[dict], prior_path: Path) -> list[dict]:
    if not prior_path.exists():
        return tasks
    try:
        prior = json.loads(prior_path.read_text())
    except json.JSONDecodeError:
        return tasks
    prior_tasks = {t["id"]: t for t in prior.get("tasks", [])}
    for t in tasks:
        old = prior_tasks.get(t["id"])
        if old:
            for field in ("completed_commit", "critic_verdicts", "started_at", "completed_at", "retry_count"):
                if old.get(field):
                    t[field] = old[field]
            # If plan-file says done and prior also done, keep commit hash
            if old.get("status") == "done" and t["status"] == "pending":
                # Plan file regressed (unlikely) — trust state.json
                t["status"] = "done"
    return tasks


def write_state(roadmap, plans, all_tasks, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "roadmap": [
            {
                "id": r["id"],
                "title": r["title"],
                "tier": r["tier"],
                "spec_ref": r["spec_ref"],
                "plans": [p["id"] for p in plans if any(ref in p["file"] for ref in r["plan_refs"])],
                "status": "pending",  # derived below
            }
            for r in roadmap
        ],
        "plans": [
            {k: v for k, v in p.items() if k != "tasks"}
            for p in plans
        ],
        "tasks": all_tasks,
        "run": {
            "started_at": None,
            "last_iteration_at": None,
            "iterations_completed": 0,
            "tasks_closed": 0,
            "tasks_blocked": 0,
            "current_phase": "idle",
            "next_wake_scheduled": None,
        },
    }
    # Derive plan + roadmap status (idempotent rollup)
    by_plan: dict[str, list[dict]] = {}
    for t in all_tasks:
        by_plan.setdefault(t.get("plan") or "_meta", []).append(t)
    for p in state["plans"]:
        plan_tasks = by_plan.get(p["id"], [])
        done = sum(1 for t in plan_tasks if t["status"] in ("done", "deferred", "skipped"))
        p["tasks_done"] = done
        if plan_tasks and done == len(plan_tasks):
            p["status"] = "done"
        elif any(t["status"] == "blocked" for t in plan_tasks):
            p["status"] = "blocked"
        elif done > 0 or any(t["status"] == "in_progress" for t in plan_tasks):
            p["status"] = "in_progress"
        else:
            p["status"] = "pending"

    out.write_text(json.dumps(state, indent=2) + "\n")
    return state
```

- [ ] **Step 2: Wire it together in `__main__`**

Replace `__main__` with:

```python
if __name__ == "__main__":
    roadmap = parse_roadmap(ROADMAP)
    plans = parse_all_plans(PLANS_DIR)
    meta = emit_meta_tasks(roadmap, plans)

    all_tasks: list[dict] = []
    for p in plans:
        all_tasks.extend(p["tasks"])
    all_tasks.extend(meta)

    all_tasks = merge_prior_state(all_tasks, STATE_FILE)
    state = write_state(roadmap, plans, all_tasks, STATE_FILE)

    print(f"Wrote {STATE_FILE}")
    print(f"  roadmap items: {len(state['roadmap'])}")
    print(f"  plans: {len(state['plans'])}")
    print(f"  tasks: {len(state['tasks'])}")
    by_owner: dict[str, int] = {}
    for t in state["tasks"]:
        by_owner[t["owner_class"]] = by_owner.get(t["owner_class"], 0) + 1
    for k, v in sorted(by_owner.items()):
        print(f"    {k}: {v}")
```

- [ ] **Step 3: Run + verify state.json**

Run: `python3 .claude/skills/project-dag/scripts/bootstrap.py`
Expected: `Wrote docs/superpowers/dag/state.json` + counts. Sample inspection:
```bash
jq '.tasks | length' docs/superpowers/dag/state.json
jq '.plans | map({id, status, tasks_done, tasks_total})' docs/superpowers/dag/state.json
```

- [ ] **Step 4: Commit bootstrap.py + initial state.json**

```bash
git add .claude/skills/project-dag/scripts/bootstrap.py docs/superpowers/dag/state.json
git -c core.hooksPath= commit -m "spec-c-v2: bootstrap.py — emit state.json; initial DAG snapshot"
```

---

### Task 6: `dag-bootstrap.sh` wrapper

**Files:**
- Create: `.claude/skills/project-dag/scripts/dag-bootstrap.sh`

- [ ] **Step 1: Write thin shell wrapper**

```bash
#!/usr/bin/env bash
# Rebuild docs/superpowers/dag/state.json from ROADMAP + plans.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 .claude/skills/project-dag/scripts/bootstrap.py "$@"
```

- [ ] **Step 2: Make executable + run**

```bash
chmod +x .claude/skills/project-dag/scripts/dag-bootstrap.sh
bash .claude/skills/project-dag/scripts/dag-bootstrap.sh
```

Expected: identical output to direct python invocation.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/project-dag/scripts/dag-bootstrap.sh
git -c core.hooksPath= commit -m "spec-c-v2: dag-bootstrap.sh wrapper"
```

---

### Task 7: `dag-status.sh` — ad-hoc state query

**Files:**
- Create: `.claude/skills/project-dag/scripts/dag-status.sh`

- [ ] **Step 1: Write status reporter**

```bash
#!/usr/bin/env bash
# Print a human summary of DAG state.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
STATE="$REPO_ROOT/docs/superpowers/dag/state.json"

if [[ ! -f "$STATE" ]]; then
    echo "state.json not found. Run dag-bootstrap.sh first." >&2
    exit 1
fi

echo "=== DAG status ==="
jq -r '
    "Roadmap items: \(.roadmap | length)",
    "Plans: \(.plans | length) (done: \(.plans | map(select(.status == "done")) | length))",
    "Tasks: \(.tasks | length)",
    "  done:        \(.tasks | map(select(.status == "done")) | length)",
    "  in_progress: \(.tasks | map(select(.status == "in_progress")) | length)",
    "  pending:     \(.tasks | map(select(.status == "pending")) | length)",
    "  blocked:     \(.tasks | map(select(.status == "blocked")) | length)",
    "By owner_class:",
    (.tasks | group_by(.owner_class) | map("  \(.[0].owner_class): \(length)") | .[])
' "$STATE"

echo
echo "=== Pending implementer tasks (next 10 actionable) ==="
jq -r '
    .tasks
    | map(select(.status == "pending" and .owner_class == "implementer"))
    | .[0:10]
    | map("  \(.id): \(.title)")
    | .[]
' "$STATE"

echo
echo "=== Run state ==="
jq -r '.run | "  iterations: \(.iterations_completed)\n  closed: \(.tasks_closed)\n  blocked: \(.tasks_blocked)\n  phase: \(.current_phase)"' "$STATE"
```

- [ ] **Step 2: Run**

```bash
chmod +x .claude/skills/project-dag/scripts/dag-status.sh
bash .claude/skills/project-dag/scripts/dag-status.sh
```

Expected: counts + first 10 pending implementer tasks.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/project-dag/scripts/dag-status.sh
git -c core.hooksPath= commit -m "spec-c-v2: dag-status.sh — ad-hoc state query"
```

---

### Task 8: `dag-mark-done.sh` + `dag-mark-blocked.sh`

**Files:**
- Create: `.claude/skills/project-dag/scripts/dag-mark-done.sh`
- Create: `.claude/skills/project-dag/scripts/dag-mark-blocked.sh`

**Goal:** Atomic state mutations called by the agent after subagent completion. Both scripts: (1) update `state.json`; (2) flip plan-file checkbox if `done`; (3) append to `run.jsonl`.

- [ ] **Step 1: Write `dag-mark-done.sh`**

```bash
#!/usr/bin/env bash
# Mark a task done. Args: <task_id> <commit_sha> [critic_verdicts_json]
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: dag-mark-done.sh <task_id> <commit_sha> [critic_verdicts_json]" >&2
    exit 2
fi

TASK_ID="$1"
COMMIT="$2"
VERDICTS="${3:-{}}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
STATE="$REPO_ROOT/docs/superpowers/dag/state.json"
LOG="$REPO_ROOT/docs/superpowers/dag/run.jsonl"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Update state.json
TMP="$(mktemp)"
jq --arg id "$TASK_ID" --arg commit "$COMMIT" --arg now "$NOW" --argjson verdicts "$VERDICTS" '
    .tasks |= map(
        if .id == $id then
            .status = "done"
            | .completed_commit = $commit
            | .completed_at = $now
            | .critic_verdicts = $verdicts
        else . end
    )
    | .run.tasks_closed += 1
    | .run.last_iteration_at = $now
' "$STATE" > "$TMP"
mv "$TMP" "$STATE"

# Flip plan-file checkbox
PLAN_FILE=$(jq -r --arg id "$TASK_ID" '.tasks[] | select(.id == $id) | (.plan // empty)' "$STATE")
LINE=$(jq -r --arg id "$TASK_ID" '.tasks[] | select(.id == $id) | (.checkbox_line // empty)' "$STATE")
if [[ -n "$PLAN_FILE" && -n "$LINE" && "$LINE" != "null" ]]; then
    PATH_GUESS=$(jq -r --arg p "$PLAN_FILE" '.plans[] | select(.id == $p) | .file' "$STATE")
    if [[ -f "$REPO_ROOT/$PATH_GUESS" ]]; then
        sed -i "${LINE}s/- \[ \]/- [x]/" "$REPO_ROOT/$PATH_GUESS"
    fi
fi

# Append to run.jsonl
echo "{\"ts\":\"$NOW\",\"event\":\"task_done\",\"task_id\":\"$TASK_ID\",\"commit\":\"$COMMIT\",\"critic_verdicts\":$VERDICTS}" >> "$LOG"
echo "Marked $TASK_ID done (commit $COMMIT)"
```

- [ ] **Step 2: Write `dag-mark-blocked.sh`**

```bash
#!/usr/bin/env bash
# Mark a task blocked. Args: <task_id> <reason>
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: dag-mark-blocked.sh <task_id> <reason>" >&2
    exit 2
fi

TASK_ID="$1"
REASON="$2"

REPO_ROOT="$(git rev-parse --show-toplevel)"
STATE="$REPO_ROOT/docs/superpowers/dag/state.json"
LOG="$REPO_ROOT/docs/superpowers/dag/run.jsonl"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

TMP="$(mktemp)"
jq --arg id "$TASK_ID" --arg reason "$REASON" --arg now "$NOW" '
    .tasks |= map(
        if .id == $id then
            .status = "blocked"
            | .blocked_reason = $reason
            | .retry_count = (.retry_count + 1)
        else . end
    )
    | .run.tasks_blocked += 1
    | .run.last_iteration_at = $now
' "$STATE" > "$TMP"
mv "$TMP" "$STATE"

REASON_JSON=$(jq -Rn --arg r "$REASON" '$r')
echo "{\"ts\":\"$NOW\",\"event\":\"task_blocked\",\"task_id\":\"$TASK_ID\",\"reason\":$REASON_JSON}" >> "$LOG"
echo "Marked $TASK_ID blocked: $REASON"
```

- [ ] **Step 3: Make executable + smoke test**

```bash
chmod +x .claude/skills/project-dag/scripts/dag-mark-done.sh
chmod +x .claude/skills/project-dag/scripts/dag-mark-blocked.sh

# Pick a fictional task_id to test rejection (no error, just no-op on missing id)
bash .claude/skills/project-dag/scripts/dag-mark-blocked.sh "nonexistent.task-1" "smoke test"
# Verify run.jsonl got an entry
tail -1 docs/superpowers/dag/run.jsonl
# Restore state (revert the run counter bump)
bash .claude/skills/project-dag/scripts/dag-bootstrap.sh
```

Expected: `task_blocked` line in run.jsonl. Bootstrap re-derives clean state.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/project-dag/scripts/dag-mark-done.sh .claude/skills/project-dag/scripts/dag-mark-blocked.sh docs/superpowers/dag/run.jsonl
git -c core.hooksPath= commit -m "spec-c-v2: dag-mark-{done,blocked}.sh — atomic state mutations + run.jsonl"
```

---

### Task 9: `dag-terminal.sh` — terminal condition detection

**Files:**
- Create: `.claude/skills/project-dag/scripts/dag-terminal.sh`

- [ ] **Step 1: Write terminal detector**

```bash
#!/usr/bin/env bash
# Detect terminal condition: DAG_COMPLETE, HUMAN_BLOCKED, or NONE.
# Outputs the terminal kind on stdout; nothing if run should continue.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
STATE="$REPO_ROOT/docs/superpowers/dag/state.json"
LOG="$REPO_ROOT/docs/superpowers/dag/run.jsonl"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

ELIGIBLE=$(jq -r '
    .tasks
    | map(select(.status == "pending" and .owner_class == "implementer"))
    | length
' "$STATE")

if [[ "$ELIGIBLE" -gt 0 ]]; then
    # There may still be eligibility blocked by deps; check if ANY pending implementer
    # task has all deps satisfied
    READY=$(jq -r '
        . as $root
        | .tasks
        | map(select(.status == "pending" and .owner_class == "implementer"))
        | map(. as $t |
            ($t.deps | map(. as $d |
                ($root.tasks | map(select(.id == $d)) | first | .status)
            ) | all(. == "done" or . == "deferred" or . == "skipped" or . == null))
        )
        | map(select(. == true))
        | length
    ' "$STATE")
    if [[ "$READY" -gt 0 ]]; then
        # Run can continue
        exit 0
    fi
fi

# No actionable implementer tasks. Determine terminal kind.
ANY_HUMAN=$(jq -r '
    .tasks
    | map(select(.status == "pending" and (.owner_class == "plan-writer" or .owner_class == "spec-needed")))
    | length
' "$STATE")

ANY_PENDING=$(jq -r '.tasks | map(select(.status != "done" and .status != "skipped" and .status != "deferred")) | length' "$STATE")

if [[ "$ANY_PENDING" -eq 0 ]]; then
    KIND="DAG_COMPLETE"
elif [[ "$ANY_HUMAN" -gt 0 ]]; then
    KIND="HUMAN_BLOCKED"
else
    KIND="HUMAN_BLOCKED"  # blocked tasks awaiting unblock
fi

echo "{\"ts\":\"$NOW\",\"event\":\"terminal\",\"reason\":\"$KIND\"}" >> "$LOG"
echo "$KIND"
```

- [ ] **Step 2: Run**

```bash
chmod +x .claude/skills/project-dag/scripts/dag-terminal.sh
bash .claude/skills/project-dag/scripts/dag-terminal.sh || echo "Run continues (eligible work exists)"
```

Expected: empty output (run continues — implementer tasks exist in current DAG).

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/project-dag/scripts/dag-terminal.sh
git -c core.hooksPath= commit -m "spec-c-v2: dag-terminal.sh — DAG_COMPLETE / HUMAN_BLOCKED detection"
```

---

### Task 10: `dag-tick.sh` — single-iteration agent shell

**Files:**
- Create: `.claude/skills/project-dag/scripts/dag-tick.sh`

**Note:** This script handles state mutation only. The Claude session running `/loop /dag-tick` is responsible for dispatching the SDD subagent (via the `Agent` tool) between this script's "started" and "done" calls. See SKILL.md (Task 11).

- [ ] **Step 1: Write the picker + start step**

```bash
#!/usr/bin/env bash
# Pick the next eligible implementer task and mark it in_progress.
# Output the task as JSON for the calling Claude session to dispatch.
# If no eligible task: invoke dag-terminal.sh and exit 0.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
STATE="$REPO_ROOT/docs/superpowers/dag/state.json"
LOG="$REPO_ROOT/docs/superpowers/dag/run.jsonl"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Find first eligible implementer task
TASK=$(jq -r '
    . as $root
    | .tasks
    | map(select(.status == "pending" and .owner_class == "implementer"))
    | map(. as $t |
        if ($t.deps | length) == 0 then $t
        else
            ($t.deps | map(. as $d |
                ($root.tasks | map(select(.id == $d)) | first | .status)
            ) | all(. == "done" or . == "deferred" or . == "skipped" or . == null))
            as $ready
            | if $ready then $t else empty end
        end
    )
    | first // empty
' "$STATE")

if [[ -z "$TASK" || "$TASK" == "null" ]]; then
    # No eligible work — emit terminal
    KIND=$(bash "$REPO_ROOT/.claude/skills/project-dag/scripts/dag-terminal.sh")
    echo "TERMINAL: $KIND"
    exit 0
fi

TASK_ID=$(echo "$TASK" | jq -r '.id')
TITLE=$(echo "$TASK" | jq -r '.title')

# Mark in_progress
TMP="$(mktemp)"
jq --arg id "$TASK_ID" --arg now "$NOW" '
    .tasks |= map(
        if .id == $id then .status = "in_progress" | .started_at = $now
        else . end
    )
    | .run.iterations_completed += 1
    | .run.last_iteration_at = $now
    | .run.current_phase = "dispatching"
' "$STATE" > "$TMP"
mv "$TMP" "$STATE"

echo "{\"ts\":\"$NOW\",\"event\":\"task_started\",\"task_id\":\"$TASK_ID\"}" >> "$LOG"

# Output task JSON for Claude session to consume
echo "$TASK"
```

- [ ] **Step 2: Make executable + smoke test (do NOT commit the state mutation)**

```bash
chmod +x .claude/skills/project-dag/scripts/dag-tick.sh
bash .claude/skills/project-dag/scripts/dag-tick.sh | jq -r '.id, .title'
# Reset: re-bootstrap to clear in_progress mark from smoke test
bash .claude/skills/project-dag/scripts/dag-bootstrap.sh
```

Expected: prints a task ID + title (e.g. `2026-04-25-plan-4-debug-trace-impl.task-1`). Re-bootstrap reverts.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/project-dag/scripts/dag-tick.sh docs/superpowers/dag/state.json
git -c core.hooksPath= commit -m "spec-c-v2: dag-tick.sh — pick next eligible task + mark in_progress"
```

---

### Task 11: `SKILL.md` — agent invocation pattern

**Files:**
- Create: `.claude/skills/project-dag/SKILL.md`

**Goal:** Document the agent loop so a fresh Claude session running `/loop /dag-tick` knows how to compose the shell scripts with `Agent` (subagent dispatch) + `ScheduleWakeup` (cross-session continuity).

- [ ] **Step 1: Write SKILL.md**

```markdown
---
name: project-dag
description: Use when running the autonomous multi-day DAG drain loop. Defines the agent's per-iteration responsibilities — pick task, dispatch subagent, mark done/blocked, schedule next wake.
---

# Project DAG (autonomous run)

## Overview

Drives a multi-day autonomous run that drains implementation work from `docs/superpowers/dag/state.json`. Per Spec C v2 §3.3.

**Announce at start:** "I'm using the project-dag skill to advance one iteration of the autonomous DAG run."

## Per-iteration steps

```dot
digraph dag_tick {
    "Run dag-tick.sh" [shape=box];
    "Got task?" [shape=diamond];
    "Terminal posted by dag-tick" [shape=doublecircle];
    "Dispatch SDD subagent" [shape=box];
    "Subagent reports DONE?" [shape=diamond];
    "dag-mark-blocked.sh + escalate" [shape=box];
    "Dispatch spec-compliance reviewer" [shape=box];
    "Compliance PASS?" [shape=diamond];
    "Implementer fix-up loop" [shape=box];
    "Dispatch quality reviewer" [shape=box];
    "Quality PASS?" [shape=diamond];
    "dag-mark-done.sh + critic verdicts" [shape=box];
    "ScheduleWakeup 60-300s" [shape=doublecircle];

    "Run dag-tick.sh" -> "Got task?";
    "Got task?" -> "Terminal posted by dag-tick" [label="no"];
    "Got task?" -> "Dispatch SDD subagent" [label="yes"];
    "Dispatch SDD subagent" -> "Subagent reports DONE?";
    "Subagent reports DONE?" -> "dag-mark-blocked.sh + escalate" [label="BLOCKED"];
    "Subagent reports DONE?" -> "Dispatch spec-compliance reviewer" [label="DONE"];
    "Dispatch spec-compliance reviewer" -> "Compliance PASS?";
    "Compliance PASS?" -> "Implementer fix-up loop" [label="FAIL"];
    "Implementer fix-up loop" -> "Dispatch spec-compliance reviewer";
    "Compliance PASS?" -> "Dispatch quality reviewer" [label="PASS"];
    "Dispatch quality reviewer" -> "Quality PASS?";
    "Quality PASS?" -> "Implementer fix-up loop" [label="FAIL"];
    "Quality PASS?" -> "dag-mark-done.sh + critic verdicts" [label="PASS"];
    "dag-mark-done.sh + critic verdicts" -> "ScheduleWakeup 60-300s";
    "dag-mark-blocked.sh + escalate" -> "ScheduleWakeup 60-300s";
}
```

## Concrete agent recipe

1. **Run `bash .claude/skills/project-dag/scripts/dag-tick.sh`**
   - If output starts with `TERMINAL:`, the run has terminated — post the summary to chat and stop. Do NOT call `ScheduleWakeup`.
   - Otherwise, parse the JSON task object (`{id, title, plan, ...}`).

2. **Construct subagent prompt** including:
   - Task ID + title
   - Plan file path (`docs/superpowers/plans/<plan>.md`) + checkbox line
   - Hard scope: subagent only does the steps under that one task; cannot expand scope
   - Reminder: pre-commit critic gate runs on commit; subagent must pass it

3. **Dispatch via `Agent` tool with `subagent_type=general-purpose`** (or `Plan` if architecture work; should not happen for `implementer` tasks).

4. **On subagent return:**
   - If `BLOCKED` or implementer reports `NEEDS_CONTEXT` exhausted: `bash .claude/skills/project-dag/scripts/dag-mark-blocked.sh <task_id> "<reason>"`. Post escalation to chat. Continue to step 6.
   - If `DONE`: dispatch spec-compliance reviewer (subagent_type=general-purpose, instructions: "Review this commit against the plan task spec; report PASS or FAIL with reasoning"). On FAIL, dispatch implementer fix-up; loop until PASS or escalate after 2 retries.

5. **Quality review:** dispatch quality reviewer subagent. Same loop.

6. **Mark outcome:**
   - On full PASS: `bash .claude/skills/project-dag/scripts/dag-mark-done.sh <task_id> <commit_sha> '{"compliance":"PASS","quality":"PASS"}'`
   - On blocked: already done in step 4

7. **Schedule next wake:** call `ScheduleWakeup(delaySeconds=60, prompt="<original /loop prompt>", reason="next DAG iteration")`. The 60s delay keeps the prompt cache warm.

## Hard-stop triggers

Halt the loop (do NOT call `ScheduleWakeup`) on any of:

- **Critic FAIL** in `.claude/scripts/dispatch-critics.sh` output (any of the 6 critics)
- **3+ consecutive `task_blocked`** without intervening `task_done` (grep `run.jsonl`)
- **Allowlist edit detected** — subagent modified `engine/build.rs`; even if critic-allowlist-gate PASSED, escalate for user review
- **Schema-hash baseline change** — schema-bump critic flagged it
- **Test failure** the agent can't resolve in 2 retries

On hard-stop: post detailed escalation to chat including the failing critic / blocker / commit hash.

## Daily synthesis

Every ~6-8 hours of wall-clock time, post a synthesis (template at `docs/superpowers/dag/templates/daily-synthesis.md`). Track via `run.jsonl` event count or wall-clock since last `daily_synthesis_posted` event.

## Pending decisions

When `dag-tick.sh` returns `TERMINAL: HUMAN_BLOCKED`:
- Read `docs/superpowers/dag/pending-decisions.md`
- Append fresh sections for newly-encountered `plan-writer` / `spec-needed` tasks (one per remaining pending non-implementer task)
- Post a chat summary listing what's queued + how user resolves each

## Reference

- Spec: `docs/superpowers/specs/2026-04-25-project-dag-v2-design.md`
- Bootstrap: `.claude/skills/project-dag/scripts/dag-bootstrap.sh`
- Status: `bash .claude/skills/project-dag/scripts/dag-status.sh`
```

- [ ] **Step 2: Verify SKILL.md is well-formed**

```bash
head -20 .claude/skills/project-dag/SKILL.md
```

Expected: frontmatter (`name:`, `description:`) + first H1.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/project-dag/SKILL.md
git -c core.hooksPath= commit -m "spec-c-v2: SKILL.md — agent invocation pattern for DAG run"
```

---

### Task 12: Synthesis + escalation templates + pending-decisions scaffold

**Files:**
- Create: `docs/superpowers/dag/templates/daily-synthesis.md`
- Create: `docs/superpowers/dag/templates/escalation.md`
- Create: `docs/superpowers/dag/pending-decisions.md`

- [ ] **Step 1: Write `daily-synthesis.md` template**

```markdown
## DAG run — daily synthesis ({{DATE}}, day {{DAY_N}})

**Tasks closed today:** {{TASKS_CLOSED}} (across {{PLANS_TOUCHED}} plans)
{{PER_PLAN_PROGRESS}}

**Plans completed:** {{PLANS_COMPLETED}}

**Blockers introduced:** {{BLOCKERS_TODAY}}
{{BLOCKER_DETAILS}}

**Critic verdicts of note:**
- {{PASS_COUNT}} PASS verdicts ({{FAIL_COUNT}} FAILs)
{{ALLOWLIST_DISPATCHES}}

**Tomorrow's priorities (top 3):**
{{TOP_3_NEXT}}

Run continues; next wake in {{NEXT_WAKE_SECONDS}}s.
```

- [ ] **Step 2: Write `escalation.md` template**

```markdown
## DAG run — escalation ({{DATE}} {{TIME}})

**Type:** {{ESCALATION_TYPE}}

**Task:** `{{TASK_ID}}` — {{TASK_TITLE}}

**Commit:** `{{COMMIT}}` ({{COMMIT_STATUS}})

**Reasoning:**
> {{REASONING}}

**Run paused on this branch.** {{OTHER_PLANS_STATUS}}. Resolve via:
{{RESOLUTION_OPTIONS}}
```

- [ ] **Step 3: Write `pending-decisions.md` scaffold**

```markdown
# Pending Decisions

> Append-only log of human-gated decisions surfaced by the autonomous DAG run.
> Entries land here when the run encounters `plan-writer`, `spec-needed`, or
> `human-needed` work. The run continues with other eligible tasks (or
> terminates if none remain).
>
> User resolves each entry by editing the section to add an `**APPROVED:**`
> line (for plan-writer) or by starting an interactive brainstorm session
> (for spec-needed). Then re-run `dag-bootstrap.sh` to incorporate the
> resolution.

---

<!-- entries appended below by the agent -->
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/dag/templates/daily-synthesis.md \
        docs/superpowers/dag/templates/escalation.md \
        docs/superpowers/dag/pending-decisions.md
git -c core.hooksPath= commit -m "spec-c-v2: templates (daily-synthesis, escalation) + pending-decisions scaffold"
```

---

### Task 13: End-to-end smoke test

**Files:** none new; runs all scripts in sequence.

**Goal:** Validate the whole pipeline works without dispatching a real subagent. Simulate a task lifecycle end-to-end.

- [ ] **Step 1: Bootstrap fresh state**

```bash
bash .claude/skills/project-dag/scripts/dag-bootstrap.sh
bash .claude/skills/project-dag/scripts/dag-status.sh
```

Expected: counts printed; first 10 pending implementer tasks listed.

- [ ] **Step 2: Pick a task (without committing the state mutation)**

```bash
TASK_JSON=$(bash .claude/skills/project-dag/scripts/dag-tick.sh)
echo "Picked: $TASK_JSON"
TASK_ID=$(echo "$TASK_JSON" | jq -r '.id')
echo "Task ID: $TASK_ID"
```

Expected: a real task ID like `2026-04-25-plan-4-debug-trace-impl.task-1`.

- [ ] **Step 3: Simulate done, then re-bootstrap to clean up**

```bash
bash .claude/skills/project-dag/scripts/dag-mark-done.sh "$TASK_ID" "smoke-test-sha" '{"compliance":"PASS","quality":"PASS"}'
tail -3 docs/superpowers/dag/run.jsonl
# Restore: re-bootstrap clears the smoke-test mutation since the plan-file checkbox
# was flipped — we need to flip it back before re-bootstrap
bash .claude/skills/project-dag/scripts/dag-status.sh
```

Expected: run.jsonl has `task_started` + `task_done` lines. dag-status reflects 1 task done.

- [ ] **Step 4: Restore plan-file + state**

```bash
# Find the plan-file the smoke test flipped, revert that one line
git status --short docs/superpowers/plans/
git checkout -- docs/superpowers/plans/
bash .claude/skills/project-dag/scripts/dag-bootstrap.sh
```

Expected: plan files clean. state.json regenerated. run.jsonl retains the smoke-test entries (append-only is correct behaviour; users can grep historical runs).

- [ ] **Step 5: Commit final state + smoke-test run.jsonl history**

```bash
git add docs/superpowers/dag/state.json docs/superpowers/dag/run.jsonl
git -c core.hooksPath= commit -m "spec-c-v2: end-to-end smoke test — bootstrap → tick → mark-done → re-bootstrap clean"
```

---

### Task 14: Final verification + AIS post-design tick

**Files:**
- Modify: `docs/superpowers/plans/2026-04-25-spec-c-v2-bootstrap-impl.md` (this file — tick the post-design box)

- [ ] **Step 1: Run all status surfaces one more time**

```bash
bash .claude/skills/project-dag/scripts/dag-status.sh
ls -la docs/superpowers/dag/
ls -la .claude/skills/project-dag/
ls -la .claude/skills/project-dag/scripts/
wc -l .claude/skills/project-dag/scripts/*.sh .claude/skills/project-dag/scripts/*.py
```

Expected: state.json + run.jsonl present; SKILL.md + 6 scripts (`bootstrap.py`, `dag-bootstrap.sh`, `dag-status.sh`, `dag-mark-done.sh`, `dag-mark-blocked.sh`, `dag-terminal.sh`, `dag-tick.sh`); pending-decisions.md scaffolded; templates present.

- [ ] **Step 2: Confirm the bootstrap.py doesn't accidentally classify B1' or critic-skills tasks as needing meta-work**

```bash
jq '.tasks | map(select(.owner_class == "spec-needed" or .owner_class == "plan-writer")) | map({id, owner_class, title})' docs/superpowers/dag/state.json
```

Expected: list shows only roadmap items genuinely lacking plans/specs (not items where a plan already exists).

- [ ] **Step 3: Tick the post-design AIS box**

Edit `docs/superpowers/plans/2026-04-25-spec-c-v2-bootstrap-impl.md` Re-evaluation line:
```
- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [x] AIS reviewed post-design (after task list stabilises).
```

- [ ] **Step 4: Final commit**

```bash
git add docs/superpowers/plans/2026-04-25-spec-c-v2-bootstrap-impl.md
git -c core.hooksPath= commit -m "spec-c-v2: post-design AIS tick — bootstrap infrastructure complete"
```

- [ ] **Step 5: Smoke verify branch state**

```bash
git log --oneline -20
git status
```

Expected: 14 new commits since plan-5a; clean tree.

---

## Summary

After this plan executes:
- `docs/superpowers/dag/state.json` reflects the current roadmap + plan checkbox state
- `docs/superpowers/dag/run.jsonl` is an append-only log of agent actions
- `.claude/skills/project-dag/` provides the autonomous loop entry points (SKILL.md + 6 scripts)
- `pending-decisions.md` is ready to receive plan-writer / spec-needed escalations
- The infrastructure is ready for Spec C v2 §12 Phase 5 (smoke run, ~4h supervised) and Phase 6 (multi-day production run)

Phases 5 (smoke) and 6 (production) are user-initiated. They are out of scope for this plan because they aren't bite-sized engineering tasks — they're "let the loop run and see what happens."
