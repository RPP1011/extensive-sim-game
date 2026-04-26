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

    # Tasks are h3 headers `### Task N: <title>`; their bodies contain
    # checkbox steps `- [ ] **Step N: ...**`. A task is `done` iff every
    # step under it is checked. Steps that appear before the first Task
    # header (e.g. preamble, AIS) are ignored.
    plan_id = path.stem
    tasks: list[dict] = []
    current_task: dict | None = None
    current_steps: list[bool] = []  # collected for current_task
    task_header_re = re.compile(r"^### Task ([A-Za-z]?\d+):\s*(.+?)\s*$")
    step_re = re.compile(r"^- \[([ x])\] \*\*(?:Step|Task) \d+:\s*(.+?)\*\*\s*$")

    def finalize(task: dict | None, steps: list[bool]) -> None:
        if task is None:
            return
        task["steps_total"] = len(steps)
        task["steps_done"] = sum(1 for s in steps if s)
        if steps and all(steps):
            task["status"] = "done"
        elif any(steps):
            task["status"] = "in_progress"
        else:
            task["status"] = "pending"
        tasks.append(task)

    for lineno, line in enumerate(lines, start=1):
        h = task_header_re.match(line)
        if h:
            finalize(current_task, current_steps)
            current_steps = []
            current_task = {
                "id": f"{plan_id}.task-{h.group(1)}",
                "plan": plan_id,
                "title": f"Task {h.group(1)}: {h.group(2).strip()}",
                "checkbox_line": lineno,
                "deps": [],  # filled after all tasks gathered
                "blocks": [],
                "status": "pending",
                "steps_total": 0,
                "steps_done": 0,
                "owner_class": classify_owner(f"Task {h.group(1)}: {h.group(2).strip()}"),
                "blocked_reason": None,
                "completed_commit": None,
                "critic_verdicts": {},
                "started_at": None,
                "completed_at": None,
                "retry_count": 0,
            }
            continue
        if current_task is None:
            continue
        s = step_re.match(line)
        if s:
            current_steps.append(s.group(1) == "x")
    finalize(current_task, current_steps)

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
            "steps_total": 0,
            "steps_done": 0,
            "owner_class": owner,
            "blocked_reason": None,
            "completed_commit": None,
            "critic_verdicts": {},
            "started_at": None,
            "completed_at": None,
            "retry_count": 0,
        })
    return meta


def merge_prior_state(tasks: list[dict], prior_path: Path) -> list[dict]:
    """Reconcile current task list with prior state.json, preserving runtime fields."""
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
    """Write state.json with derived rollup status for plans and roadmap."""
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
