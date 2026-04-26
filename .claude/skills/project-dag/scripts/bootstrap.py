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
    task_header_re = re.compile(r"^### Task (\d+):\s*(.+?)\s*$")
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
                "id": f"{plan_id}.task-{int(h.group(1))}",
                "plan": plan_id,
                "title": f"Task {h.group(1)}: {h.group(2).strip()}",
                "checkbox_line": lineno,
                "deps": [],  # filled after all tasks gathered
                "blocks": [],
                "status": "pending",
                "steps_total": 0,
                "steps_done": 0,
                "owner_class": "implementer",  # default; overridden in Task 4
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


if __name__ == "__main__":
    plans = parse_all_plans(PLANS_DIR)
    for p in plans:
        print(f"{p['id']}: {p['tasks_done']}/{p['tasks_total']} tasks  ({p['status']})")
