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
