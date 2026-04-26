# Project DAG (v2) — Design Spec

> **Status:** Design (2026-04-25). **Supersedes** `2026-04-24-project-dag-skill-design.md`.
>
> The 2026-04-24 design's invariant-enforcement layer was overtaken by Spec
> B' (build sentinels + ast-grep CI) and Spec D + D-amendment (6 critic
> skills + dispatch-critics orchestrator). This v2 reframes Spec C as
> agent-infrastructure for a multi-day autonomous run that drains the
> roadmap, not as user-tooling for invariant tracking.

## §0 What changed from v1

1. **Drop the invariant catalog (10 invariants).** All structural enforcement is now done by Spec B' build sentinels (`engine/build.rs` allowlist, engine_rules + engine_data `// GENERATED` headers), Spec B' ast-grep CI rules (`impl CascadeHandler` location), Spec D 6 critic skills (compiler-first, schema-bump, cross-backend-parity, no-runtime-panic, reduction-determinism, allowlist-gate), and Spec D-amendment hooks + pre-commit gate. v2 references these layers; doesn't recreate them.

2. **Drop TaskWarrior.** Plan files (existing) are the source of truth for task state via their `- [ ]` / `- [x]` checkboxes. v2 adds a sidecar JSON file for cross-plan dependencies + run metadata. No external system to install.

3. **Add the agent runtime.** v1 was a tracking skill the user invoked. v2 is an autonomous-loop infrastructure: agent wakes, picks next actionable task, dispatches a subagent, marks done, schedules next wake. Halts cleanly on terminal conditions.

4. **Tiered autonomy gates (D).** Implementation work (plan exists with concrete tasks + acceptance criteria) runs unattended. Plan-writing, spec-writing, and architectural decisions ALWAYS escalate to user. Run terminates when only human-gated work remains.

## §1 Goals

1. **Encode the entire roadmap as a single executable DAG.** Every roadmap item maps to a node; every plan task is a leaf; every cross-plan dep is an edge. The DAG IS the canonical "what's left to ship" state.

2. **Drive a multi-day autonomous run** through the implementation work. Agent picks any reachable eligible task, executes it via subagent-driven-development, marks done with verified commit + critic verdicts, repeats. Stops cleanly when only human-gated work remains.

3. **Surface human-gated decisions in batches.** Plan-writing, spec-writing, architectural decisions all escalate. Agent doesn't try to be a designer.

4. **Provide observability without polling.** Run emits a structured log per action; posts a daily synthesis; escalates immediately on hard-stop conditions. User can be away for stretches and return to a clear picture.

5. **Be lightweight enough to bootstrap in one afternoon.** No external dependencies; existing artifacts (ROADMAP.md, plan files) drive most of the state.

## §2 Non-goals

- **Replacing ROADMAP.md or plan files.** They stay authoritative for human-readable structure; v2 just adds machine-readable metadata alongside.
- **Real-time progress dashboards.** The structured log is greppable; that's enough.
- **Multi-agent orchestration.** Single agent loop; no parallel agent instances against the same DAG (locking would be a separate spec).
- **Estimating effort or due dates.** Tasks are done or not done. No story points.
- **Replacing the dispatch-critics gate.** Critics still run on every commit; v2 just records their verdicts in the DAG state.
- **Becoming a general project-management tool.** Project-specific; lives in this repo.

## §3 Architecture

### §3.1 Six locked decisions

From brainstorming (2026-04-25):

| # | Decision | Implication |
|---|---|---|
| 1 | **(D) Tiered autonomy** | Implementation work is autonomous; plan-writing, spec-writing, architectural decisions always escalate to user |
| 2 | **(α) Per-task iteration** | Agent loop dispatches one plan-task at a time via subagent-driven-development |
| 3 | **Plan files + sidecar JSON storage** | Plan files own checkbox state; `docs/superpowers/dag/state.json` owns cross-plan metadata |
| 4 | **Hybrid observability** | Always-on `run.jsonl` log + daily synthesis chat post + interrupt escalations + hard-stop conditions |
| 5 | **Single DAG, any reachable eligible node** | No priority weighting; first eligible task wins; deterministic ordering by plan-id + task-index |
| 6 | **(D) Terminal halt** | Run stops cleanly when no implementation tasks remain (only human-gated work left); user handles batches periodically |

### §3.2 Component layout

```
docs/
  ROADMAP.md                                  human-curated index (existing)
  superpowers/
    plans/*.md                                per-plan checkbox tracking (existing)
    dag/
      state.json                              NEW: cross-plan metadata + run state
      run.jsonl                               NEW: structured action log (append-only)
      bootstrap.py                            NEW: parse ROADMAP + plans → state.json
      pending-decisions.md                    NEW: human-gated escalations (terminal output)

.claude/
  skills/
    project-dag/
      SKILL.md                                NEW: agent invocation + commands
      scripts/
        dag-tick.sh                           NEW: one-iteration agent loop
        dag-status.sh                         NEW: report current state
        dag-bootstrap.sh                      NEW: rebuild state.json from artifacts
```

`state.json` is the only DAG mutation surface. Plan files are read-only from the DAG's perspective EXCEPT for the agent flipping `- [ ]` → `- [x]` on task completion.

### §3.3 Run lifecycle

```
1. User runs `/dag-bootstrap`                  rebuilds state.json from ROADMAP + plans
2. User runs `/loop /dag-tick`                  starts the autonomous run (Claude Code /loop)
3. Per iteration:
   a. Read state.json
   b. Find any eligible task (next_actionable per §3.1#5)
   c. If found:
        - Dispatch subagent (subagent-driven-development per task)
        - Subagent reports DONE / BLOCKED / NEEDS_CONTEXT
        - Update state.json + plan-file checkbox
        - Append to run.jsonl
        - ScheduleWakeup for next iteration
   d. If not found:
        - Check terminal conditions
        - Post terminal report (DAG_COMPLETE | HUMAN_BLOCKED | HARD_STOP)
        - Exit /loop
4. User reviews terminal report; resolves human-gated work; re-runs /dag-bootstrap + /loop
```

## §4 State model

### §4.1 `state.json` schema

```json
{
  "schema_version": 1,
  "generated_at": "2026-04-25T15:30:00Z",
  "roadmap": [
    {
      "id": "roadmap-plan-4-debug-trace",
      "title": "Plan 4 — debug & trace runtime",
      "tier": "engine-plans-not-yet-written",
      "spec_ref": "docs/spec/runtime.md#§23",
      "plans": ["plan-4-debug-trace"],
      "status": "pending"
    }
  ],
  "plans": [
    {
      "id": "plan-4-debug-trace",
      "file": "docs/superpowers/plans/2026-XX-plan-4-debug-trace.md",
      "roadmap_parent": "roadmap-plan-4-debug-trace",
      "status": "pending",
      "tasks_total": 12,
      "tasks_done": 0
    }
  ],
  "tasks": [
    {
      "id": "plan-4-debug-trace.task-1",
      "plan": "plan-4-debug-trace",
      "title": "Add trace_mask primitive",
      "checkbox_line": 142,
      "deps": [],
      "blocks": ["plan-4-debug-trace.task-2", "plan-4-debug-trace.task-3"],
      "status": "pending",
      "owner_class": "implementer",
      "blocked_reason": null,
      "completed_commit": null,
      "critic_verdicts": {},
      "started_at": null,
      "completed_at": null,
      "retry_count": 0
    }
  ],
  "run": {
    "started_at": null,
    "last_iteration_at": null,
    "iterations_completed": 0,
    "tasks_closed": 0,
    "tasks_blocked": 0,
    "current_phase": "idle",
    "next_wake_scheduled": null
  }
}
```

### §4.2 Status enums

`task.status`: `pending | in_progress | done | blocked | deferred | skipped`
- `pending` — eligible if deps satisfied
- `in_progress` — subagent currently working it (set on dispatch, unset on completion)
- `done` — completed; commit + critic verdicts recorded
- `blocked` — agent failed 2+ retries OR explicit user defer; needs human input
- `deferred` — user explicitly punted; doesn't block dependents (treated as `done` for dep satisfaction)
- `skipped` — superseded by another task; doesn't block dependents

`task.owner_class`: `implementer | plan-writer | spec-needed | human-needed`
- `implementer` — agent runs SDD subagent autonomously
- `plan-writer` — escalates to `pending-decisions.md`; user approves draft request
- `spec-needed` — escalates; spec brainstorm with user always required
- `human-needed` — explicitly human work (UX decisions, design calls); never picked by agent

`plan.status`: `pending | in_progress | done | blocked`
- Auto-derived: `done` iff all child tasks `done | deferred | skipped`; `in_progress` iff any task `in_progress | done`; `blocked` iff any task `blocked`; else `pending`.

`roadmap.status`: same shape, derived from child plans.

### §4.3 Plan-file integration

Plan files use the existing checkbox convention:

```markdown
- [ ] **Step 1: Write the failing test**
- [x] **Step 2: Run test to verify it fails**
```

Bootstrap parses these into task entries with `checkbox_line` referring to the source line. Agent flips the checkbox on completion via `Edit` tool, AND updates `state.json`. Both writes happen in the same commit.

The two sources can drift if a human hand-edits a checkbox without the agent's knowledge. Bootstrap is idempotent and reconciles on re-run: if `- [x]` in plan file but `pending` in state.json, prefer the plan file (mark as `done`, set `completed_commit` to "manual"). If `- [ ]` in plan file but `done` in state.json, prefer the state.json (silently flip to `- [x]`; agent is authoritative once the run is live).

### §4.4 `run.jsonl` schema

One line per agent action:

```json
{"ts": "2026-04-25T15:30:00Z", "event": "task_started", "task_id": "plan-4-debug-trace.task-1", "iteration": 1}
{"ts": "2026-04-25T15:34:12Z", "event": "subagent_dispatched", "task_id": "...", "subagent_id": "abc123"}
{"ts": "2026-04-25T15:42:30Z", "event": "task_done", "task_id": "...", "commit": "deadbeef", "critic_verdicts": {"compiler-first": "PASS", "schema-bump": "N/A"}}
{"ts": "2026-04-25T15:42:31Z", "event": "next_wake_scheduled", "delay_seconds": 60}
{"ts": "...", "event": "task_blocked", "task_id": "...", "reason": "subagent reported BLOCKED: missing context X", "retry_count": 2}
{"ts": "...", "event": "daily_synthesis_posted", "tasks_closed_today": 8, "blockers_introduced": 1}
{"ts": "...", "event": "terminal", "reason": "DAG_COMPLETE | HUMAN_BLOCKED | HARD_STOP", "summary_path": "..."}
```

Append-only. Diffable. Greppable. Replayable.

## §5 Agent loop

### §5.1 `dag-tick`

One iteration of the run:

```bash
#!/usr/bin/env bash
# .claude/skills/project-dag/scripts/dag-tick.sh
set -e

cd "$(git rev-parse --show-toplevel)"

# 1. Find next eligible task
TASK_JSON=$(jq -r '
  .tasks[]
  | select(.status == "pending")
  | select(.owner_class == "implementer")
  | select(.blocked == false or .blocked == null)
  | select(
      [.deps[] as $d | (.. | objects | select(.id == $d) | .status)]
      | all(. == "done" or . == "deferred" or . == "skipped")
    )
  | first
' docs/superpowers/dag/state.json)

if [ -z "$TASK_JSON" ] || [ "$TASK_JSON" = "null" ]; then
    # No eligible task — check terminal conditions
    bash .claude/skills/project-dag/scripts/dag-terminal.sh
    exit 0
fi

# 2. Mark in_progress + log
TASK_ID=$(echo "$TASK_JSON" | jq -r '.id')
jq --arg id "$TASK_ID" '.tasks |= map(if .id == $id then .status = "in_progress" | .started_at = (now | todate) else . end)' \
    docs/superpowers/dag/state.json > /tmp/state.json && mv /tmp/state.json docs/superpowers/dag/state.json
echo "{\"ts\":\"$(date -Is)\",\"event\":\"task_started\",\"task_id\":\"$TASK_ID\"}" >> docs/superpowers/dag/run.jsonl

# 3. Dispatch subagent (via the agent's Agent tool)
# This step is invoked from within the Claude session running /loop /dag-tick.
# The shell script orchestrates; the subagent dispatch happens in the agent's
# thinking. See SKILL.md for the orchestration pattern.

# 4. Subagent reports back; agent updates state via dag-mark-done.sh / dag-mark-blocked.sh
# (Those scripts are invoked by the agent after subagent completion.)

# 5. ScheduleWakeup for next iteration
# Done by the agent via the ScheduleWakeup tool, not the shell script.
```

The shell script handles state mutations; the agent's Claude session handles subagent dispatch + scheduling.

### §5.2 Eligibility check (in detail)

```python
def next_actionable(state):
    for plan in sorted(state.plans, key=lambda p: p.id):
        for task in sorted(state.tasks_in_plan(plan.id), key=lambda t: t.checkbox_line):
            if task.status != "pending":
                continue
            if task.owner_class != "implementer":
                continue
            if task.blocked:
                continue
            if not all(state.task(d).status in ("done", "deferred", "skipped") for d in task.deps):
                continue
            return task
    return None
```

Sequential within plan (no skipping ahead). Alphabetical across plans (deterministic; could be randomized later but determinism aids debugging).

### §5.3 Subagent dispatch

Per task, agent dispatches an SDD-style implementer subagent (per `superpowers:subagent-driven-development`). Prompt template includes:
- Task title + spec/plan reference
- Dependency commits (so subagent reads recent context)
- Hard scope boundaries (subagent CANNOT decide to add scope; if it surfaces a need, escalate via BLOCKED)
- Time budget (default 30 min; longer for known-large tasks)
- Critic gate awareness (subagent knows the dispatch-critics gate runs on commit)

After commit lands, agent dispatches the spec compliance reviewer (per SDD), then code quality reviewer. Both must approve. Then mark task `done`.

### §5.4 Failure handling

| Subagent return | Action |
|---|---|
| `DONE` | Mark task `done`; advance reviewers; if both PASS, finalize; if either FAILS, dispatch implementer fix-up; if 2 fix-ups fail, mark `blocked` |
| `DONE_WITH_CONCERNS` | Same as DONE but log concerns to run.jsonl + daily synthesis |
| `BLOCKED` | Mark task `blocked` with reason; retry_count += 1; if retry_count < 2, requeue (next_actionable picks it up next iteration); else escalate |
| `NEEDS_CONTEXT` | Agent provides additional context, re-dispatches; same retry budget |

### §5.5 Hard-stop conditions

Agent immediately halts run + posts:
- 3+ consecutive task_blocked events without intervening task_done
- Critic FAIL on a commit (any of the 6 Spec D critics)
- Test failure agent can't resolve in 2 retries
- An (D)-spec-needed task surfaces (always halt; never auto-attempt spec brainstorm)
- Allowlist edit detected (`engine/build.rs` modification) — even with critic PASS, user must weigh in
- Schema-hash baseline change

### §5.6 Terminal conditions

Run terminates (clean exit, not hard-stop) when no eligible task remains AND no in-flight subagents:

| Terminal | Condition | Report |
|---|---|---|
| `DAG_COMPLETE` | All tasks `done | deferred | skipped` | Celebration message + commit-count + roadmap items closed |
| `HUMAN_BLOCKED` | Eligible-task pool is empty BUT plan-writer / spec-needed / human-needed tasks remain | List of escalation queue (`pending-decisions.md`) — user resolves, re-runs |
| `HARD_STOP` | Hard-stop condition fired | Specific blocker + recommended action |

## §6 Bootstrap

### §6.1 `dag-bootstrap`

Parses ROADMAP.md + scans `docs/superpowers/plans/*.md` + reads existing `state.json` (if present, for migration) → produces fresh `state.json`.

```python
# .claude/skills/project-dag/scripts/bootstrap.py (sketch)

def bootstrap():
    roadmap = parse_roadmap("docs/ROADMAP.md")  # tier headings → roadmap items
    plans = []
    tasks = []
    
    for plan_file in glob("docs/superpowers/plans/*.md"):
        plan = parse_plan_header(plan_file)  # title, supersedes, etc.
        plan_tasks = parse_checkboxes(plan_file)  # one entry per `- [ ]` / `- [x]` line
        
        # Map plan to roadmap item by reference
        roadmap_parent = match_roadmap_item(plan, roadmap)
        plan.roadmap_parent = roadmap_parent.id if roadmap_parent else None
        
        # Sequential within-plan deps
        for i, task in enumerate(plan_tasks):
            if i > 0:
                task.deps.append(plan_tasks[i-1].id)
            task.owner_class = classify_owner(task)  # implementer / plan-writer / etc.
        
        plans.append(plan)
        tasks.extend(plan_tasks)
    
    # Detect roadmap items lacking plans → emit plan-writer meta-tasks
    for item in roadmap:
        if not item.has_plan:
            tasks.append(Task(
                id=f"{item.id}.write-plan",
                owner_class="plan-writer",
                title=f"Write plan for: {item.title}",
                deps=[],  # spec must exist; if not, this becomes spec-needed
            ))
    
    # Detect roadmap items lacking specs → emit spec-needed meta-tasks
    for item in roadmap:
        if not item.has_spec and not item.has_plan:
            # Replace plan-writer task with spec-needed
            ...
    
    # Reconcile with existing state.json: preserve completed_commit, critic_verdicts, etc.
    if exists("docs/superpowers/dag/state.json"):
        merge_completed_metadata(tasks)
    
    write_state_json(roadmap, plans, tasks)
```

### §6.2 Cross-plan dependencies

The bootstrap doesn't infer cross-plan deps automatically — those need explicit declaration. Plans declare their dependencies via a YAML front-matter or a "Depends on" line in the plan header:

```markdown
# Plan 6 — GpuBackend foundation

> **Depends on:** Plan 5 (ComputeBackend trait extraction)
```

Bootstrap parses these and adds the cross-plan `task.deps` link.

### §6.3 Owner classification

`classify_owner(task)` runs heuristics:

- Task body contains `Write the failing test` / `Run test` / `Implement` / `Edit` → `implementer`
- Task body or plan header indicates plan-write meta-task → `plan-writer`
- Spec brainstorm / "Brainstorm with user" → `spec-needed`
- Plan body says "User decides" / "Design call" / "UX consultation" → `human-needed`

Default: `implementer` (most plan tasks).

## §7 Observability

### §7.1 `run.jsonl`

See §4.4. Append-only. Agent emits a line per state mutation. Persistent across sessions. Greppable.

Useful one-liners:
```bash
# How many tasks closed today?
jq -c 'select(.event == "task_done" and .ts >= "2026-04-25T00:00:00Z")' docs/superpowers/dag/run.jsonl | wc -l

# Which plans had progress?
jq -r 'select(.event == "task_done") | .task_id | split(".")[0]' docs/superpowers/dag/run.jsonl | sort -u

# Recent blockers?
jq -c 'select(.event == "task_blocked")' docs/superpowers/dag/run.jsonl | tail -10
```

### §7.2 Daily synthesis

Agent posts a chat message every ~6-8 hours of run wall-clock time. Format:

```markdown
## DAG run — daily synthesis (2026-04-25, day 2)

**Tasks closed today:** 14 (across 4 plans)
- Plan 4 — debug & trace: 6/12 tasks (50%)
- Plan 5 — ComputeBackend trait: 4/8 tasks (50%)
- Mems behavior: 2/3 tasks (67%) — plan complete after 1 more
- Plan 6 — GpuBackend: 2/14 tasks (14%)

**Plans completed:** 0 (Mems behavior at 67%, expected to close tomorrow)

**Blockers introduced:** 1
- `plan-5-task-7`: subagent BLOCKED on missing trait method `Backend::tick_count`. Reason: spec/runtime.md §24 doesn't specify it. Marked for user decision.

**Critic verdicts of note:**
- 14 PASS verdicts (no FAILs)
- 1 allowlist-gate dispatch (Plan 4 added `engine/src/trace.rs`); both biased-against critics returned PASS

**Tomorrow's priorities (top 3):**
- `plan-4-task-7`: Wire trace_mask into step_full
- `plan-5-task-8`: ComputeBackend::flush method
- `plan-6-task-3`: GpuBackend::new with engine_data buffers

Run continues; next wake in 60s.
```

### §7.3 Interrupt-driven escalations

Agent posts immediately (not waiting for daily) on:
- Critic FAIL on any commit (with the critic's reasoning)
- (D)-gated work surfacing (plan-writer or spec-needed task encountered; queued in `pending-decisions.md`)
- 3+ consecutive blocked tasks (systemic issue signal)
- Test failure agent can't resolve

Escalation post format:
```markdown
## DAG run — escalation (2026-04-25 16:42)

**Type:** allowlist-gate critic FAIL

**Task:** `plan-7-task-3` — Add per-creature spatial index

**Commit:** `abc1234` (rolled back)

**Critic reasoning:**
> The added `engine/src/spatial_per_creature.rs` is rule-aware (wolves-vs-deer
> creature-type mapping baked in). Per Spec B' §5 D11, this should live in
> engine_rules, not engine. Recommend moving the file or refactoring to be
> creature-type-agnostic.

**Run paused on this branch.** Other plans continue. Resolve via:
- (a) Move to engine_rules + retry
- (b) Refactor to be creature-agnostic + retry
- (c) Defer the task; agent skips it
```

### §7.4 `pending-decisions.md`

Append-only file in the repo. One section per pending decision:

```markdown
# Pending Decisions

## 2026-04-25 14:33 — plan-write request: Plan 5 (ComputeBackend trait)

Spec: `docs/spec/runtime.md` §24
Roadmap tier: Engine plans not yet written
Existing plans: none

The agent encountered the "Plan 5" roadmap item with no plan drafted. Per (D)
tiered autonomy, plan-writing is gated. Approve to draft via writing-plans skill?

**Status:** awaiting user
**To approve:** edit this section and add `**APPROVED:** [your name]` line below.

---

## 2026-04-25 16:42 — spec-needed: Mission system

Roadmap tier: Game / UX layer
Existing spec: none
Brainstorming required: yes

The agent encountered the "Mission system" roadmap item with no spec. Spec
brainstorming is always human-gated; the agent will not auto-brainstorm.

**Status:** awaiting user (always — brainstorm is interactive)
**To proceed:** start a fresh chat with `/brainstorm Mission system`. The agent
will not retry this until a spec lands.
```

User reviews + responds; agent reads on next bootstrap.

## §8 Tiered-autonomy gate (D)

Per task `owner_class`:

### §8.1 `implementer`

Agent dispatches subagent freely. No user intervention until terminal or hard-stop.

### §8.2 `plan-writer`

Agent encounters: emit a `pending-decisions.md` entry titled "plan-write request: [item]". Skip the task in this iteration. Continue with next eligible.

If user approves later (re-bootstrap detects approval), agent invokes `superpowers:writing-plans` with the spec as input. Drafts the plan; commits it; re-bootstrap to populate the new plan's tasks; proceeds.

### §8.3 `spec-needed`

Agent encounters: emit `pending-decisions.md` entry titled "spec-needed: [item]". Skip. Never auto-resolves — spec brainstorming is always human-led.

If a spec lands later (the user starts a fresh brainstorm session), bootstrap re-detects and demotes the meta-task to `plan-writer`.

### §8.4 `human-needed`

Agent never picks. These are roadmap items requiring genuine human work (UX decisions, external coordination, business calls). They sit in the DAG as documentation of "what's left for humans" but never gate the agent's progress.

### §8.5 Architectural decisions

These are caught by the existing critic gate (Spec D-amendment). When a commit triggers `critic-allowlist-gate` or `critic-schema-bump`, the gate either passes (agent proceeds) or fails (hard-stop, escalation). Agent doesn't try to override.

## §9 Decision log

- **D1.** (D) Tiered autonomy. Implementation: free; plan-writing: ask; spec-writing: always ask; architecture: critic-gated.
- **D2.** (α) Per-task iteration via subagent-driven-development.
- **D3.** Plan files + sidecar JSON. No TaskWarrior. State has two sources (checkboxes + state.json); reconciled by bootstrap.
- **D4.** Hybrid observability: `run.jsonl` + daily chat post + interrupt escalations + hard-stops + `pending-decisions.md`.
- **D5.** Single DAG, any reachable eligible node. No priority weighting; deterministic ordering by plan-id then checkbox-line.
- **D6.** (D) Terminal halt. Run terminates clean when only human-gated work remains. User handles batches periodically; re-bootstrap + re-run.
- **D7.** Bootstrap parses ROADMAP.md + plan files; cross-plan deps from explicit "Depends on:" declarations in plan headers.
- **D8.** Owner-class classification heuristics (implementer / plan-writer / spec-needed / human-needed); defaults to implementer.
- **D9.** No multi-agent. One run, one DAG, one Claude session at a time.
- **D10.** Run state in git: `state.json` and `run.jsonl` are tracked; commits to them are minor housekeeping, batched per iteration.

## §10 Open questions

These warrant resolution during implementation but don't block the spec:

1. **Cross-plan dep discovery.** Plans must declare deps explicitly; the spec assumes a "Depends on:" line in plan headers. What's the syntax? YAML front-matter or markdown convention? Pin during plan-template update.
2. **Subagent dispatch from inside `/loop` autonomous mode.** Claude Code's `/loop` provides cross-session continuity via `ScheduleWakeup`. The agent's Claude session calls `Agent` to dispatch SDD subagents inside a single iteration. Verify this composes cleanly in practice.
3. **Critic-output staleness during multi-day run.** The dispatch-critics gate caches verdicts in `.claude/critic-output-*.txt`. Long-running agent must invalidate when crossing engine-touching boundaries. Reference Spec D-amendment's mtime-based stale guard.
4. **Concurrent state.json writes.** Single agent → no concurrency. But what if the user manually edits a checkbox in a plan file mid-run? Bootstrap reconciles on next run; the agent's iteration sees stale state. Acceptable for v1; add a stat-check on state.json mtime if it becomes a problem.
5. **Plan-completion bookkeeping.** Marking a task `done` should propagate `tasks_done++` to the parent plan; marking the last task done should propagate to the parent roadmap item. Idempotent recompute on each iteration is simplest; lazy.

## §11 Out of scope (explicit)

- Replacing or modifying ROADMAP.md, plan files, or any spec.
- Real-time progress dashboards / web UI.
- Multi-agent / parallel runs against same DAG.
- Effort estimation / time-tracking.
- Replacing dispatch-critics or build sentinels (existing gates).
- Plan-template updates (separate work; happens once across all existing plans).
- Auto-detection of completed roadmap items from git history (manual via re-bootstrap is enough for v1).

## §12 Implementation phases

### Phase 1 — Bootstrap + state model (~1 day)
- Write `bootstrap.py` parsing ROADMAP + plans → `state.json`
- Verify against current repo state (~30 roadmap items + 8 plan files + ~150 task checkboxes)
- Hand-test: classify owner-class correctly; sequential within-plan deps; cross-plan deps via "Depends on" lines

### Phase 2 — Plan-drafting prep (~1-2 days, before run starts)
- Per ROADMAP.md, identify all "Drafted (spec exists, plan does not)" + "Engine plans not yet written" items
- Draft plans for each in this conversation, with user oversight
- Re-bootstrap; new plans contribute to the DAG

### Phase 3 — `dag-tick` + agent loop (~1 day)
- Write `dag-tick.sh`, `dag-mark-done.sh`, `dag-mark-blocked.sh`, `dag-terminal.sh`
- Define the `.claude/skills/project-dag/SKILL.md` with the agent invocation pattern (how the Claude session running `/loop /dag-tick` invokes Agent + ScheduleWakeup)
- Smoke test: run for 1 iteration; verify state mutation + run.jsonl append

### Phase 4 — Observability (~½ day)
- Daily synthesis post template
- Escalation post template
- `dag-status.sh` for ad-hoc human queries

### Phase 5 — Smoke run (~1 day)
- Run the agent for ~4 hours against the actual DAG
- Verify daily synthesis fires, escalations land, terminal conditions detected
- Iterate on owner-class heuristics + escalation triggers

### Phase 6 — Production run (multi-day)
- Land all autonomous-implementable tasks
- User periodically resolves `pending-decisions.md` batches
- Run continues until DAG complete OR explicit user stop

## §13 Cross-references

- `docs/ROADMAP.md` — source for roadmap-tier nodes
- `docs/superpowers/plans/*.md` — source for task nodes (checkbox state)
- `docs/superpowers/specs/2026-04-24-project-dag-skill-design.md` — v1, superseded
- `docs/superpowers/specs/2026-04-25-engine-crate-split-design-v2.md` — Spec B', architectural settlement
- `.claude/skills/critic-*/SKILL.md` — Spec D, 6 biased-against critics
- `.claude/scripts/dispatch-critics.sh` — Spec D-amendment orchestrator
- `.githooks/pre-commit` — Spec B' + D-amendment pre-commit gate
- `superpowers:subagent-driven-development` — per-task subagent dispatch pattern
- `superpowers:writing-plans` — invoked for plan-writer tasks (after user approval)
- Claude Code `/loop` autonomous mode + `ScheduleWakeup` — cross-session triggering
