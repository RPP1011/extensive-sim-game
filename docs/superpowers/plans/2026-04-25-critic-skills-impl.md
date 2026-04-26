# Critic Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the 6 critic skills (`critic-compiler-first`, `critic-schema-bump`, `critic-cross-backend-parity`, `critic-no-runtime-panic`, `critic-reduction-determinism`, `critic-allowlist-gate`) as project-local SKILL.md files at `.claude/skills/<name>/SKILL.md`. Each file's content is verbatim from `docs/superpowers/specs/2026-04-25-critic-skills-design.md` §4.1–§4.6.

**Architecture:** Pure file creation; no Rust code, no test runs against engine. The SKILL.md files are static prompt + few-shot content the Skill tool surfaces or the Agent tool dispatches. Smoke-test verifies the markdown loads and the prompt is well-formed.

**Tech Stack:** Markdown only. `git add` + `git commit`. One bash verification per skill (markdown structure check).

## Architectural Impact Statement

- **Existing primitives searched:** existing skills in `~/.claude/plugins/cache/claude-plugins-official/superpowers/5.0.7/skills/` (templates for SKILL.md schema), `.claude/skills/` (currently empty in this project — first project-local skill batch).
- **Decision:** new (project-local skills are net-new; the framework supports them but this project hasn't used it yet).
- **Rule-compiler touchpoints:** none — no DSL changes, no engine code touched.
- **Hand-written downstream code:** none — entirely prompt content.
- **Constitution check:**
  - P1: PASS — no engine extension; no rule logic introduced.
  - P2: PASS — no schema changes.
  - P3: PASS — no engine behavior; no parity concern.
  - P4: PASS — no IR changes.
  - P5: PASS — no RNG code.
  - P6: PASS — no state mutations.
  - P7: PASS — no event additions.
  - P8: PASS — this AIS section satisfies P8.
  - P9: PASS — tasks will close with verified commit SHAs.
  - P10: PASS — no runtime code introduced.
  - P11: PASS — no reduction code.
- **Re-evaluation:** [x] AIS reviewed at design phase. [x] AIS reviewed post-design — 6 SKILL.md files added (738 total ln), CLAUDE.md +1 ln, llms.txt +9 ln, .gitignore exception added. No engine code touched, no rule logic introduced. Identified follow-up: dispatch-critics wrapper skill + Stop hook for auto-dispatch on engine touches.

---

## File Structure

All 6 skills land under `.claude/skills/<name>/SKILL.md`. Each file's content already exists verbatim in the spec — the implementation is "extract from spec section and place at the right path."

```
.claude/skills/
  critic-compiler-first/SKILL.md          (from spec §4.1)
  critic-schema-bump/SKILL.md             (from spec §4.2)
  critic-cross-backend-parity/SKILL.md    (from spec §4.3)
  critic-no-runtime-panic/SKILL.md        (from spec §4.4)
  critic-reduction-determinism/SKILL.md   (from spec §4.5)
  critic-allowlist-gate/SKILL.md          (from spec §4.6)
```

Plus:
- `CLAUDE.md` — add a one-line pointer under "Conventions" mentioning the critics + when to invoke.
- `docs/llms.txt` — add a "Critics" section pointing at `.claude/skills/critic-*/SKILL.md`.

Verification:
- Each SKILL.md must have valid YAML frontmatter (`name:`, `description:`).
- Each must contain the required sections: Role, Principle, Required tools, BAD examples (≥3), GOOD examples (≥1), Output format.
- The Skill loads cleanly when invoked (`Skill` tool with `skill: critic-compiler-first` returns the skill body).

---

### Task 1: `critic-compiler-first` skill

**Files:**
- Create: `.claude/skills/critic-compiler-first/SKILL.md`

- [x] **Step 1: Create the directory**

```bash
mkdir -p .claude/skills/critic-compiler-first
```

- [x] **Step 2: Write SKILL.md**

Open `docs/superpowers/specs/2026-04-25-critic-skills-design.md`. Locate §4.1 (`critic-compiler-first` (P1)). Copy the markdown body INSIDE the outer code fence (i.e., starting from the line `---` (frontmatter open) through the final `If TOOLS RUN is empty → automatic FAIL.` line and the closing `---` if present). Paste verbatim into `.claude/skills/critic-compiler-first/SKILL.md`.

The file MUST start with frontmatter:

```markdown
---
name: critic-compiler-first
description: Use when reviewing changes that add or modify behavior in crates/engine/src/ — including new modules, new struct impls, or any code that performs a per-tick action. Biased toward rejecting hand-written rule logic that should be DSL-emitted.
---

# Critic: Compiler-First (P1)

## Role
...
```

The body matches §4.1 exactly. Don't paraphrase, don't summarize.

- [x] **Step 3: Verify frontmatter and required sections**

```bash
head -5 .claude/skills/critic-compiler-first/SKILL.md
```

Expect first 4 lines to start with `---`, contain `name: critic-compiler-first`, contain `description:` (single line), close with `---`.

```bash
grep -c "^## Role\|^## Principle\|^## Required tools\|^## Few-shot BAD\|^## Few-shot GOOD\|^## Output format" .claude/skills/critic-compiler-first/SKILL.md
```

Expect: `6` (one heading per required section).

```bash
grep -c "^### Example [0-9]*: " .claude/skills/critic-compiler-first/SKILL.md
```

Expect: ≥4 (at least 3 BAD + 1 GOOD).

- [x] **Step 4: Verify file is valid markdown (no broken fences)**

```bash
awk '/^```/ { count++ } END { if (count % 2 != 0) print "ODD count"; else print "PAIRED" }' .claude/skills/critic-compiler-first/SKILL.md
```

Expect: `PAIRED` (every code fence has a closing fence).

- [x] **Step 5: Commit**

```bash
git add .claude/skills/critic-compiler-first/SKILL.md
git commit -m "feat(skills): critic-compiler-first — biased P1 reviewer with few-shot"
```

---

### Task 2: `critic-schema-bump` skill

**Files:**
- Create: `.claude/skills/critic-schema-bump/SKILL.md`

- [x] **Step 1: Create the directory**

```bash
mkdir -p .claude/skills/critic-schema-bump
```

- [x] **Step 2: Write SKILL.md**

From `docs/superpowers/specs/2026-04-25-critic-skills-design.md` §4.2, copy the inner markdown body verbatim into `.claude/skills/critic-schema-bump/SKILL.md`. Frontmatter:

```markdown
---
name: critic-schema-bump
description: Use when reviewing changes that touch SimState SoA fields, event variant definitions, mask predicate semantics, or scoring row contracts. Biased toward rejecting changes that don't regenerate crates/engine/.schema_hash.
---

# Critic: Schema-Hash Bumps on Layout Change (P2)
...
```

Body matches §4.2 exactly.

- [x] **Step 3: Verify frontmatter and structure**

```bash
head -5 .claude/skills/critic-schema-bump/SKILL.md
grep -c "^## Role\|^## Principle\|^## Required tools\|^## Few-shot BAD\|^## Few-shot GOOD\|^## Output format" .claude/skills/critic-schema-bump/SKILL.md
grep -c "^### Example [0-9]*: " .claude/skills/critic-schema-bump/SKILL.md
awk '/^```/ { count++ } END { if (count % 2 != 0) print "ODD"; else print "PAIRED" }' .claude/skills/critic-schema-bump/SKILL.md
```

Expect: name/description present; 6 sections; ≥4 examples; PAIRED fences.

- [x] **Step 4: Commit**

```bash
git add .claude/skills/critic-schema-bump/SKILL.md
git commit -m "feat(skills): critic-schema-bump — biased P2 reviewer with few-shot"
```

---

### Task 3: `critic-cross-backend-parity` skill

**Files:**
- Create: `.claude/skills/critic-cross-backend-parity/SKILL.md`

- [x] **Step 1: Create the directory**

```bash
mkdir -p .claude/skills/critic-cross-backend-parity
```

- [x] **Step 2: Write SKILL.md**

From spec §4.3, copy inner markdown body verbatim. Frontmatter:

```markdown
---
name: critic-cross-backend-parity
description: Use when reviewing new engine behavior, physics rules, view folds, or anything that runs in the per-tick path. Biased toward rejecting changes that won't preserve byte-equal SHA-256 across SerialBackend and GpuBackend.
---
```

Body matches §4.3 exactly.

- [x] **Step 3: Verify**

```bash
head -5 .claude/skills/critic-cross-backend-parity/SKILL.md
grep -c "^## Role\|^## Principle\|^## Required tools\|^## Few-shot BAD\|^## Few-shot GOOD\|^## Output format" .claude/skills/critic-cross-backend-parity/SKILL.md
grep -c "^### Example [0-9]*: " .claude/skills/critic-cross-backend-parity/SKILL.md
awk '/^```/ { count++ } END { if (count % 2 != 0) print "ODD"; else print "PAIRED" }' .claude/skills/critic-cross-backend-parity/SKILL.md
```

Expect: name/desc; 6 sections; ≥4 examples; PAIRED.

- [x] **Step 4: Commit**

```bash
git add .claude/skills/critic-cross-backend-parity/SKILL.md
git commit -m "feat(skills): critic-cross-backend-parity — biased P3 reviewer with few-shot"
```

---

### Task 4: `critic-no-runtime-panic` skill

**Files:**
- Create: `.claude/skills/critic-no-runtime-panic/SKILL.md`

- [x] **Step 1: Create the directory**

```bash
mkdir -p .claude/skills/critic-no-runtime-panic
```

- [x] **Step 2: Write SKILL.md**

From spec §4.4, copy verbatim. Frontmatter:

```markdown
---
name: critic-no-runtime-panic
description: Use when reviewing changes to crates/engine/src/step.rs, kernels in crates/engine_gpu/, or any code in the deterministic per-tick path. Biased toward rejecting unwrap/expect/panic on hot paths.
---
```

Body matches §4.4 exactly.

- [x] **Step 3: Verify**

```bash
head -5 .claude/skills/critic-no-runtime-panic/SKILL.md
grep -c "^## Role\|^## Principle\|^## Required tools\|^## Few-shot BAD\|^## Few-shot GOOD\|^## Output format" .claude/skills/critic-no-runtime-panic/SKILL.md
grep -c "^### Example [0-9]*: " .claude/skills/critic-no-runtime-panic/SKILL.md
awk '/^```/ { count++ } END { if (count % 2 != 0) print "ODD"; else print "PAIRED" }' .claude/skills/critic-no-runtime-panic/SKILL.md
```

Expect: name/desc; 6 sections; ≥4 examples; PAIRED.

- [x] **Step 4: Commit**

```bash
git add .claude/skills/critic-no-runtime-panic/SKILL.md
git commit -m "feat(skills): critic-no-runtime-panic — biased P10 reviewer with few-shot"
```

---

### Task 5: `critic-reduction-determinism` skill

**Files:**
- Create: `.claude/skills/critic-reduction-determinism/SKILL.md`

- [x] **Step 1: Create the directory**

```bash
mkdir -p .claude/skills/critic-reduction-determinism
```

- [x] **Step 2: Write SKILL.md**

From spec §4.5, copy verbatim. Frontmatter:

```markdown
---
name: critic-reduction-determinism
description: Use when reviewing changes to view folds, atomic-append paths, or RNG-touching code. Biased toward rejecting reductions that aren't sort-stable or fixed-point.
---
```

Body matches §4.5 exactly.

- [x] **Step 3: Verify**

```bash
head -5 .claude/skills/critic-reduction-determinism/SKILL.md
grep -c "^## Role\|^## Principle\|^## Required tools\|^## Few-shot BAD\|^## Few-shot GOOD\|^## Output format" .claude/skills/critic-reduction-determinism/SKILL.md
grep -c "^### Example [0-9]*: " .claude/skills/critic-reduction-determinism/SKILL.md
awk '/^```/ { count++ } END { if (count % 2 != 0) print "ODD"; else print "PAIRED" }' .claude/skills/critic-reduction-determinism/SKILL.md
```

Expect: name/desc; 6 sections; ≥4 examples; PAIRED.

- [x] **Step 4: Commit**

```bash
git add .claude/skills/critic-reduction-determinism/SKILL.md
git commit -m "feat(skills): critic-reduction-determinism — biased P11 reviewer with few-shot"
```

---

### Task 6: `critic-allowlist-gate` skill

**Files:**
- Create: `.claude/skills/critic-allowlist-gate/SKILL.md`

- [x] **Step 1: Create the directory**

```bash
mkdir -p .claude/skills/critic-allowlist-gate
```

- [x] **Step 2: Write SKILL.md**

From spec §4.6, copy verbatim. Frontmatter:

```markdown
---
name: critic-allowlist-gate
description: Use when reviewing edits to crates/engine/build.rs ALLOWED_TOP_LEVEL or ALLOWED_DIRS. Biased toward rejecting additions; the bar for new engine primitives is high.
---
```

Body matches §4.6 exactly. Note this critic's BAD examples have section names like "Adding `theory_of_mind` because it's 'infrastructure'" rather than "Example 1:" — the structural check (≥3 BAD examples) still applies but the heading style varies. Make sure the file has at least 3 H3-level BAD example sections and at least 1 GOOD.

- [x] **Step 3: Verify**

```bash
head -5 .claude/skills/critic-allowlist-gate/SKILL.md
grep -c "^## Role\|^## Principle\|^## Required tools\|^## Few-shot BAD\|^## Few-shot GOOD\|^## Output format" .claude/skills/critic-allowlist-gate/SKILL.md
```

Expect: name/desc present, 6 sections (note: this critic uses "Few-shot BAD examples" and "Few-shot GOOD examples (very rare)" — the regex above counts "Few-shot BAD" + "Few-shot GOOD" as long as the heading begins with one of those strings).

```bash
awk '/^### / && /[Bb]ad/ { bad++ } /^### / && /[Gg]ood/ { good++ } END { print "bad:", bad, "good:", good }' .claude/skills/critic-allowlist-gate/SKILL.md
```

(This counts H3 examples that mention bad/good; allowlist-gate uses descriptive H3 names.) An alternative count:

```bash
grep -c "\*\*Verdict:\*\* FAIL" .claude/skills/critic-allowlist-gate/SKILL.md
grep -c "\*\*Verdict:\*\* PASS" .claude/skills/critic-allowlist-gate/SKILL.md
```

Expect: ≥3 FAIL examples, ≥1 PASS example.

```bash
awk '/^```/ { count++ } END { if (count % 2 != 0) print "ODD"; else print "PAIRED" }' .claude/skills/critic-allowlist-gate/SKILL.md
```

Expect: PAIRED.

- [x] **Step 4: Commit**

```bash
git add .claude/skills/critic-allowlist-gate/SKILL.md
git commit -m "feat(skills): critic-allowlist-gate — biased reviewer for engine/build.rs allowlist edits"
```

---

### Task 7: Smoke-test invocation

**Goal:** verify at least one critic skill loads and emits the expected prompt shape.

**Files:**
- (no new files; smoke test only)

- [x] **Step 1: Manually invoke `critic-compiler-first` via the Skill tool**

This is a manual verification (the test runner can't directly invoke a skill). The implementer should:

1. In their Claude Code session, invoke the `Skill` tool with `skill: critic-compiler-first`.
2. Verify the skill body loads and contains the frontmatter + Role + Principle + Required tools + BAD examples + Output format.
3. If it doesn't load, the cause is one of:
   - Frontmatter malformed (missing `---`, missing `name:` or `description:`).
   - File at wrong path (`.claude/skills/critic-compiler-first/SKILL.md` not `.../critic-compiler-first.md`).
   - YAML parse error (special characters in `description:` not escaped).

Document the verification in the commit message of the next task (Task 8). No file changes here.

- [x] **Step 2: (Optional) Dispatch a fresh-context Agent invocation as independence test**

Equivalent to a real allowlist-gate dispatch, but on a no-op target. The implementer dispatches:

```
Agent tool:
  description: "Smoke-test critic-compiler-first"
  subagent_type: general-purpose
  model: haiku
  prompt: |
    [Paste full body of .claude/skills/critic-compiler-first/SKILL.md here.]

    ## Target

    diff_sha: HEAD~5
    Range: HEAD~5..HEAD

    Run the required tools, follow the format, return your verdict.
```

The subagent should produce output in the rigid format. The verdict will probably be PASS (HEAD~5..HEAD doesn't introduce engine extensions); the point is verifying the format renders.

- [x] **Step 3: No commit (smoke test only). Record findings in Task 8's commit message.**

---

### Task 8: Update `CLAUDE.md` and `docs/llms.txt`

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/llms.txt`

- [x] **Step 1: Add a "Critics" line to `CLAUDE.md` Conventions section**

Read current `CLAUDE.md` ("## Conventions" section, ~line 40). Add a single bullet:

```markdown
- Critic skills (`.claude/skills/critic-*`) gate engine extensions and architectural changes. Invoke via Skill tool for in-context review or via Agent dispatch for fresh-context independent verdicts (e.g., the engine/build.rs allowlist gate requires two parallel Agent invocations — see `docs/superpowers/specs/2026-04-25-engine-crate-split-design.md` §5.2).
```

Verify line count stays ≤100:

```bash
wc -l CLAUDE.md
```

If over, the constitution-section bullet about the SessionStart hook is a candidate for tightening but DO NOT TRIM without justification. If at exactly 100 lines, leave alone.

- [x] **Step 2: Add a "Critics" subsection to `docs/llms.txt`**

Read current `docs/llms.txt`. Under the existing structure (after the "Process" section), add:

```markdown
## Critics

- [critic-compiler-first](.claude/skills/critic-compiler-first/SKILL.md): biased P1 reviewer.
- [critic-schema-bump](.claude/skills/critic-schema-bump/SKILL.md): biased P2 reviewer.
- [critic-cross-backend-parity](.claude/skills/critic-cross-backend-parity/SKILL.md): biased P3 reviewer.
- [critic-no-runtime-panic](.claude/skills/critic-no-runtime-panic/SKILL.md): biased P10 reviewer.
- [critic-reduction-determinism](.claude/skills/critic-reduction-determinism/SKILL.md): biased P11 reviewer.
- [critic-allowlist-gate](.claude/skills/critic-allowlist-gate/SKILL.md): biased gate for engine/build.rs allowlist edits.
```

- [x] **Step 3: Verify links**

```bash
grep -oE '\(\.claude/skills/[^)]*\)' docs/llms.txt | tr -d '()' | sort -u | while read f; do
    [ -f "$f" ] || echo "MISSING: $f"
done
```

Expect: zero output (all 6 skill files exist).

- [x] **Step 4: Commit**

```bash
git add CLAUDE.md docs/llms.txt
git commit -m "docs: register critic skills in CLAUDE.md + llms.txt index

Smoke test (Task 7): critic-compiler-first SKILL.md loads via Skill tool;
frontmatter parses; required sections present. Independent dispatch via
Agent tool with prompt = full skill body works as expected — produces
output in rigid VERDICT/EVIDENCE/REASONING/TOOLS-RUN format."
```

---

### Task 9: AIS post-design re-evaluation

**Files:**
- Modify: this plan file (`docs/superpowers/plans/2026-04-25-critic-skills-impl.md`)

- [x] **Step 1: Tick the "post-design" checkbox**

Edit the AIS preamble in this plan file. Change:

```markdown
- **Re-evaluation:** [ ] AIS reviewed at design phase. [ ] AIS reviewed post-design.
```

To:

```markdown
- **Re-evaluation:** [x] AIS reviewed at design phase. [x] AIS reviewed post-design — 8 SKILL.md files added, 2 docs updated, no engine code touched, no rule logic introduced. P1–P11 PASS confirmed.
```

- [x] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-04-25-critic-skills-impl.md
git commit -m "chore(plan): tick critic-skills post-impl AIS re-evaluation"
```

---

## Self-Review

**Spec coverage:**
- §3.1 skill layout → Tasks 1-6 land all 6 SKILL.md files at correct paths.
- §3.2 invocation patterns → documented in Task 7 (smoke-test) and CLAUDE.md addition (Task 8).
- §3.3 common contract (output format) → embedded in each SKILL.md body via spec verbatim copy.
- §4.1–§4.6 per-critic content → Tasks 1-6 each copy from their corresponding spec section.
- §5 integration with Spec B → Task 8's CLAUDE.md addition cross-refs the Spec B allowlist gate workflow.
- §6 integration with Spec C (project-DAG) → not addressed in this plan; Spec C's plan will integrate.
- §7 decision log → no implementation needed; documented in spec.
- §8 out of scope → no tasks needed.

**Placeholder scan:** No TBDs / "implement later" / "similar to Task N". Each task has the explicit copy source (§4.X) and full verification commands.

**Type consistency:** Skill name strings match across CLAUDE.md, llms.txt, file paths, and frontmatter. The 6 names (`critic-compiler-first`, `critic-schema-bump`, `critic-cross-backend-parity`, `critic-no-runtime-panic`, `critic-reduction-determinism`, `critic-allowlist-gate`) are used identically everywhere.

---

## Execution handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task; review between tasks. Tasks 1-6 are nearly identical (copy + verify + commit) so haiku is fast and cheap.
2. **Inline Execution** — execute in this session via executing-plans.

Which approach?
