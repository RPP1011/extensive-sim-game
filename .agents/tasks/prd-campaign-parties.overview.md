# PRD Overview: Mount and Blade Style Multi-Party Campaign Flow

- File: .agents/tasks/prd-campaign-parties.json
- Stories: 12 total (12 open, 0 in_progress, 0 done)

## Quality Gates
- cargo test

## Stories
- [open] US-001: Complete party state foundation and persistence
- [open] US-002: Enforce start-menu action hierarchy and continue semantics (depends on: US-001)
- [open] US-003: Isolate menu subtitle/status text from in-session notices (depends on: US-002)
- [open] US-004: Implement character creation step 1 (faction selection) (depends on: US-002)
- [open] US-005: Implement character creation step 2 (backstory/archetype effects) (depends on: US-004)
- [open] US-006: Deliver party list control UX with command handoff (depends on: US-001, US-005)
- [open] US-007: Implement explicit region target picker workflow (depends on: US-006)
- [open] US-008: Add camera focus interpolation for party swaps (depends on: US-006)
- [open] US-009: Implement overworld-to-region transition payload (depends on: US-007, US-008)
- [open] US-010: Bootstrap region-to-local eagle-eye intro sequence (depends on: US-009)
- [open] US-011: Persist layered campaign progression and migration guards (depends on: US-010)
- [open] US-012: Add deterministic regression fixtures and screenshot stages (depends on: US-011)
