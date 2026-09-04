# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What this holds

3 standalone, self-contained HTML files — no build step, no shared assets, no server
code in this directory. Each is a single-page live-simulation viewer that connects to
`ws://localhost:9090` for a JSON (and, for `voxel.html`, binary) frame stream and
falls back to synthetic demo data when no server answers:

- `index.html` — "Cartographer's View": a 2D canvas world map (factions panel, tile
  terrain rendering, city-grid buildings, chronicle event feed, playback timeline).
  Pure Canvas 2D, no external libraries.
- `3d.html` — "World Sim — 3D View": a Three.js (r128, loaded from cdnjs/jsdelivr)
  scene rendering the same world-sim frame data in 3D — greedy-meshed terrain,
  instanced buildings/NPCs/monsters/resource nodes, WFC building interiors, an NPC
  detail panel, and an economy/contracts board.
- `voxel.html` — "Voxel World — WebGPU SDF": a raw WebGPU (no library) raymarched
  signed-distance-field renderer over a voxel material volume, with hand-written WGSL
  compute shaders (brute-force JFA-style SDF build + raymarch + blit). Currently
  hardcoded into a local test-mode single 16×16×16 chunk (`connectWS()` — the real
  chunk-streaming path — is defined but never called; `init()` disables it and fills
  `materialData` synthetically instead).

All three expect a specific JSON frame shape (`tick`, `summary.{alive_npcs,
alive_monsters,season,year}`, `settlements[]`, `entities[]` with `kind`/`pos`/`alive`,
`regions[]`, `trade_routes[]`, `events[]`, optionally `city_grids[]`,
`building_interiors[]`, `voxel_surfaces[]`/`voxel_chunks[]`, `selected_npc`) — this
shape is not defined or documented anywhere else in this directory.

## How to invoke

Open directly in a browser (`file://` or any static file server) — no build step:
```bash
# any static server, e.g.:
python3 -m http.server --directory web 8080
# then open http://localhost:8080/index.html, /3d.html, or /voxel.html
```
`3d.html` needs WebGL; `voxel.html` needs WebGPU (Chrome/Edge 113+) and shows an
on-page warning otherwise. Without a server on `ws://localhost:9090`, `index.html`
and `3d.html` auto-play synthetic demo frames; `voxel.html` just renders its hardcoded
test chunk and never attempts the WebSocket path in its current form.

## What in `crates/` depends on this

Nothing serves the expected data. Searched all of `crates/` for `9090`, `WebSocket`,
`TraceFrame`, and references to `web/index.html`/`3d.html`/`voxel.html` — no matches.
No crate currently runs a WebSocket server on port 9090 or produces the frame JSON
shape these pages expect.

## Live or stale

**Unclear whether these were ever wired up in this checkout, but currently orphaned
either way** — there is no live counterpart in `crates/` today. The frame shape
(`city_grids`, `building_interiors`, `voxel_surfaces`, economy `contracts`, NPC
`needs`/`emotions`/`goals`/`memories`) implies a fairly complete world-sim trace
server once existed or was planned, but nothing in the current crate set produces it.
Root `CLAUDE.md`'s Phase 7 wolf-sim wipe (2026-05-02) deleted the heaviest parts of the
old engine (`engine_gpu`, `engine_rules`, `viz`, `tactical_sim`, `src/`) — a websocket
trace server is a plausible casualty of that wipe, but this could not be confirmed from
the current tree (no `git log` available — this checkout is not a git repo). Treat as
dead frontend code pending a live trace-frame server; do not assume any crate feeds it.
