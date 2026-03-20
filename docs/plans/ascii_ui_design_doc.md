# Project Chimera — ASCII UI Design Document

**Version 1.0 — March 2026**
**Adventurer's Guild — Tactical Autobattler with AI-Driven Combat**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Philosophy](#2-design-philosophy)
3. [Screen Flow](#3-screen-flow)
4. [Shared Layout Architecture](#4-shared-layout-architecture)
5. [Screen 1: Start Menu](#5-screen-1-start-menu)
6. [Screen 2: Character Creation](#6-screen-2-character-creation)
7. [Screen 3: Overworld Map](#7-screen-3-overworld-map)
8. [Screen 4: Region View](#8-screen-4-region-view)
9. [Screen 5: Eagle Eye Intro](#9-screen-5-eagle-eye-intro)
10. [Screen 6: Mission Execution](#10-screen-6-mission-execution)
11. [Screen 7: Replay Viewer](#11-screen-7-replay-viewer)
12. [Combat Window Component](#12-combat-window-component)
13. [Morale & Culture System (UI)](#13-morale--culture-system-ui)
14. [Overworld Rendering Pipeline](#14-overworld-rendering-pipeline)
15. [Open Questions & Follow-Ups](#15-open-questions--follow-ups)
16. [Implementation Constraint: Zero Native egui Chrome](#16-implementation-constraint-zero-native-egui-chrome)

---

## 1. Overview

Project Chimera is a tactical autobattler built on a deterministic, tick-based combat simulation. The game features a Mount & Blade-style campaign overworld, procedurally generated missions, and AI-driven squad combat — all rendered in an ASCII art aesthetic using egui as the UI framework.

This document defines the UI layout, screen architecture, rendering strategies, and visual language for the ASCII interface. The game is rendered using egui but styled to look and feel like a terminal/roguelike application, with full per-character RGBA coloring via `RichText` and `ui.painter()`.

### Key Technical Context

- **Simulation engine:** Deterministic `step()` function, 100ms fixed tick, pure functional (state in, state + events out).
- **Combat positions:** Continuous `SimVec2` (floating-point 2D), discretized to a character grid for display.
- **Hero abilities:** Defined via a custom DSL (`.ability` files), highly variable count per hero (4 to 15+).
- **AI:** Neural ability evaluator, transformer decision head, GOAP planner, behavior DSL, personality roles.
- **Missions:** Multi-room dungeon sequences (Entry → Standard → Elite → Boss), procedurally generated rooms with cover, elevation, and obstacles.
- **Overworld:** Spatial grid with procedurally generated terrain, Voronoi-based region boundaries, roaming parties, faction territories.

---

## 2. Design Philosophy

### ASCII as Aesthetic, egui as Framework

The game renders through egui but presents as ASCII art. This is not a terminal application — it's a full GUI application that uses box-drawing characters, ASCII glyphs, and per-character coloring to create the roguelike aesthetic. Key advantages:

- Full RGBA per character via `RichText` or `ui.painter()`.
- Floating windows, tooltips, buttons, and scrollable regions from egui.
- Mouse interaction (hover, click, drag) layered over the ASCII canvas.
- No terminal emulator constraints — font size, window size, and layout are fully controlled.

### Color as Layer Separation

Color is the primary tool for visual hierarchy. The ASCII character set is limited, so color differentiates layers that share the same rendering space:

| Layer | Color Strategy |
|-------|---------------|
| Terrain | Muted, desaturated (dim greens, grays, tans, steel blues) |
| Faction borders | Single neutral color, subtle |
| Faction territory | Background tint at 10–15% opacity |
| Settlements | Bright faction colors, distinct glyphs |
| Roaming parties | Bright faction-colored glyphs, pop against terrain |
| Player marker | Always brightest, unique color (bright green `@`) |
| UI overlays | egui-native rendering on top of painted canvas |

### Information on Demand

The ASCII grid shows spatial state at a glance. Detailed information surfaces through hover tooltips, side panel displays, and expandable sub-panels. The player never needs to memorize what a glyph means — hovering always reveals full context.

---

## 3. Screen Flow

```
                    ┌─────────────┐
                    │ Start Menu  │
                    └──────┬──────┘
                 ┌─────────┴──────────┐
                 ▼                    ▼
        ┌────────────────┐   ┌───────────────┐
        │ Char Creation  │   │ Overworld Map │◄──── Continue
        │ (multi-step)   │   │               │
        └───────┬────────┘   └───────┬───────┘
                │                    │
                └──────► ┌───────────┘
                         ▼
                  ┌──────────────┐
                  │ Region View  │
                  │ (local map   │
                  │  + scenes)   │
                  └──────┬───────┘
                         │ Scout
                         ▼
                  ┌──────────────┐
                  │ Eagle Eye    │
                  │ Intro        │
                  └──────┬───────┘
                         │ Auto-advance
                         ▼
                  ┌──────────────┐
                  │ Mission      │
                  │ Execution    │◄───┐
                  └──────┬───────┘    │
                         │            │
                         ▼            │
                  ┌──────────────┐    │
                  │ Replay       │────┘ (back to overworld
                  │ Viewer       │       or next mission)
                  └──────────────┘
```

**Navigation rules:**
- Start Menu → New Game → Character Creation → Overworld Map
- Start Menu → Continue → Overworld Map (loaded save)
- Overworld Map → Select Region → Region View
- Region View → Scout → Eagle Eye Intro → Mission Execution
- Mission Execution → Post-mission → Replay Viewer (optional) → Overworld Map
- Most screens have a Back button returning to the previous screen

---

## 4. Shared Layout Architecture

### Primary Layout Pattern

Almost every screen uses egui's two-panel layout:

```
┌─────────────────────────────────────────────────┐
│ ┌──────────────┐ ┌────────────────────────────┐ │
│ │              │ │                            │ │
│ │  SidePanel   │ │     CentralPanel           │ │
│ │  (left)      │ │                            │ │
│ │  ~160px      │ │     (remaining space)      │ │
│ │  fixed       │ │                            │ │
│ │              │ │                            │ │
│ └──────────────┘ └────────────────────────────┘ │
│  [status bar / contextual info]                  │
└─────────────────────────────────────────────────┘
```

**SidePanel** (left, ~160px fixed width): Navigation buttons, context-specific status, player/party info.

**CentralPanel** (remaining space): Primary content — maps, combat grids, scene illustrations, dialogue.

**Bottom status bar** (optional): Contextual one-line info based on hover or selection.

### Exception: Character Creation

Character Creation uses a full-screen CentralPanel with no side panel — it's a paginated overlay that takes the entire screen for the questionnaire flow.

### Exception: Mission Execution

Mission Execution uses the SidePanel for battle management, but the CentralPanel becomes a **window manager workspace** where combat windows float as draggable, resizable `egui::Window` instances.

---

## 5. Screen 1: Start Menu

### Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ┌──────────────┐  ┌──────────────────────────────────────────────────┐  │
│  │              │  │                                                  │  │
│  │ [> New Game ]│  │      ╔═══════════════════════════════════╗       │  │
│  │              │  │      ║                                   ║       │  │
│  │ [  Continue ]│  │      ║     ADVENTURER'S  GUILD           ║       │  │
│  │              │  │      ║                                   ║       │  │
│  │ [  Settings ]│  │      ╚═══════════════════════════════════╝       │  │
│  │              │  │                                                  │  │
│  │ [  Credits  ]│  │           ~~~ background art area ~~~            │  │
│  │              │  │        /\       .  *  .       /\                 │  │
│  │ [  Quit     ]│  │       /  \  .       *       /  \   .            │  │
│  │              │  │      /    \     .       .  /    \               │  │
│  │              │  │     /______\  *    .      /______\    *         │  │
│  │              │  │        ||          .         ||                  │  │
│  └──────────────┘  └──────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Components

- **SidePanel:** 5 stacked buttons — New Game, Continue, Settings, Credits, Quit. `[> marker]` indicates keyboard focus for gamepad/keyboard navigation.
- **CentralPanel:** Title card rendered as a framed ASCII label using double-line box-drawing characters (`╔═╗║╚═╝`). Below: decorative ASCII art (mountains, stars, etc.) as atmospheric filler.
- **Button actions:** New Game → Character Creation. Continue → load save → Overworld Map. Quit → `app.quit()`.

---

## 6. Screen 2: Character Creation

### Design Reference

Mount & Blade character creation — a multi-step questionnaire where each page presents a narrative question with 3–4 choices. Selections accumulate stat bonuses and narrative flags, building the player's background, skills, and faction allegiance.

### Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 2 of 6: Your father was...                         [ < Back ]    │
│─────────────────────────────────────────────────────────────────────────│
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │   Your upbringing shaped your earliest skills.                    │  │
│  │   Choose the life your father led:                                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────┐   │
│  │ (●) A veteran       │ │ ( ) A travelling    │ │ ( ) A farmer    │   │
│  │     soldier         │ │     merchant        │ │                 │   │
│  │                     │ │                     │ │                 │   │
│  │  +2 STR, +1 END    │ │  +2 CHA, +1 INT    │ │  +2 END, +1 STR │   │
│  │  Start: shortsword  │ │  Start: 50 gold     │ │  Start: rations │   │
│  └─────────────────────┘ └─────────────────────┘ └─────────────────┘   │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ So far:  Noble birth (+1 INT)  >  Veteran's child (+2 STR +1 END)│  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                    [ < Back ] [ Next >] │
└─────────────────────────────────────────────────────────────────────────┘
```

### Components

- **Full-screen CentralPanel** (no side panel).
- **Header bar:** Step counter (`Step 2 of 6`), question title, Back button.
- **Flavor text frame:** Narrative context for the question.
- **Option cards:** 3–4 selectable radio-group cards showing choice name, stat bonuses, and starting equipment/resources. Selected card highlighted with `(●)`.
- **Story-so-far bar:** Running summary of all prior selections, showing the accumulating build.
- **Nav buttons:** Back and Next at bottom-right.

### Step Sequence (Example)

1. Birth/origin (noble, commoner, exile)
2. Father's profession (soldier, merchant, farmer, scholar)
3. Childhood experience (trained with weapons, studied, worked fields, travelled)
4. Defining moment (battle, discovery, loss, betrayal)
5. Motivation (glory, revenge, duty, wealth)
6. Faction allegiance (Iron Pact, Ashen Vow, Freeholds, etc.)

---

## 7. Screen 3: Overworld Map

### Design Reference

Mount & Blade overworld — a spatial map with terrain, named locations, roaming parties, and the player marker moving along paths between locations.

### Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ┌──────────────┐ ┌────────────────────────────────────────────────────────┐ │
│ │ [ < Guild  ] │ │  /\ /\    ^^^  . . . ^^^  /\    ≈≈≈≈≈             ≈≈ │ │
│ │ [ Details  ] │ │ /  \/  \  ^^^^ . . .^^^^  /  \  ≈≈≈≈≈≈≈          ≈≈≈ │ │
│ │              │ │/    \   \  ^^....[Ironpeak]..  \ ≈≈≈≈≈≈≈≈        ≈≈≈≈ │ │
│ │──────────────│ │ ~~mountains~~  . / . . . . .\   ≈≈river≈≈       ≈≈≈≈ │ │
│ │ Travel: IDLE │ │  . . . . . .  ./ . . . . . .\   ≈≈≈≈≈≈  . . . .≈≈≈ │ │
│ │              │ │ . . . . . . . / . . . . . . . \   ≈≈≈ . . . . . .≈≈ │ │
│ │──────────────│ │─ ─ ─ ─ ─ ─ ─/ ─ ─ ─ BORDER ─ ─\ ─ ─ ─ ─ ─ ─ ─ ─ ─│ │
│ │ FACTION CTRL │ │ . . . .[Millhaven]. . . . . . . \. .[Thornwall] . . │ │
│ │ Iron Pact: 4 │ │ . .♣♣. . .@ . . . . .♣♣♣. . . . \ . . . . . . . . │ │
│ │ Ashen Vow: 3 │ │ .♣♣♣♣♣. . .\ . . . .♣♣♣♣♣. . . . \ . . . . . . . │ │
│ │ Freeholds: 2 │ │ . .♣♣. . . . \ . . . .♣♣. . . . . .\ . . . . . . .│ │
│ │              │ │ . . . . . . . .\ . . . . . . . . . . .\ . . . . . . │ │
│ │──────────────│ │ . . . . . . . . \___road_____[Ashford] . \ . . . . . │ │
│ │ YOUR PARTY   │ │ . . . . . . . . . . . . . . . . . . . . .\ . . . . │ │
│ │ Aldric  ████ │ │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ BORDER ─ ─ ─ ─ ─ ─ \ ─ ─ ─ ─│ │
│ │ Serra   ███░ │ │ ≈≈≈ . . . . . . . . . . . . . . . .[Greymoor] . . .│ │
│ │ Brokk   ██░░ │ │ ≈≈≈≈ . . .♣♣♣♣♣. . . . . . .♣♣. . . . . . . . . .│ │
│ │              │ │ ≈≈≈≈≈ . . .♣♣♣♣♣♣♣. . . . .♣♣♣♣♣. . . . . . . . . │ │
│ └──────────────┘ └────────────────────────────────────────────────────────┘ │
│  [Millhaven]  Iron Pact ◆  Pop: 340  Garrison: 12       [ Enter Region ]  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Map Specifications

- **Size:** 150x80+ characters, pannable and zoomable.
- **Terrain generation:** Procedural heightmap + biome → ASCII glyph mapping (see Section 14).
- **Faction territory:** Subtle background tint (10–15% opacity) per faction zone. Neutral-colored border line (`─ │ ╭ ╮ ╰ ╯`) where zones meet.
- **Settlements:** Static location markers (`⌂` town, `■` castle, `▲` camp, `†` ruin), colored by owning faction.
- **Roaming parties:** Single `◆` glyph per party, colored by faction. Player is `@` (always bright green).
- **Party count:** 15–30+ roaming parties on screen at peak.
- **Travel:** Click destination, player `@` animates along a path. Side panel shows travel cooldown.

### Side Panel

- Navigation buttons (Guild, Details)
- Travel state and cooldown timer
- Faction control counts
- Party roster with HP bars

### Bottom Status Bar

Shows hovered entity info: settlement name, faction affiliation, population, garrison — or party name, faction, size, leader for roaming parties.

### Rendering Layers (bottom to top)

| Layer | Z | Content | Color |
|-------|---|---------|-------|
| L0 — Faction tint | Lowest | `painter.rect_filled()` per cell | Faction color @ 10–15% alpha |
| L1 — Terrain | Low | Glyph per cell from heightmap/biome | Muted, desaturated |
| L2 — Borders | Mid | Box-drawing chars at zone edges | Single neutral color |
| L3 — Settlements | High | Fixed position glyphs | Bright faction color |
| L4 — Parties | Highest | Moving glyphs at world positions | Bright faction color |
| L5 — UI overlays | Top | egui tooltips, path preview, selection | egui native |

### Party Collision Handling

When multiple party glyphs occupy the same or adjacent cells:
- Render a count digit (`2`, `3`, etc.) in the dominant faction's color.
- `*` for mixed-faction clusters.
- Hover expands into a list popup showing all parties in the cluster.

---

## 8. Screen 4: Region View

### Design Reference

Daggerfall — two modes sharing the same screen slot: a top-down local map for navigating the settlement, and fixed scene views when entering buildings or points of interest.

### Mode A: Local Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ┌──────────────┐ ┌──────────────────────────────────────────────────────┐   │
│ │[< Overworld ]│ │    ################       ###########                │   │
│ │              │ │    #   Tavern     #       # Market  #                │   │
│ │──────────────│ │    #    ⌂         +       #    ⌂    #                │   │
│ │ MILLHAVEN    │ │    #              #       #         +                │   │
│ │ Iron Pact    │ │    ########+#######       #####+#####                │   │
│ │              │ │            |                   |                     │   │
│ │──────────────│ │    ════════╪═══════════════════╪══════════           │   │
│ │ Garrison: 12 │ │            |    Main Street    |                     │   │
│ │ Pop: 340     │ │    ════════╪═══════════════════╪══════╤═══════       │   │
│ │ Mood: Calm   │ │            |                   |      |              │   │
│ │              │ │    ########+#######       ############+####          │   │
│ │──────────────│ │    # Smithy       #       # Garrison       #        │   │
│ │ QUEST LOG    │ │    #    ⌂         +       #    ⌂    @      #        │   │
│ │ · Clear rats │ │    #              #       #                #        │   │
│ │ · Deliver msg│ │    ################       ##################        │   │
│ │              │ │                                                      │   │
│ │──────────────│ │    @ = You   ⌂ = Entrance   + = Door   # = Wall    │   │
│ │ [Inventory ] │ │    ═ = Road  | = Path                               │   │
│ │ [Party     ] │ │                                                      │   │
│ └──────────────┘ └──────────────────────────────────────────────────────┘   │
│  Garrison — Iron Pact stronghold. A scarred captain eyes you from the gate.│
└─────────────────────────────────────────────────────────────────────────────┘
```

**Glyph vocabulary:** `#` wall, `+` door, `═` road, `|` path, `⌂` building entrance, `@` player.

**Navigation:** Click-to-move or arrow keys. Walking into a `⌂` door transitions to Scene View for that building.

### Mode B: Scene View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ┌──────────────┐ ┌──────────────────────────────────────────────────────┐   │
│ │[< Back      ]│ │                                                      │   │
│ │              │ │          THE RUSTY TANKARD  —  Tavern                 │   │
│ │──────────────│ │                                                      │   │
│ │ THE RUSTY    │ │       _____________________________                  │   │
│ │ TANKARD      │ │      |  _____   ___________   ___  |                 │   │
│ │              │ │      | |ALES | |   FIRE   | | ~ | |                 │   │
│ │ People here: │ │      | |_____| |__________| |___| |                 │   │
│ │ · Innkeeper  │ │      |  ┌──┐  ┌──┐  ┌──┐  ╭───╮  |                 │   │
│ │   Marta      │ │      |  │%%│  │%%│  │%%│  │BAR│  |                 │   │
│ │ · Hooded     │ │      |  └──┘  └──┘  └──┘  ╰───╯  |                 │   │
│ │   stranger   │ │      |_____________________________|                 │   │
│ │ · Merchant   │ │                                                      │   │
│ │   Tobren     │ │──────────────────────────────────────────────────────│   │
│ │              │ │  Marta polishes a glass and nods as you enter.       │   │
│ │──────────────│ │  A hooded figure sits alone in the corner.          │   │
│ │ Gold: 47     │ │                                                      │   │
│ │              │ │  > [Talk to Marta]  [Talk to stranger]  [Tobren]    │   │
│ │              │ │  > [Buy a drink — 2g]  [Rest — 5g]  [Leave]        │   │
│ └──────────────┘ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Central panel zones:** Top — ASCII art illustration of the interior. Middle — narrative text. Bottom — action buttons (talk, buy, rest, leave).

**Transitions:** Clicking an NPC opens dialogue (sub-panel or takes over the narrative zone). "Leave" returns to Local Map with `@` at the door the player entered.

---

## 9. Screen 5: Eagle Eye Intro

### Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ┌──────────────┐ ┌────────────────────────────────────────────────────────┐ │
│ │ SCOUTING     │ │                                                        │ │
│ │ Millhaven    │ │                                                        │ │
│ │              │ │                                                        │ │
│ │──────────────│ │              (dark / empty canvas)                     │ │
│ │ Your scout   │ │                                                        │ │
│ │ approaches   │ │                                                        │ │
│ │ the outskirts│ │                                                        │ │
│ │ of Millhaven.│ │                                                        │ │
│ │              │ │                                                        │ │
│ │ The garrison │ │                                                        │ │
│ │ looks thin   │ │                                                        │ │
│ │ today...     │ │                                                        │ │
│ │              │ │                                                        │ │
│ │ ▓▓▓▓▓▓▓░░░  │ │                                                        │ │
│ │ scouting...  │ │                                                        │ │
│ └──────────────┘ └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Behavior

- **Side panel:** Narrative text types out line-by-line with a progress bar.
- **Central panel:** Intentionally dark/empty — atmospheric, builds tension.
- **No player interaction required** — this is a narrative beat.
- **Auto-transitions** to Mission Execution when the progress bar fills.

---

## 10. Screen 6: Mission Execution

### Layout: Window Manager Workspace

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ ┌──────────────┐                                                                │
│ │ ACTIVE (3)   │  ┌─[ Millhaven Assault · Rm 2/4 ]─────────────────────┐       │
│ │              │  │ Tk 147  ▶  Morale: Steady     Obj: Eliminate        │       │
│ │ ▶ Millhaven  │  │────────────────────────────────────────────────────-│       │
│ │   Rm 2/4     │  │                                                     │       │
│ │   3v4 ██████ │  │          (combat grid — see Section 12)             │       │
│ │              │  │                                                     │       │
│ │   Thornwall  │  │────────────────────────────────────────────────────-│       │
│ │   Rm 3/4     │  │ ROSTER / EVENTS / HERO COMMANDS                     │       │
│ │   3v5 ████░░ │  └─────────────────────────────────────────────────────┘       │
│ │   ⚠ SHAKEN  │                                                                │
│ │              │  ┌─[ Thornwall ]────────┐  ┌─[ Road Ambush ]── MIN ──────────┐│
│ │   Road Ambush│  │ (smaller tiled view) │  │ Rd 1 ▶  4v3  Allies winning    ││
│ │   Rm 1/2     │  │                      │  └────────────────────────────────-┘│
│ │   4v3 ██████ │  └──────────────────────┘                                      │
│ │              │                                                                │
│ │──────────────│                                                                │
│ │ PARTY MORALE │                                                                │
│ │ ██████░░░░   │                                                                │
│ │ Steady       │                                                                │
│ │ [Retreat All]│                                                                │
│ └──────────────┘                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### OS-Style Window Manager

Each active battle is its own `egui::Window` — draggable, resizable, collapsible. The player arranges them freely:

- **Fullscreen one battle** — maximize a single window for focused tactical play.
- **Tile multiple** — arrange 2–3 windows side by side for simultaneous awareness.
- **Minimize** — collapse to a single status line showing round, unit count, and who's winning.

egui provides drag, resize, and collapse for free via `egui::Window::new().resizable(true).collapsible(true)`.

### Side Panel

- **Battle list:** All active battles with compact status — room progress, unit count ratio, HP summary bar, morale warning flags. Click to bring window to front.
- **Hero status:** HP, SP, current command (only for the battle containing the player's hero).
- **Party morale:** Global aggregate across all engagements.
- **Retreat All:** Emergency button to withdraw from all active missions.

### Multiple Simultaneous Missions

The multi-window setup is for viewing multiple *missions* (different overworld flashpoints or engagements) simultaneously, not multiple rooms within one mission. Each window shows one active room of one mission.

---

## 11. Screen 7: Replay Viewer

### Layout

Same window-based approach as Mission Execution, but reading from recorded frame data instead of live simulation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ┌──────────────┐  ┌─[ Millhaven Assault — Replay ]─────────────────────┐   │
│ │ REPLAYS      │  │ Frame 23/71     ▶ Playing       Speed: 2x          │   │
│ │              │  │────────────────────────────────────────────────────-│   │
│ │ ▶ Millhaven  │  │                                                     │   │
│ │   Thornwall  │  │          (combat grid from recorded frame)          │   │
│ │              │  │                                                     │   │
│ │──────────────│  │────────────────────────────────────────────────────-│   │
│ │ Frame: 23/71 │  │ ROSTER / EVENTS (frame 23)                          │   │
│ │ ▶ Playing    │  └─────────────────────────────────────────────────────┘   │
│ │ Speed: 2x    │                                                           │
│ │──────────────│                                                           │
│ │ [|< First  ] │                                                           │
│ │ [<  Prev   ] │                                                           │
│ │ [▶  Play   ] │                                                           │
│ │ [>  Next   ] │                                                           │
│ │ [>| Last   ] │                                                           │
│ │ Speed:       │                                                           │
│ │ [1x][2x][4x]│                                                           │
│ │ [< Back    ] │                                                           │
│ └──────────────┘                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Differences from Live Combat

- **Data source:** Recorded frames instead of live `step()` output.
- **VCR controls:** First, Prev, Play/Pause, Next, Last. Speed selector (1x, 2x, 4x).
- **Scrub bar:** Progress bar showing frame position, clickable to jump.
- **No hero commands:** View-only, no interaction with the grid.
- **Multiple replays** can be opened simultaneously as separate windows.

### Shared Rendering

The combat grid, roster, and event feed use identical rendering code to live combat — the only difference is the data source. This is a single reusable component (see Section 12).

---

## 12. Combat Window Component

The combat window is a reusable component instantiated inside an `egui::Window`. It renders identically for live combat, view-only observation, and replay — only the data source and command UI differ.

### Grid Rendering

```
┌─[ Mission Name — Room N/M: Type ]──────────────────────────────────────────┐
│ Tick NNN  ▶ Running   Obj: Eliminate   Party Morale: ██████░░░░ Steady     │
│────────────────────────────────────────────────────────────────────────────-│
│ . . . . . . . . . . ░░ . . . . . . . . . . . . . . . . . . . . . . . . . │
│ . . . . . . ░░ . . . ░░ . . . . . . . . . . . . . . . . . . . . . . . . │
│ . . . . . . ░░ . . . . . . . . . e1. . . . . . . . . . . . . . . . . . . │
│ . . . a1. . . . . . ██ . . . . . . . . . ██ . . . . . . . . . . . . . . │
│ . . . . . . . . . . ██ . . . . . . . . . ██ . . e2. . . . . . . . . . . │
│ . . ▲▲▲▲▲▲▲▲. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . │
│ .@H. ▲▲▲▲▲▲▲. . . . . . . . . . e3. . . . . . . . . . . . . . . . . . . │
│ . . . . . . . . . . . . . . ░░ . . . . . . . . . . . . . . . . . . . . . │
│ . . a2. . . . . . . . . . . ░░ . . . . . . . . . . . . . . . . . . . . . │
│ . . . . . . . . . . . . . . . . . . . . e4. . . . . . . . . . . . . . . │
│────────────────────────────────────────────────────────────────────────────-│
```

### Discretization

Continuous `SimVec2` positions are quantized to character cells. The grid scales to room size — small rooms produce tight grids, large rooms produce expansive ones. If the window is too small for the grid, it scrolls.

### Glyph Vocabulary — Environment

| Glyph | Meaning | Color |
|-------|---------|-------|
| `.` | Open floor | Dim, muted |
| `░░` | Half cover (damage reduction) | Slightly brighter than floor |
| `██` | Full cover (blocks LOS) | Solid, contrasting |
| `▲▲` | Elevated terrain (+damage, +range) | Distinct warm tone |

### Glyph Vocabulary — Units

| Glyph | Meaning | Color |
|-------|---------|-------|
| `@H` | Player's hero | Brightest, unique (green) |
| `a1`, `a2` | Allied units | Team color |
| `e1`, `e2` | Enemy units | Enemy faction color |

### Sub-Panels

Below the grid, the combat window has three sub-panels:

**Roster (left):**
```
ALLIES                              ENEMIES
@H Aldric  ████████░ Steady   CMD   e1 Raider  ████░░░░░ Aggressive
a1 Serra   ██████░░░ Steady         e2 Guard   ████████░ Defensive (cover)
a2 Brokk   █████████ Fired Up      e3 Brute   █████░░░░ Wavering  ⚠ wounded
                                    e4 Archer  ██████░░░ Routing   ⊘ fleeing
```

Shows: unit ID, name, HP bar, morale state, AI behavior label, status icons.

**Event Feed (right):**
```
e4 ROUTING — fleeing battlefield
e3 wounded — morale drops
a2 rallied by Aldric's presence
a1 casts Fireball → e1 (3t)
@H basic atk → e3: 24 dmg
```

Pulls directly from `Vec<SimEvent>` returned by `step()`, formatted for display. Morale events rendered prominently.

**Hero Command Panel (bottom-right, only when player's hero is present):**

```
HERO: Aldric (Warrior)          Morale: Steady
HP ████████░ 312/400  Shield: ░░ 40
Resource ██████░░░ 45/60

Abilities:          (scroll ▼ for more)
┌──────────────────────────────────────────────────┐
│ [1] Shield Bash    ●ready   Melee  CC            │
│ [2] Charge         ○ 4t cd  Dash   Gap-close     │
│ [3] Battle Cry     ●ready   Aura   Rally         │
│ [4] Whirlwind      ●ready   AoE    Damage        │
│ [5] Iron Will      ○12t cd  Self   Shield        │
│ [6] Taunt          ●ready   AoE    CC            │
│ ▼ 3 more abilities                                │
└──────────────────────────────────────────────────┘
Cmd: [Move] [Attack] [Hold] [Retreat]
```

### Ability List Design

Heroes can have 4 to 15+ abilities. The ability panel is a scrollable table:

| Column | Content |
|--------|---------|
| Hotkey | `[1]`, `[2]`, etc. |
| Name | Ability name |
| State | `●ready` or `○ Nt cd` (cooldown remaining) |
| Type | Delivery type from DSL (Projectile, AoE, Channel, Chain, Zone, Summon, etc.) |
| Tags | Functional tags (Damage, CC, Heal, Shield, Mobility, etc.) |

Optional `Filter: [All ▼]` dropdown to filter by type or tag when ability count is high.

Passives listed below the ability table as a simple text list.

### Hero Interaction

- **Click empty tile:** Dotted path (`·····`) previews the AI pathfinding route. Click to confirm, hero begins moving.
- **Click enemy:** Hover tooltip shows stats. Hero pathfinds into range and engages.
- **Click ability → click target:** Select ability from list, then click target location or unit on grid.
- **Hold command:** Cancels movement, hero defends in place.
- **Retreat:** Hero pathfinds toward map edge / exit.

All other units continue acting via AI regardless of player commands. The player's hero is just another unit with a player-issued command queue.

---

## 13. Morale & Culture System (UI)

### Morale Architecture

Morale operates at two levels:

- **Party morale baseline:** Global value reflecting overall engagement state.
- **Per-hero morale:** Individual modifier based on personal experience and cultural filters.

### What Morale Does

Morale does not directly affect stat bonuses. It changes AI behavior:

- **Fired Up:** Aggressive pushes, prioritizes offense.
- **Steady:** Normal behavior, balanced decisions.
- **Wavering:** Disengages from aggressive pushes, seeks cover, prioritizes self-preservation.
- **Routing:** Flees the battlefield. Ignores commands.

### Morale Triggers

Morale shifts are caused by observable events, not abstract stat changes:

- Wounded status (HP thresholds) → personal morale drops
- Ally routing nearby → nearby allies' morale drops
- Kill secured → personal and nearby morale boost
- Leader presence (aura) → nearby morale stabilized
- Outnumbered → party morale pressure
- Objective state → party morale modifier

### Culture as Morale Input Filter

Culture is not a flat modifier — it determines which inputs the morale calculation even considers. Each culture archetype defines weight multipliers for five input categories:

| Culture | Self | Allies | Threats | Leadership | Situation |
|---------|------|--------|---------|------------|-----------|
| Collectivist | Normal | HIGH | Normal | HIGH | Normal |
| Individualist | Normal | MINIMAL | HIGH | LOW | Normal |
| Fanatical | LOW | Normal | LOW | EXTREME | HIGH |
| Mercenary | HIGH | LOW | HIGH | LOW | Normal |
| Disciplined | Normal | Normal | Normal | Normal | HIGH |

**Example:** An Individualist hero barely registers ally state — their morale is almost exclusively driven by personal threat and their own condition. A Collectivist hero's morale is heavily influenced by the state of nearby allies.

### Morale Breakdown Panel (UI)

When inspecting a hero's morale, the breakdown shows all inputs with their weights. Inputs that the hero's culture filters out are displayed grayed out (`░░░ FILTERED`), teaching the player the system without a tutorial:

```
┌─ Kael (Duelist) ────────────────────────────────┐
│ Culture: Freehold — Individualist                │
│ Morale: Steady                                   │
│                                                  │
│ SELF                                             │
│  + 5  healthy (HP > 60%)                         │
│  + 3  recent kill                                │
│  - 2  outnumbered locally                        │
│                                                  │
│ ALLIES            (weight: MINIMAL)              │
│  ░░░  ally Brokk routing nearby         FILTERED │
│  ░░░  ally Serra wounded                FILTERED │
│                                                  │
│ THREATS           (weight: HIGH)                 │
│  -10  elite enemy in engagement range            │
│  + 4  no direct threat targeting self            │
│                                                  │
│ NET: -1           → Steady                       │
└──────────────────────────────────────────────────┘
```

### Cascade Dynamics

Morale cascades create emergent tactical situations:

- Hero wounded → personal morale drops → starts routing → nearby Collectivist allies shaken → party morale dips → more cautious play across the board.
- Conversely: hero scores a clutch kill → rallies nearby allies → party morale surges → aggressive push.
- Mixed-culture parties create resilience: Individualist fighters keep pushing when Collectivists waver, buying time for a rally.

### Party Composition Strategy

Culture composition becomes a strategic decision:

- **Full Collectivist:** Swings hard both ways — a single rout can cascade, but a clutch moment rallies everyone.
- **Full Individualist:** Stable but isolated — no cascading routs, but no rallying off each other either.
- **Mixed:** Resilience through diversity. Individualists anchor while Collectivists amplify good moments.
- **Fanatical:** Nearly immune to personal threat, but if their leader falls, catastrophic collapse.

---

## 14. Overworld Rendering Pipeline

### Overview

The overworld is a 150x80+ character ASCII landscape, procedurally generated, with faction territories, settlements, and roaming parties layered on top. Rendering happens in a fixed layer order each frame.

### Generation Pipeline

```
Heightmap + Biome Noise
        │
        ▼
  Cell → Glyph Mapping
        │
        ▼
  Post-Processing
  (ridge tracing, forest clumping, river flow)
        │
        ▼
  Faction Territory Assignment
  (Voronoi regions → per-cell faction ownership)
        │
        ▼
  Border Computation
  (detect cells where faction changes → border glyphs)
        │
        ▼
  Settlement Placement
  (fixed positions within regions)
```

### Terrain Glyph Mapping

| Biome/Feature | Glyph | Color | Notes |
|---------------|-------|-------|-------|
| Plains | `.` | Dim green-gray | Most common, low visual noise |
| Forest | `♣` | Dark green | Clumps into connected groves |
| Dense forest | `♣♣♣` | Darker green | Larger contiguous areas |
| Hills | `~` | Tan | Gentle elevation |
| Mountains | `/\ ^^^` | Brown | Connected ridgelines, not scattered |
| River | `≈` | Steel blue | Connected paths flowing downhill |
| Road | `═` `─` | Light gray | Connects settlements |
| Coast/shore | `,` | Sandy | Transition between land and water |

### Post-Processing Requirements

Raw noise-to-character mapping produces visual static. Post-processing passes are essential:

- **Mountain ridges:** Trace connected ridgelines so mountains form `/\ /\` ranges, not scattered `^` characters.
- **Forest clumping:** Grow forests as contiguous blobs of `♣`, not random scatter.
- **River continuity:** Ensure rivers are connected paths of `≈` flowing downhill from source to coast/lake.
- **Road connectivity:** Roads follow paths between settlements, not random placement.

### Faction Territory Rendering

- **Background tint:** Each cell gets a `painter.rect_filled()` call with the owning faction's color at 10–15% alpha. Drawn first, terrain glyphs render on top. The tint is what the eye reads for "who controls this area."
- **Border line:** Neutral-colored box-drawing characters (`─ │ ╭ ╮ ╰ ╯`) drawn where the faction tint changes. ~50–60% opacity. Marks the edge without adding visual noise.
- **Contested zones:** Could show as a blended or flickering tint, or a slightly different neutral wash to signal instability.

### Performance Considerations

At 150x80, you're painting 12,000+ individually colored glyphs per frame. Optimization strategies:

- **Glyph atlas batching:** Build a texture atlas of all ASCII glyphs, emit one textured quad per cell into a single `Mesh`, rather than calling `painter.text()` per cell.
- **Dirty region tracking:** Only re-render cells that changed since last frame (party movements, faction tint updates).
- **Zoom-level LOD:** At full zoom-out, reduce detail — simplify terrain, use density heatmap for party clusters. At zoom-in, full detail rendering.

---

## 15. Open Questions & Follow-Ups

### Planned Follow-Up: egui Code Architecture

The next design phase should cover:

- **Screen state machine:** Enum-based screen state with transitions, shared app state.
- **Shared layout skeleton:** Reusable SidePanel + CentralPanel pattern.
- **Combat window component:** The reusable struct that takes a data source (live sim or replay frames) and renders the grid + roster + feed + hero commands.
- **Window manager:** How `egui::Window` instances are created, tracked, and laid out for multi-combat.
- **Input routing:** Keyboard/mouse input dispatch when multiple windows are present.

### Planned Follow-Up: Overworld ASCII Rendering

The next design phase should cover:

- **Terrain generation algorithm:** Specific noise functions, biome assignment, heightmap parameters.
- **Post-processing passes:** Algorithms for ridge tracing, forest clumping, river flow.
- **Glyph atlas implementation:** Texture atlas layout, batched mesh construction, cache invalidation.
- **Pan/zoom implementation:** Camera state, viewport culling, smooth scrolling.
- **Party movement interpolation:** Smooth `@` marker animation between cells as the party travels.

### Other Open Questions

- **Dialogue system:** When talking to NPCs in Scene View, does dialogue take over the central panel, or is it a sub-panel? How are dialogue trees structured?
- **Inventory/equipment UI:** What does the inventory screen look like? Separate screen or overlay?
- **Combat grid scaling:** How does the grid handle extreme room size variance (15x10 skirmish vs 40x30 siege) within the same window?
- **Save/load UI:** Settings and save management screens.
- **Sound design:** Audio cues for morale events, combat actions, UI navigation (even ASCII games benefit enormously from sound).
- **Accessibility:** Colorblind modes for faction differentiation, screen reader considerations for the ASCII grid.

---

---

## 16. Implementation Constraint: Zero Native egui Chrome

**CRITICAL CONSTRAINT — applies to all screens and all future implementation work.**

The final application must have **zero visible native egui widgets**. Everything the player sees must be ASCII text rendered via `LayoutJob` or `ui.painter()`. egui is used only as a rendering backend — its widget library is not part of the visual output.

### What must be removed (replace with ASCII equivalents)

| Native egui | ASCII replacement |
|-------------|-------------------|
| `egui::Button` | `[ label ]` as colored `LayoutJob` text, click detected via `ui.allocate_response()` or `Response` on the label |
| `egui::Frame` | Box-drawing borders `╔═╗║╚═╝` or `┌─┐│└─┘` rendered as text |
| `egui::SidePanel` / `egui::CentralPanel` | Single full-screen `CentralPanel`, regions split manually via `LayoutJob` column math or `ui.allocate_rect()` |
| `egui::Window` | ASCII-bordered floating panes drawn via painter, draggable via manual hit-testing |
| `egui::ProgressBar` | `████░░░░` text bars via `LayoutJob` |
| `egui::Separator` | `────────────` text lines |
| `egui::ScrollArea` | Manual viewport offset with arrow key / scroll wheel input |
| `egui::TopBottomPanel` | Part of the single `CentralPanel` text layout |
| `egui::RichText` labels (styled) | `LayoutJob` with explicit `TextFormat` per character |

### End state

Each screen is a single `egui::CentralPanel` (used only for its `ui` handle). All content inside is rendered as:
- `egui::text::LayoutJob` — per-character colored monospace text, one job per row
- `ui.painter()` — for background fills (faction tint rectangles, panel backgrounds)

No egui chrome, frames, buttons, separators, scroll areas, or window decorations are visible. The application looks like a terminal emulator, not a GUI app.

### Implementation order

1. First: get all screens rendering correct content (current phase)
2. Then: replace egui widgets with ASCII equivalents screen by screen
3. Finally: collapse all panels into a single CentralPanel with manual region splitting

This is a phased migration — don't rip out egui widgets until the ASCII replacement is working.

---

*Document generated March 2026. Consolidates UI design decisions from iterative wireframing sessions covering all seven game screens, the combat window component, morale/culture system visualization, and overworld rendering architecture.*
