# Objectives

Each mission room can have objectives beyond "kill all enemies." The objective
system adds strategic variety to combat encounters.

## Module: `src/mission/objectives.rs`

## Objective Types

| Objective | Win Condition |
|-----------|--------------|
| **Eliminate** | Kill all enemies |
| **Survive** | Keep at least one hero alive for N ticks |
| **Protect** | Keep a specific NPC alive |
| **Capture** | Move a hero to a specific position and hold |
| **Escape** | Move all heroes to the exit zone |
| **Boss** | Defeat the boss unit (ignore minions) |

## Objective Interaction with AI

Objectives affect AI behavior through the squad blackboard. For example:
- **Protect** objectives make the squad AI prioritize defending the NPC
- **Capture** objectives assign one unit to the capture point
- **Escape** objectives bias the formation toward the exit

Enemy AI also reacts to objectives — enemies in a Protect mission will
target the VIP; enemies in an Escape mission will block the exit.
