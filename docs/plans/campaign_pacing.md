# Campaign Pacing Overhaul

## Context
Campaign ticks are 100ms (inherited from combat sim). A BFS campaign of 9000 ticks = 15 minutes of game time. Classes reach level 7 out of 100. Victory happens too fast. None of this matches a 12-20 hour campaign experience.

## Goal
- 12-20 hours of real play per campaign
- ~700-1200 meaningful decisions (turns)
- Classes reach level 30-50 by endgame
- Level 100 is legendary (multi-campaign)
- BFS explores 2000-5000 turns deep

## Key Constant: CAMPAIGN_TURN_DURATION
```rust
/// One campaign turn = 60 seconds of in-game time.
/// This is the overworld decision rate — combat uses CAMPAIGN_TICK_MS (100ms).
pub const CAMPAIGN_TURN_SECS: u32 = 60;
```

## System Interval Changes
All `tick % N` intervals need rescaling. Currently tuned for 100ms ticks.
With 1 turn = 1 minute, intervals should be in "turns" not "ticks":

| System | Current (100ms ticks) | New (1-min turns) | Real-time equivalent |
|--------|----------------------|-------------------|---------------------|
| Class system | every 50 ticks (5s) | every 1 turn | every minute |
| Quest generation | every 100 ticks (10s) | every 5 turns | every 5 min |
| Economy | every 100 ticks | every 1 turn | every minute |
| Diplomacy | every 200 ticks | every 10 turns | every 10 min |
| Seasons | every 2000 ticks (3.3 min) | every 90 turns | every 1.5 hours |
| Weather | every 500 ticks | every 30 turns | every 30 min |
| Crisis | every 300 ticks | every 20 turns | every 20 min |
| Consolidation | every 500 ticks | every 60 turns | every hour |
| Reactive narrative | every 200 ticks | every 15 turns | every 15 min |

## XP Scaling
With class system ticking every turn and ~700 turns per campaign:
- XP per turn from behavior: ~10 (current with 50x multiplier)
- Total XP in full campaign: ~7000
- Level 30 cumulative (linear): 30 * base = reachable
- Level 50 cumulative (linear): 50 * base = stretch goal

**Use linear XP**: `level * base` instead of `level^2 * base`
- base = 20
- Level 10: cumulative 1,100 XP (110 turns)
- Level 30: cumulative 9,300 XP (930 turns = ~15 hours)
- Level 50: cumulative 25,500 XP (2550 turns = multi-campaign)
- Level 100: cumulative 101,000 XP (legendary)

## Victory Conditions
Current: complete N quests → win
Should be: multi-factor endgame requiring sustained development:
- Guild reputation above threshold
- Territory control percentage
- Threat clock managed (not at 1.0)
- At least one level 20+ class holder
- Multiple faction alliances
- Crisis arcs resolved

## BFS Configuration
- trajectory_max_ticks: 5000 (was 15000, but turns are now 60s each)
- root_sample_interval: 50 (was 300)
- initial_roots: 50
- ticks_per_branch: 20 (was 200, since each turn is more meaningful)

## Implementation Steps
1. [ ] Add CAMPAIGN_TURN_SECS constant
2. [ ] Rescale ALL system tick intervals in step.rs
3. [ ] Switch XP from quadratic to linear
4. [ ] Update BFS config defaults
5. [ ] Update victory conditions for longer campaigns
6. [ ] Adjust behavior credit rates for new timescale
7. [ ] Test with BFS — verify level 20+ reachable, 6+ hour campaigns
