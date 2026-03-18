# LoL Champion Imports

The project includes 172 League of Legends champions converted to the game's
hero template format. These serve as a massive test corpus for the ability
system and provide diverse training scenarios for the AI.

## Source Data

Champion data was sourced from `assets/lol_champions/` as JSON files containing
official champion statistics and ability descriptions. A conversion pipeline
transforms these into the project's TOML + ability DSL format.

## Directory: `assets/lol_heroes/`

Contains 345 files (172 champions × 2 files each):

```
lol_heroes/
├── aatrox.toml
├── aatrox.ability
├── ahri.toml
├── ahri.ability
├── akali.toml
├── akali.ability
└── ... (166 more champions)
```

## Conversion Pipeline

The conversion process (`docs/OLD/LOL_CONVERSION_PLAN.md`):

1. **Parse JSON** — extract champion stats and ability data
2. **Map stats** — convert LoL stats to the game's stat model
3. **Translate abilities** — convert ability descriptions to DSL syntax
4. **Handle LoL mechanics** — map LoL-specific systems:
   - Ammo systems → `charges`
   - Toggle abilities → `is_toggle`
   - Transform champions → `form` / `swap_form`
   - Multi-cast ultimates → `recast_count`

## Coverage

The LoL imports test the full breadth of the ability system:

| Feature | Count | Examples |
|---------|-------|---------|
| Projectile delivery | 80+ | Ezreal Q, Morgana Q |
| AoE abilities | 100+ | Annie R, Zyra R |
| Chain/bounce | 10+ | Ryze E, Fiddlesticks E |
| Zone abilities | 30+ | Singed Q, Morgana W |
| Toggle abilities | 8 | Singed Q, Anivia R |
| Form swap | 6 | Nidalee, Jayce, Elise |
| Charge system | 12 | Teemo R, Akali E |
| Recast | 15 | Ahri R, Riven Q |
| Unstoppable | 5 | Malphite R, Vi R |

## Purpose in Training

LoL champions provide:

1. **Ability diversity** — tests whether the AI can handle 800+ unique abilities
2. **Balance testing** — ensures the simulation doesn't break with extreme stat
   combinations
3. **Training coverage** — the AI encounters a huge variety of ability interactions
4. **Stress testing** — edge cases in the effect system (chains of chains, zones
   inside zones, etc.)

## Known Limitations

Some LoL mechanics don't map perfectly:
- **Skill shots** — approximated as targeted projectiles
- **Terrain creation** — simplified to zone effects
- **Stealth** — implemented but not all interactions preserved
- **Passive scaling** — flat approximations of scaling passives
