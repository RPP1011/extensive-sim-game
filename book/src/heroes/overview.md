# Hero Templates

Heroes are defined as pairs of files — a TOML file for stats and an `.ability`
file for abilities. The base game includes 27 hero archetypes, plus 172 imported
League of Legends champions.

## Directory Layout

```
assets/
├── hero_templates/     # 27 base archetypes
│   ├── warrior.toml    # stats
│   ├── warrior.ability # abilities
│   ├── mage.toml
│   ├── mage.ability
│   ├── ranger.toml
│   ├── ranger.ability
│   ├── necromancer.toml
│   ├── necromancer.ability
│   └── ... (23 more heroes)
│
└── lol_heroes/         # 172 LoL champion imports
    ├── aatrox.toml
    ├── aatrox.ability
    ├── ahri.toml
    ├── ahri.ability
    └── ... (170 more champions)
```

## Base Archetypes

The 27 base heroes span all tactical roles:

| Role | Heroes |
|------|--------|
| **Tank** | Warrior, Knight, Berserker |
| **Melee DPS** | Assassin, Monk, Duelist |
| **Ranged DPS** | Ranger, Arcanist, Elementalist |
| **Mage** | Mage, Cryomancer, Blood Mage |
| **Healer** | Cleric, Druid, Bard |
| **Support** | Paladin, Engineer, Alchemist |
| **Specialist** | Necromancer, Shadow, Warlock |

Each archetype has a distinct playstyle defined by its stat distribution and
ability kit.

## Loading Pipeline

```
hero_templates/warrior.toml  ──┐
                               ├──▶ HeroTemplate ──▶ UnitState
hero_templates/warrior.ability ─┘
```

1. TOML file is parsed for base stats (HP, armor, attack damage, etc.)
2. `.ability` file is parsed by the DSL parser for abilities and passives
3. Both are combined into a `HeroTemplate`
4. At mission start, templates are instantiated into `UnitState` on the field
