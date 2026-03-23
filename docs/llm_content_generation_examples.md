# LLM Content Generation — Example Inputs

Source: campaign trace `campaign_2088.trace.json`, tick 5100

---

## Example 1: Ability Generation for Player Character

### System Prompt

```
You generate ability definitions in a custom DSL. Output ONLY the ability block. No thinking. No explanation. No markdown.

## Grammar

ability <snake_case_name> {
    type: active|passive
    cooldown: <N>s                    (required for active, omit for passive)
    trigger: <trigger_type>           (required for passive, omit for active)
    effect: <effect_description>
    tag: <single_tag>
    description: "<flavor text>"
}

## Valid stats (ONLY these exist)
attack, defense, speed, max_hp, ability_power

## Valid tags (pick ONE)
ranged, nature, stealth, tracking, survival,
melee, defense, leadership, fortification, honor,
arcane, elemental, ritual, knowledge, enchantment,
healing, divine, protection, purification, restoration,
assassination, agility, deception, sabotage,
crisis, legendary, inspiration, sacrifice

## Valid triggers (passive only)
on_damage_dealt, on_damage_taken, on_kill, on_ally_damaged,
on_death, on_ability_used, on_hp_below, on_hp_above,
on_shield_broken, periodic

## Effect types
damage <N>, heal <N>, shield <N>, stun <N>s, slow <factor> <N>s,
knockback <N>, dash, buff <stat> <N>% <N>s, debuff <stat> <N>% <N>s,
stealth <N>s, evasion <N>%, tenacity <N>, teleport, aura <stat> +<N>

## Examples

ability shield_bash {
    type: active
    cooldown: 10s
    effect: stun 2s + damage 30
    tag: melee
    description: "A devastating shield strike"
}

ability battle_sense {
    type: passive
    trigger: on_damage_taken
    effect: buff defense 15% 5s
    tag: defense
    description: "Pain sharpens focus"
}

ability shadow_step {
    type: active
    cooldown: 15s
    effect: teleport + damage 40 + stealth 3s
    tag: stealth
    description: "Vanish and reappear behind your prey"
}
```

### User Prompt

```
Generate an ability for this adventurer:

Name: Player
Archetype: knight (level 6)
Stats: HP 115, ATK 25, DEF 20, SPD 10, AP 8
Condition: stress 43, fatigue 100, injury 100, morale 89, loyalty 100
Traits: battle_scarred
Fame: 3796 (quests: 34, victories: 34)
Status: Traveling

Tags available: melee, defense, leadership, fortification, honor

Quest record: 37 victories, 0 defeats
Quest types: 12 Combat, 7 Gather, 6 Rescue, 6 Exploration, 5 Escort, 4 Diplomatic

Guild: 16 adventurers (0 injured, 0 in combat)
Resources: 4568 gold, 335 supplies, 100 reputation
Active quests: 3, Completed: 40
Roster: Player (L6 knight), Brother Aldric (L6 cleric), Sera Bright (L6 knight), Alaric (L6 ranger), Brynn (L6 mage), Cira (L6 cleric), Daven (L6 rogue), Apprentice (L5 rogue), Sera the Strategist (L7 knight), Vorn the Smith (L6 knight)...

Regions:
  Coral Haven — contested (owner: Coral Compact, control: 34, unrest: 23)
  Driftwood Shoals — stable (owner: Coral Compact, control: 64, unrest: 5)
  The Drowned Reach — contested (owner: Coral Compact, control: 60, unrest: 30)
  Stormbreak Isle — stable (owner: Abyssal Cult, control: 70, unrest: 20)
  The Abyssal Shelf — falling (owner: Abyssal Cult, control: 0, unrest: 100)
Factions:
  Coral Compact — Friendly, relation: 100, military: 42, coalition: False
  Tideborne League — Friendly, relation: 92, military: 74, coalition: False
  Abyssal Cult — AtWar, relation: -69, military: 124, coalition: False
Active crises:
  Sleeping King — 0/7 champions arrived
  Corruption — 1/5 regions affected
  Unifier — 1 factions absorbed
  Dungeon Breach — wave 0, strength 15
Campaign progress: 91%
Global threat: 47

Player origin: wandering_mercenary
Backstory: The Last Contract → The Crossroads → Word Gets Out

Trigger: Level 10 milestone reached + 30 victories in combat. This battle-scarred noble has fought through exile and restored their honor.

The ability should reflect this adventurer's specific experiences, personality, and role in the guild. It should feel earned, not generic.
```

### Expected Output Format

```
ability <snake_case_name> {
    type: active|passive
    cooldown: <N>s
    effect: <effect_description>
    tag: <single_tag>
    description: "<flavor text>"
}
```

---

## Example 2: Class Generation for Brother Aldric

### System Prompt

```
You generate class definitions in a custom DSL. Output ONLY the class block. No thinking. No explanation. No markdown.

## Grammar

class <PascalCaseName> {
    stat_growth: +<N> <stat>, +<N> <stat>, ... per level
    tags: <tag>, <tag>, ...
    scaling <source> {
        when <condition>: <bonus>
        always: <bonus>
    }
    abilities {
        level <N>: <snake_case_name> "<description>"
    }
    requirements: <req>, <req>, ...
    consolidates_at: <N>              (optional, for prestige classes)
}

## Valid stats (ONLY these, no others)
attack, defense, speed, max_hp, ability_power

## Stat growth rules
- Total per level should be 5-15 for normal classes, up to 25 for hero classes
- Use "+N all" as shorthand for equal growth in all stats

## Valid tags
ranged, nature, stealth, tracking, survival,
melee, defense, leadership, fortification, honor,
arcane, elemental, ritual, knowledge, enchantment,
healing, divine, protection, purification, restoration,
assassination, agility, deception, sabotage,
crisis, legendary, inspiration, sacrifice

## Valid scaling sources
party_alive_count, party_size, faction_strength, coalition_strength,
crisis_severity, fame, territory_control, adventurer_count,
gold, reputation, threat_level

## Valid conditions in "when" clauses
party_members > N, party_members >= N, faction_alive,
faction_territory >= N, crisis_active, crisis_severity > N,
solo (party_members == 1)

## Valid bonuses
+N% <stat>          (percentage stat boost)
+N <stat>           (flat stat boost)
tenacity <N>        (CC reduction, 0-1)
escape <N>          (disengage chance)
aura <stat> +<N>    (buff nearby allies)
last_stand below <N>% max_hp +<N>% attack
inspire nearby +<N> <stat>

## Valid requirements
level <N>, fame <N>, quests <N>, trait <name>,
active_crisis, gold <N>, group_size <N>, allies <N>

## Examples

class Sentinel {
    stat_growth: +1 attack, +3 defense, +3 max_hp per level
    tags: melee, defense, leadership
    scaling party_alive_count {
        when party_members > 0: +10% defense
        when party_members >= 3: tenacity 0.5
        always: aura defense +2
    }
    abilities {
        level 1: shield_wall "Reduces incoming damage to party"
        level 5: taunt "Forces enemies to target this unit"
        level 10: iron_will "Immune to morale effects"
    }
    requirements: level 5, fame 50
}

class Shadowmaster {
    stat_growth: +3 attack, +3 speed, +1 ability_power per level
    tags: stealth, assassination, agility
    scaling party_alive_count {
        when party_members == 1: +25% attack
        always: +5% speed
    }
    abilities {
        level 1: ambush "Bonus damage on first strike"
        level 5: evasion "Chance to dodge attacks"
        level 10: assassinate "Attempt to one-shot a target"
    }
    requirements: level 10, fame 100
}
```

### User Prompt

```
Generate a class specialization for this adventurer:

Name: Brother Aldric
Archetype: cleric (level 6)
Stats: HP 95, ATK 17, DEF 19, SPD 8, AP 22
Condition: stress 42, fatigue 100, injury 100, morale 85, loyalty 100
Fame: 3792 (quests: 34, victories: 34)
Status: Traveling

Tags available: healing, divine, protection, purification, restoration

Quest record: 37 victories, 0 defeats
Quest types: 12 Combat, 7 Gather, 6 Rescue, 6 Exploration, 5 Escort, 4 Diplomatic

Guild: 16 adventurers (0 injured, 0 in combat)
Resources: 4568 gold, 335 supplies, 100 reputation
Active quests: 3, Completed: 40
Roster: Player (L6 knight), Brother Aldric (L6 cleric), Sera Bright (L6 knight), Alaric (L6 ranger), Brynn (L6 mage), Cira (L6 cleric), Daven (L6 rogue), Apprentice (L5 rogue), Sera the Strategist (L7 knight), Vorn the Smith (L6 knight)...

Regions:
  Coral Haven — contested (owner: Coral Compact, control: 34, unrest: 23)
  Driftwood Shoals — stable (owner: Coral Compact, control: 64, unrest: 5)
  The Drowned Reach — contested (owner: Coral Compact, control: 60, unrest: 30)
  Stormbreak Isle — stable (owner: Abyssal Cult, control: 70, unrest: 20)
  The Abyssal Shelf — falling (owner: Abyssal Cult, control: 0, unrest: 100)
Factions:
  Coral Compact — Friendly, relation: 100, military: 42, coalition: False
  Tideborne League — Friendly, relation: 92, military: 74, coalition: False
  Abyssal Cult — AtWar, relation: -69, military: 124, coalition: False
Active crises:
  Sleeping King — 0/7 champions arrived
  Corruption — 1/5 regions affected
  Unifier — 1 factions absorbed
  Dungeon Breach — wave 0, strength 15
Campaign progress: 91%
Global threat: 47

Player origin: wandering_mercenary
Backstory: The Last Contract → The Crossroads → Word Gets Out

Trigger: Fame threshold 200 reached. Brother Aldric has proven themselves through 34 quests and earned recognition.

The class should represent a natural evolution of how this adventurer has been playing. The scaling source should match their role (party-based for squad leaders, faction-based for champions, crisis-based for heroes). Abilities should feel like a culmination of their journey.
```

### Expected Output Format

```
class <PascalCaseName> {
    stat_growth: +<N> <stat>, +<N> <stat>, ... per level
    tags: <tag>, <tag>, ...
    scaling <source> {
        when <condition>: <bonus>
        always: <bonus>
    }
    abilities {
        level <N>: <ability_name> "<description>"
    }
    requirements: level <N>, fame <N>
}
```
