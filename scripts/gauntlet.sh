#!/usr/bin/env bash
# Generate 4v16 d3 scenarios for ALL hero compositions and run them.
set -euo pipefail

DIR="scenarios/gauntlet"
rm -rf "$DIR"
mkdir -p "$DIR"

HEROES=(
  alchemist assassin bard berserker blood_mage cleric cryomancer druid
  elementalist engineer knight mage monk necromancer paladin pyromancer
  ranger rogue samurai shadow_dancer shaman templar warden warlock
  warrior witch_doctor
)

# Convert snake_case to PascalCase
cap() {
  echo "$1" | sed -r 's/(^|_)([a-z])/\U\2/g'
}

idx=0
write_scenario() {
  local name="$1"; shift
  local templates="$*"
  local hero_count=$(echo "$templates" | tr ',' '\n' | wc -l)
  local tpl_toml=$(echo "$templates" | sed 's/,/", "/g; s/^/"/; s/$/"/')
  local seed=$((idx * 7919 + 42))
  local max_ticks=5000
  local f=$(printf "gauntlet_%04d.toml" $idx)
  cat > "$DIR/$f" <<TOML
[scenario]
name        = "$name"
seed        = $seed
hero_count  = $hero_count
enemy_count = 16
difficulty  = 3
max_ticks   = $max_ticks
room_type   = "Climax"
hero_templates = [$tpl_toml]

[assert]
outcome = "Either"
TOML
  idx=$((idx + 1))
}

# 1) All 26 heroes x4 (mono stacks)
for h in "${HEROES[@]}"; do
  H=$(cap "$h")
  write_scenario "4x ${H}" "${H},${H},${H},${H}"
done

# 2) Every hero x3 + Cleric
for h in "${HEROES[@]}"; do
  [[ "$h" == "cleric" ]] && continue
  H=$(cap "$h")
  write_scenario "3x ${H} + Cleric" "${H},${H},${H},Cleric"
done

# 3) Every hero x3 + Warrior
for h in "${HEROES[@]}"; do
  [[ "$h" == "warrior" ]] && continue
  H=$(cap "$h")
  write_scenario "3x ${H} + Warrior" "${H},${H},${H},Warrior"
done

# 4) Every hero x2 + Cleric + Warrior
for h in "${HEROES[@]}"; do
  [[ "$h" == "cleric" || "$h" == "warrior" ]] && continue
  H=$(cap "$h")
  write_scenario "2x ${H} + Cleric + Warrior" "${H},${H},Cleric,Warrior"
done

# 5) Every pair of distinct heroes x2+x2
pairs=(
  "warrior,ranger" "warrior,mage" "warrior,rogue" "warrior,paladin"
  "ranger,mage" "ranger,rogue" "ranger,cleric"
  "mage,cleric" "mage,rogue"
  "paladin,cleric" "paladin,mage" "paladin,ranger"
  "berserker,cleric" "berserker,ranger" "berserker,mage"
  "assassin,cleric" "assassin,ranger" "assassin,mage"
  "knight,cleric" "knight,ranger" "knight,mage"
  "samurai,cleric" "samurai,ranger" "samurai,mage"
  "monk,cleric" "monk,ranger" "monk,mage"
  "necromancer,cleric" "necromancer,warrior"
  "pyromancer,cleric" "pyromancer,warrior"
  "cryomancer,cleric" "cryomancer,warrior"
  "druid,cleric" "druid,warrior"
  "shaman,cleric" "shaman,warrior"
  "bard,cleric" "bard,warrior"
  "warlock,cleric" "warlock,warrior"
  "engineer,cleric" "engineer,warrior"
  "templar,cleric" "templar,warrior"
  "warden,cleric" "warden,warrior"
  "shadow_dancer,cleric" "shadow_dancer,warrior"
  "blood_mage,cleric" "blood_mage,warrior"
  "witch_doctor,cleric" "witch_doctor,warrior"
  "elementalist,cleric" "elementalist,warrior"
  "alchemist,cleric" "alchemist,warrior"
)
for pair in "${pairs[@]}"; do
  IFS=',' read -r a b <<< "$pair"
  A=$(cap "$a"); B=$(cap "$b")
  write_scenario "2x ${A} + 2x ${B}" "${A},${A},${B},${B}"
done

# 6) Strong mixed parties
write_scenario "Warrior+Ranger+Mage+Cleric" "Warrior,Ranger,Mage,Cleric"
write_scenario "Warrior+Ranger+Ranger+Cleric" "Warrior,Ranger,Ranger,Cleric"
write_scenario "Berserker+Ranger+Ranger+Cleric" "Berserker,Ranger,Ranger,Cleric"
write_scenario "Knight+Ranger+Mage+Cleric" "Knight,Ranger,Mage,Cleric"
write_scenario "Samurai+Ranger+Mage+Cleric" "Samurai,Ranger,Mage,Cleric"
write_scenario "Assassin+Ranger+Mage+Cleric" "Assassin,Ranger,Mage,Cleric"
write_scenario "Monk+Ranger+Mage+Cleric" "Monk,Ranger,Mage,Cleric"
write_scenario "Berserker+Warrior+Cleric+Mage" "Berserker,Warrior,Cleric,Mage"
write_scenario "Paladin+Warrior+Ranger+Cleric" "Paladin,Warrior,Ranger,Cleric"
write_scenario "Assassin+Rogue+Cleric+Warrior" "Assassin,Rogue,Cleric,Warrior"
write_scenario "Necromancer+Warlock+Cleric+Warrior" "Necromancer,Warlock,Cleric,Warrior"
write_scenario "Pyromancer+Cryomancer+Mage+Cleric" "Pyromancer,Cryomancer,Mage,Cleric"
write_scenario "Druid+Shaman+Cleric+Warrior" "Druid,Shaman,Cleric,Warrior"
write_scenario "ShadowDancer+Assassin+Cleric+Warrior" "ShadowDancer,Assassin,Cleric,Warrior"
write_scenario "Templar+Knight+Paladin+Cleric" "Templar,Knight,Paladin,Cleric"
write_scenario "Engineer+Alchemist+Cleric+Warrior" "Engineer,Alchemist,Cleric,Warrior"
write_scenario "Elementalist+Pyromancer+Cryomancer+Cleric" "Elementalist,Pyromancer,Cryomancer,Cleric"
write_scenario "Bard+Ranger+Ranger+Ranger" "Bard,Ranger,Ranger,Ranger"
write_scenario "WitchDoctor+Druid+Shaman+Cleric" "WitchDoctor,Druid,Shaman,Cleric"
write_scenario "BloodMage+Necromancer+Warlock+Cleric" "BloodMage,Necromancer,Warlock,Cleric"

echo "Generated $idx scenarios in $DIR"
