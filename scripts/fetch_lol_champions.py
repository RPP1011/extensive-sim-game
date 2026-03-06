#!/usr/bin/env python3
"""
Fetch all League of Legends champion ability data from:
1. Riot Data Dragon (structured JSON: descriptions, cooldowns, costs, ranges)
2. LoL Wiki templates (detailed: damage values, scaling ratios, mechanics)

Saves one JSON file per champion in assets/lol_champions/
"""

import json
import re
import sys
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Semaphore

OUT_DIR = Path(__file__).parent.parent / "assets" / "lol_champions"
WIKI_API = "https://leagueoflegends.fandom.com/api.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (LoL-Ability-Fetcher/1.0)"}

# Concurrency controls
DDRAGON_WORKERS = 20
WIKI_WORKERS = 8
wiki_semaphore = Semaphore(WIKI_WORKERS)


def fetch_url(url: str, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(1)


def fetch_json(url: str):
    return json.loads(fetch_url(url))


def clean_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    for old, new in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">")]:
        text = text.replace(old, new)
    return text.strip()


def extract_spell(spell: dict) -> dict:
    return {
        "id": spell["id"],
        "name": spell["name"],
        "description": clean_html(spell["description"]),
        "tooltip": clean_html(spell["tooltip"]),
        "cooldown": spell["cooldownBurn"],
        "cost": spell["costBurn"],
        "cost_type": spell.get("costType", ""),
        "range": spell["rangeBurn"],
        "max_rank": spell["maxrank"],
        "effect_burn": spell.get("effectBurn"),
        "leveltip": spell.get("leveltip", {}),
        "resource": clean_html(spell.get("resource", "")),
    }


def extract_passive(passive: dict) -> dict:
    return {
        "name": passive["name"],
        "description": clean_html(passive["description"]),
    }


def parse_ability_wikitext(text: str) -> dict:
    result = {}
    var_defs = re.findall(r'\{\{#vardefine:(\w+)\|([^}]*)\}\}', text)
    if var_defs:
        result["variables"] = {k: v for k, v in var_defs}
    param_pattern = re.compile(r'\|(\w[\w\s]*\w|\w)\s*=\s*(.*?)(?=\n\||\n\}\}|\Z)', re.DOTALL)
    for match in param_pattern.finditer(text):
        key = match.group(1).strip()
        value = match.group(2).strip()
        if value and key not in ("1", "2", "3", "4", "5"):
            result[key] = value
    comments = re.findall(r'<!--(.+?)-->', text)
    if comments:
        result["_comments"] = [c.strip() for c in comments if c.strip()]
    return result


def fetch_wiki_ability(champ_name: str, ability_name: str) -> dict | None:
    page = f"Template:Data {champ_name}/{ability_name}"
    params = urllib.parse.urlencode({
        "action": "parse", "page": page, "prop": "wikitext", "format": "json",
    })
    try:
        with wiki_semaphore:
            data = fetch_json(f"{WIKI_API}?{params}")
        if "parse" not in data:
            return None
        return parse_ability_wikitext(data["parse"]["wikitext"]["*"])
    except Exception:
        return None


def fetch_ddragon_champion(version: str, champ_id: str) -> dict | None:
    """Phase 1: Fast parallel fetch from Data Dragon CDN."""
    try:
        url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion/{champ_id}.json"
        data = fetch_json(url)
        return data["data"][champ_id]
    except Exception as e:
        print(f"  ERROR (ddragon) {champ_id}: {e}", flush=True)
        return None


def enrich_with_wiki(champ_name: str, ability_names: dict) -> dict:
    """Phase 2: Fetch wiki detail for all abilities of one champion."""
    wiki_data = {}
    for slot, name in ability_names.items():
        data = fetch_wiki_ability(champ_name, name)
        if data:
            wiki_data[slot] = data
    return wiki_data


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching Data Dragon version...", flush=True)
    version = fetch_json("https://ddragon.leagueoflegends.com/api/versions.json")[0]
    print(f"  Version: {version}", flush=True)

    print("Fetching champion list...", flush=True)
    champions = fetch_json(
        f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
    )["data"]
    champ_ids = sorted(champions.keys())
    print(f"  Found {len(champ_ids)} champions", flush=True)

    # Phase 1: Parallel Data Dragon fetch (fast CDN)
    print("\nPhase 1: Fetching from Data Dragon...", flush=True)
    dd_data = {}
    with ThreadPoolExecutor(max_workers=DDRAGON_WORKERS) as pool:
        futures = {
            pool.submit(fetch_ddragon_champion, version, cid): cid
            for cid in champ_ids
        }
        for i, future in enumerate(as_completed(futures), 1):
            cid = futures[future]
            result = future.result()
            if result:
                dd_data[cid] = result
            if i % 20 == 0 or i == len(champ_ids):
                print(f"  {i}/{len(champ_ids)}", flush=True)

    print(f"  Got {len(dd_data)}/{len(champ_ids)} from Data Dragon", flush=True)

    # Build ability name map for wiki lookups
    slot_keys = ["Q", "W", "E", "R"]
    champ_ability_names = {}
    for cid, dd in dd_data.items():
        names = {"passive": dd["passive"]["name"]}
        for i, spell in enumerate(dd["spells"]):
            names[slot_keys[i]] = spell["name"]
        champ_ability_names[cid] = (dd["name"], names)

    # Phase 2: Parallel wiki enrichment
    print(f"\nPhase 2: Fetching wiki detail ({WIKI_WORKERS} workers)...", flush=True)
    wiki_results = {}
    with ThreadPoolExecutor(max_workers=WIKI_WORKERS) as pool:
        futures = {
            pool.submit(enrich_with_wiki, name, ability_names): cid
            for cid, (name, ability_names) in champ_ability_names.items()
        }
        for i, future in enumerate(as_completed(futures), 1):
            cid = futures[future]
            try:
                wiki_results[cid] = future.result()
            except Exception as e:
                print(f"  WARN wiki {cid}: {e}", flush=True)
                wiki_results[cid] = {}
            if i % 20 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)}", flush=True)

    # Phase 3: Assemble and save
    print("\nPhase 3: Saving...", flush=True)
    saved = 0
    for cid, dd in dd_data.items():
        abilities = {}
        abilities["passive"] = extract_passive(dd["passive"])
        for i, spell in enumerate(dd["spells"]):
            abilities[slot_keys[i]] = extract_spell(spell)

        # Merge wiki detail
        wiki = wiki_results.get(cid, {})
        for slot, wiki_ability in wiki.items():
            if slot in abilities:
                abilities[slot]["wiki_detail"] = wiki_ability

        result = {
            "id": cid,
            "name": dd["name"],
            "title": dd["title"],
            "resource_type": dd.get("partype", ""),
            "tags": dd.get("tags", []),
            "abilities": abilities,
        }

        out_file = OUT_DIR / f"{cid}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        saved += 1

    print(f"\nDone! Saved {saved} champions to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
