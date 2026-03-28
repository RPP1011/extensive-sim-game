#!/usr/bin/env python3
"""Generate diverse NL descriptions for abilities using LFM via vLLM.

Reads all .ability files, parses each ability block, sends it to LFM
with a prompt asking for 5 varied natural language descriptions.
Saves (description, ability_dsl) pairs as JSONL.

Usage:
    # Start vLLM first:
    uv run --with vllm --with liquid-ai-lfm vllm serve LiquidAI/LFM2.5-1.2B-Instruct --dtype float16 &

    # Then run:
    uv run --with openai --with tqdm python scripts/generate_descriptions.py
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

VLLM_URL = "http://localhost:8000/v1"
MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"
OUTPUT = "dataset/ability_descriptions.jsonl"

SYSTEM_PROMPT = """You are a game designer writing natural language descriptions for tactical RPG abilities.
Given an ability definition in DSL format, write 5 different natural language descriptions.
Each description should be 1-2 sentences, using different vocabulary and phrasing.
Include: what the ability does, its element/type, who it targets, and how powerful it feels.

Vary your descriptions:
- D1: Technical/mechanical (what it does precisely)
- D2: Flavor/RPG style (how it feels in fiction)
- D3: Short keywords (tags a player might search for)
- D4: Comparative (like "similar to X but with Y")
- D5: Casual/simple (explain to a new player)

Output ONLY the 5 descriptions, one per line, prefixed with D1: through D5:."""

def find_ability_files(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith('.ability'):
                files.append(os.path.join(dirpath, f))
    return sorted(files)

def split_blocks(content):
    blocks = []
    current = []
    depth = 0
    in_block = False
    for line in content.split('\n'):
        stripped = line.strip()
        if not in_block and (stripped.startswith('ability ') or stripped.startswith('passive ')):
            in_block = True
            current = []
        if in_block:
            current.append(line)
            depth += stripped.count('{') - stripped.count('}')
            if depth <= 0 and '{' in '\n'.join(current):
                blocks.append('\n'.join(current).strip())
                current = []
                depth = 0
                in_block = False
    return blocks

def generate_descriptions(client, dsl_block):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": dsl_block},
            ],
            temperature=0.8,
            max_tokens=300,
        )
        text = resp.choices[0].message.content.strip()
        descriptions = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('D') and ':' in line[:4]:
                desc = line.split(':', 1)[1].strip()
                if len(desc) > 10:
                    descriptions.append(desc)
            elif len(line) > 10 and not line.startswith('#'):
                descriptions.append(line)
        return descriptions[:5]
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return []

def main():
    client = OpenAI(base_url=VLLM_URL, api_key="dummy")

    # Test connection
    try:
        client.models.list()
        print("Connected to vLLM")
    except Exception as e:
        print(f"Cannot connect to vLLM at {VLLM_URL}: {e}")
        print("Start it with: uv run --with vllm --with liquid-ai-lfm vllm serve LiquidAI/LFM2.5-1.2B-Instruct --dtype float16")
        sys.exit(1)

    files = find_ability_files("dataset/abilities")
    print(f"Found {len(files)} .ability files")

    # Collect all blocks first
    all_blocks = []
    for fpath in files:
        content = Path(fpath).read_text()
        blocks = split_blocks(content)
        for block in blocks:
            all_blocks.append((block, os.path.basename(fpath)))

    print(f"Total ability blocks: {len(all_blocks)}")

    # Process in batches using concurrent requests
    import concurrent.futures
    all_pairs = []
    batch_size = 32  # concurrent requests to vLLM

    def process_block(args):
        block, source = args
        descriptions = generate_descriptions(client, block)
        return [(desc, block, source) for desc in descriptions]

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = list(tqdm(
            executor.map(process_block, all_blocks),
            total=len(all_blocks),
            desc="Generating descriptions",
        ))
        for results in futures:
            for desc, block, source in results:
                all_pairs.append({
                    "description": desc,
                    "dsl": block,
                    "source": source,
                })

    # Save
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"\nSaved {len(all_pairs)} (description, DSL) pairs to {OUTPUT}")

    # Stats
    descs_per_ability = len(all_pairs) / max(1, sum(len(split_blocks(Path(f).read_text())) for f in files))
    print(f"Average descriptions per ability: {descs_per_ability:.1f}")

    # Sample
    print("\nSample descriptions:")
    for pair in all_pairs[:10]:
        print(f"  {pair['description'][:100]}")

if __name__ == "__main__":
    main()
