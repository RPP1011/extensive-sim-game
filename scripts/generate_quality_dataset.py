#!/usr/bin/env python3
"""Generate quality-filtered abilities + LFM descriptions at scale.

Generates abilities from grammar space, filters by quality,
sends to LFM with varied prompts for diverse descriptions.

Usage:
    uv run --with openai --with tqdm python scripts/generate_quality_dataset.py
"""

import json
import os
import subprocess
import sys
import concurrent.futures
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

VLLM_URL = "http://localhost:8000/v1"
MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"
OUTPUT = "dataset/ability_descriptions_v2.jsonl"

# Multiple prompt styles for diverse descriptions
PROMPTS = [
    # Technical
    """Describe this ability mechanically in 1 sentence. What does it do, who does it target, what's its cooldown?""",
    # RPG flavor
    """Write a dramatic 1-sentence description of this ability as it would appear in a fantasy RPG tooltip.""",
    # Keywords
    """List 5-8 search keywords for this ability (damage type, effect, targeting, element, class role).""",
    # Player guide
    """Explain this ability to a new player in 1 simple sentence.""",
    # Comparison
    """Describe this ability by comparing it to a common RPG/MOBA archetype (e.g. "like Fireball but with...", "a typical tank taunt that...").""",
]


def generate_descriptions(client, dsl_block, prompt_idx=None):
    """Generate descriptions using a specific or random prompt style."""
    if prompt_idx is None:
        import random
        prompt_idx = random.randint(0, len(PROMPTS) - 1)

    prompt = PROMPTS[prompt_idx]

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a game designer writing descriptions for tactical RPG abilities. Be concise."},
                {"role": "user", "content": f"{prompt}\n\n```\n{dsl_block}\n```"},
            ],
            temperature=0.9,
            max_tokens=150,
        )
        text = resp.choices[0].message.content.strip()
        # Split into lines, filter short ones
        descriptions = []
        for line in text.split('\n'):
            line = line.strip().lstrip('- •·').strip()
            if len(line) > 10 and not line.startswith('#'):
                descriptions.append(line)
        return descriptions[:3]  # max 3 per prompt
    except Exception as e:
        return []


def main():
    client = OpenAI(base_url=VLLM_URL, api_key="dummy")
    try:
        client.models.list()
        print("Connected to vLLM")
    except:
        print(f"Cannot connect to vLLM at {VLLM_URL}")
        sys.exit(1)

    # Load existing descriptions
    existing = []
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            existing = [json.loads(l) for l in f if l.strip()]
    print(f"Existing descriptions: {len(existing)}")

    existing_dsls = set()
    for e in existing:
        existing_dsls.add(e['dsl'][:50])

    # Find all ability blocks
    ability_blocks = []
    for root, dirs, files in os.walk("dataset/abilities"):
        for fname in sorted(files):
            if fname.endswith('.ability'):
                content = Path(os.path.join(root, fname)).read_text()
                blocks = split_blocks(content)
                for block in blocks:
                    ability_blocks.append((block, fname))

    # Filter to blocks without descriptions
    new_blocks = [(b, s) for b, s in ability_blocks if b[:50] not in existing_dsls]
    print(f"Total ability blocks: {len(ability_blocks)}")
    print(f"New blocks needing descriptions: {len(new_blocks)}")

    if not new_blocks:
        # Generate descriptions for existing blocks using different prompt styles
        print("\nAll blocks have descriptions. Generating additional varied descriptions...")
        import random
        random.seed(42)
        # Pick 5000 random blocks to get additional descriptions
        sample = random.sample(ability_blocks, min(5000, len(ability_blocks)))
        new_blocks = sample
        print(f"Generating additional descriptions for {len(new_blocks)} blocks")

    # Generate in parallel
    new_pairs = []

    def process(args):
        block, source = args
        results = []
        # Use multiple prompt styles per block
        for pidx in range(len(PROMPTS)):
            descs = generate_descriptions(client, block, pidx)
            for d in descs:
                results.append((d, block, source))
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(
            executor.map(process, new_blocks),
            total=len(new_blocks),
            desc="Generating descriptions",
        ))
        for result in results:
            for desc, block, source in result:
                new_pairs.append({
                    "description": desc,
                    "dsl": block,
                    "source": source,
                })

    print(f"New pairs generated: {len(new_pairs)}")

    # Merge and save
    all_pairs = existing + new_pairs
    with open(OUTPUT, 'w') as f:
        for p in all_pairs:
            f.write(json.dumps(p) + '\n')

    print(f"Total saved: {len(all_pairs)} pairs to {OUTPUT}")


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


if __name__ == "__main__":
    main()
