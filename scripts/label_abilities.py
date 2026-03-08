#!/usr/bin/env python3
"""Label 10k generated abilities using the LFM2.5-1.2B model via vLLM.

Reads .ability files from generated/ability_dataset/, sends them in batches
to the LFM model, and writes structured labels to generated/ability_labels.jsonl.

Usage:
    cd ~/Projects/lfm-agent
    uv run python ~/Projects/game/scripts/label_abilities.py

Labels per ability:
    - name: ability name
    - file: source filename
    - description: 1-2 sentence natural language summary
    - primary_type: damage | heal | cc | buff | debuff | utility | defense | mobility
    - sub_types: list of secondary types
    - complexity: 1-5 scale
    - key_effects: list of main effect names
    - power_level: weak | moderate | strong | overloaded
"""

import json
import os
import sys
import time
from pathlib import Path

# Add lfm-agent to path so we can import the model
LFM_DIR = Path.home() / "Projects" / "lfm-agent"
sys.path.insert(0, str(LFM_DIR))

from lfm_agent.model import LFMModel, SamplingParams

GAME_DIR = Path.home() / "Projects" / "game"
DATASET_DIR = GAME_DIR / "generated" / "ability_dataset"
OUTPUT_PATH = GAME_DIR / "generated" / "ability_labels.jsonl"

BATCH_SIZE = 256  # vLLM handles large batches efficiently

LABEL_SAMPLING = SamplingParams(
    temperature=0.0,  # deterministic for labeling
    max_tokens=256,
)

SYSTEM_PROMPT = """\
You label game abilities. Given an ability definition, output a JSON object with these fields:
- "description": 1-2 sentence summary of what the ability does
- "primary_type": one of: damage, heal, cc, buff, debuff, utility, defense, mobility
- "sub_types": array of secondary types from the same list (can be empty)
- "complexity": integer 1-5 (1=simple single effect, 5=many nested effects)
- "key_effects": array of the main effect names used (e.g. ["damage","stun","slow"])
- "power_level": one of: weak, moderate, strong, overloaded

Output ONLY the JSON object, no other text."""


def load_abilities() -> list[tuple[str, str]]:
    """Load all .ability files, return list of (filename, content)."""
    files = sorted(DATASET_DIR.glob("*.ability"))
    result = []
    for f in files:
        content = f.read_text()
        result.append((f.name, content))
    return result


def make_messages(ability_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ability_text},
    ]


def parse_label(raw: str, filename: str) -> dict | None:
    """Try to extract JSON from model output."""
    # Strip special tokens
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        raw = raw.replace(tok, "")
    raw = raw.strip()

    # Find JSON object boundaries
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def main():
    abilities = load_abilities()
    total = len(abilities)
    print(f"Loaded {total} abilities from {DATASET_DIR}")

    if total == 0:
        print("No abilities found. Run generate_ability_dataset first.")
        sys.exit(1)

    # Check for existing progress
    done = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                obj = json.loads(line)
                done.add(obj.get("file", ""))
        print(f"Resuming: {len(done)} already labeled")

    remaining = [(fn, content) for fn, content in abilities if fn not in done]
    print(f"To label: {len(remaining)}")

    if not remaining:
        print("All done!")
        return

    print("Loading LFM model...")
    model = LFMModel()
    # Override sampling params for labeling
    print(f"Model loaded. Labeling in batches of {BATCH_SIZE}...")

    labeled = 0
    failed = 0
    t0 = time.time()

    with open(OUTPUT_PATH, "a") as out:
        for batch_start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[batch_start : batch_start + BATCH_SIZE]
            batch_msgs = [make_messages(content) for _, content in batch]

            # Use the model's tokenizer and LLM directly for custom sampling
            prompts = [model._apply_template(msgs) for msgs in batch_msgs]
            outputs = model.llm.generate(prompts, LABEL_SAMPLING)

            for (filename, content), output in zip(batch, outputs):
                raw = model.tokenizer.decode(
                    output.outputs[0].token_ids, skip_special_tokens=False
                )
                label = parse_label(raw, filename)

                if label is None:
                    failed += 1
                    continue

                # Extract ability name from content
                name = filename.replace(".ability", "")
                record = {
                    "file": filename,
                    "name": name,
                    **label,
                }
                out.write(json.dumps(record) + "\n")
                labeled += 1

            elapsed = time.time() - t0
            total_done = labeled + failed
            rate = total_done / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - total_done) / rate if rate > 0 else 0
            print(
                f"  [{total_done}/{len(remaining)}] "
                f"{labeled} labeled, {failed} failed | "
                f"{rate:.0f} abilities/s | ETA {eta:.0f}s"
            )

    elapsed = time.time() - t0
    print(f"\nDone: {labeled} labeled, {failed} failed in {elapsed:.1f}s")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
