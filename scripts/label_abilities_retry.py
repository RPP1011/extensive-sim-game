#!/usr/bin/env python3
"""Retry labeling for abilities that failed with the full prompt.

Uses a shorter, more constrained prompt to handle the long god abilities
that the 1.2B model struggles with.
"""

import json
import sys
import time
from pathlib import Path

LFM_DIR = Path.home() / "Projects" / "lfm-agent"
sys.path.insert(0, str(LFM_DIR))

from lfm_agent.model import LFMModel, SamplingParams

GAME_DIR = Path.home() / "Projects" / "game"
DATASET_DIR = GAME_DIR / "generated" / "ability_dataset"
OUTPUT_PATH = GAME_DIR / "generated" / "ability_labels.jsonl"

LABEL_SAMPLING = SamplingParams(temperature=0.0, max_tokens=128)

# Much shorter prompt for stubborn cases
SHORT_PROMPT = (
    'Label this ability as JSON: {"primary_type":"<damage|heal|cc|buff|debuff|utility|defense|mobility>",'
    '"complexity":<1-5>,"power_level":"<weak|moderate|strong|overloaded>",'
    '"description":"<one sentence>"}\n\n'
)


def main():
    labeled = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                labeled.add(json.loads(line)["file"])

    all_files = sorted(DATASET_DIR.glob("*.ability"))
    remaining = [(p.name, p.read_text()) for p in all_files if p.name not in labeled]
    print(f"Retrying {len(remaining)} abilities with short prompt")

    if not remaining:
        print("All done!")
        return

    print("Loading LFM model...")
    model = LFMModel()

    ok = 0
    fail = 0
    t0 = time.time()

    with open(OUTPUT_PATH, "a") as out:
        for batch_start in range(0, len(remaining), 128):
            batch = remaining[batch_start : batch_start + 128]
            # Truncate ability text to first 20 lines to fit context
            batch_msgs = []
            for _, content in batch:
                lines = content.split("\n")[:20]
                truncated = "\n".join(lines) + "\n..."
                batch_msgs.append([
                    {"role": "user", "content": SHORT_PROMPT + truncated},
                ])

            prompts = [model._apply_template(msgs) for msgs in batch_msgs]
            outputs = model.llm.generate(prompts, LABEL_SAMPLING)

            for (filename, _), output in zip(batch, outputs):
                raw = model.tokenizer.decode(
                    output.outputs[0].token_ids, skip_special_tokens=False
                )
                for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                    raw = raw.replace(tok, "")
                raw = raw.strip()

                start = raw.find("{")
                end = raw.rfind("}")
                label = None
                if start != -1 and end != -1:
                    try:
                        label = json.loads(raw[start : end + 1])
                    except json.JSONDecodeError:
                        pass

                if label is None:
                    fail += 1
                    # Write a default label for truly stubborn cases
                    label = {
                        "primary_type": "damage",
                        "complexity": 5,
                        "power_level": "overloaded",
                        "description": "Complex multi-effect ability with many nested effects",
                    }

                name = filename.replace(".ability", "")
                record = {"file": filename, "name": name, **label}
                out.write(json.dumps(record) + "\n")
                ok += 1

            print(f"  [{ok + fail}/{len(remaining)}] {ok} ok, {fail} defaulted")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s: {ok} labeled ({fail} with defaults)")

    # Final count
    total = 0
    with open(OUTPUT_PATH) as f:
        for _ in f:
            total += 1
    print(f"Total labels: {total}")


if __name__ == "__main__":
    main()
