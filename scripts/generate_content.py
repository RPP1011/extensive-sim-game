#!/usr/bin/env python3
"""Generate game content (abilities, classes) using a local LLM.

Usage:
    uv run --with transformers --with torch --with accelerate python3 scripts/generate_content.py --type ability --context "level 10 stealth ranger"
    uv run --with transformers --with torch --with accelerate python3 scripts/generate_content.py --type class --context "level 15 knight, defended 3 regions"
    uv run --with transformers --with torch --with accelerate python3 scripts/generate_content.py --batch abilities.jsonl
"""

import argparse
import json
import sys
import time

MODEL_ID = "Qwen/Qwen3.5-4B"

ABILITY_SYSTEM = '''Output ONLY the ability block. No thinking. No explanation.

ability shield_bash {
    type: active
    cooldown: 10s
    effect: stun 2s + 30 damage
    tag: melee
    description: "A devastating shield strike"
}

ability divine_grace {
    type: active
    cooldown: 20s
    effect: heal 50 to all allies
    tag: healing
    description: "A prayer answered with golden light"
}

ability shadow_step {
    type: active
    cooldown: 15s
    effect: teleport behind target + 40 damage
    tag: stealth
    description: "Vanish and reappear behind your prey"
}'''

CLASS_SYSTEM = '''Output ONLY the class block. No thinking. No explanation.

class Sentinel {
    stat_growth: +1 attack, +3 defense, +3 max_hp per level
    tags: melee, defense, leadership
    scaling party_alive_count {
        when party_members > 0: +10% defense
        always: aura defense +2
    }
    abilities {
        level 1: shield_wall "Reduces incoming damage"
        level 5: taunt "Forces enemies to target this unit"
        level 10: iron_will "Immune to morale effects"
    }
    requirements: level 5, fame 50
}'''


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map='cuda')
    return tokenizer, model


def generate(tokenizer, model, system, user, prefix, max_tokens=250):
    import torch
    raw = f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{prefix}'
    ids = tokenizer(raw, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_tokens,
            temperature=0.7, do_sample=True, repetition_penalty=1.1
        )
    txt = prefix + tokenizer.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)

    # Find matching closing brace
    depth = 0
    for i, c in enumerate(txt):
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth <= 0:
                return txt[:i+1].strip()
    return txt.strip()


def generate_ability(tokenizer, model, context, tags=""):
    prompt = f"Write an ability for {context}."
    if tags:
        prompt += f" Tags: {tags}"
    return generate(tokenizer, model, ABILITY_SYSTEM, prompt, "ability ")


def generate_class(tokenizer, model, context, tags=""):
    prompt = f"Write a class for {context}."
    if tags:
        prompt += f" Tags: {tags}"
    return generate(tokenizer, model, CLASS_SYSTEM, prompt, "class ", max_tokens=350)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["ability", "class"], default="ability")
    parser.add_argument("--context", default="a level 5 ranger")
    parser.add_argument("--tags", default="")
    parser.add_argument("--batch", help="JSONL file with batch requests")
    parser.add_argument("--count", type=int, default=1, help="Number to generate")
    parser.add_argument("--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    print(f"Loading {MODEL_ID}...", file=sys.stderr)
    tokenizer, model = load_model()
    print("Ready.", file=sys.stderr)

    out = open(args.output, 'w') if args.output else sys.stdout

    if args.batch:
        with open(args.batch) as f:
            requests = [json.loads(line) for line in f if line.strip()]
        for i, req in enumerate(requests):
            t0 = time.time()
            gen_type = req.get("type", "ability")
            context = req.get("context", "a level 5 ranger")
            tags = req.get("tags", "")
            if gen_type == "ability":
                result = generate_ability(tokenizer, model, context, tags)
            else:
                result = generate_class(tokenizer, model, context, tags)
            elapsed = time.time() - t0
            print(json.dumps({"type": gen_type, "content": result, "context": context, "time": elapsed}), file=out)
            out.flush()
            print(f"  [{i+1}/{len(requests)}] {gen_type} in {elapsed:.1f}s", file=sys.stderr)
    else:
        for i in range(args.count):
            t0 = time.time()
            if args.type == "ability":
                result = generate_ability(tokenizer, model, args.context, args.tags)
            else:
                result = generate_class(tokenizer, model, args.context, args.tags)
            elapsed = time.time() - t0
            print(result)
            print()
            print(f"Generated in {elapsed:.1f}s", file=sys.stderr)

    if args.output:
        out.close()


if __name__ == "__main__":
    main()
