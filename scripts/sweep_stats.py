#!/usr/bin/env python3
"""Quick stats on a VAE sweep contexts file.

Usage:
    uv run --with pandas python3 scripts/sweep_stats.py [path]
    uv run --with pandas python3 scripts/sweep_stats.py generated/vae_v2/vae_contexts.jsonl
"""

import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "generated/vae/vae_contexts.jsonl"
df = pd.read_json(path, lines=True)

print(f"=== {path} ===")
print(f"Total contexts: {len(df):,}")
print(f"\nContent types:")
print(df.content_type.value_counts().to_string())

llm = df[df.content_type.isin(["ability", "class"])].copy()
llm["key"] = (
    llm.archetype + "_"
    + (llm.level // 5 * 5).astype(str) + "_"
    + llm.trigger + "_"
    + llm.content_type
)

print(f"\nCoarse keys: {llm.key.nunique()}")
for ct in ["ability", "class"]:
    sub = llm[llm.content_type == ct]
    print(f"  {ct}: {sub.key.nunique()} keys, {len(sub):,} contexts")

print(f"\nArchetypes ({llm.archetype.nunique()}):")
print(llm.archetype.value_counts().to_string())

print(f"\nLevel: min={llm.level.min()}, max={llm.level.max()}, mean={llm.level.mean():.1f}")
print(f"Unique (archetype, level): {llm.groupby(['archetype','level']).ngroups}")

# Dead dims
inputs = pd.DataFrame(df["input"].tolist())
stds = inputs.std()
dead = (stds < 0.001).sum()
print(f"\nInput dims: {inputs.shape[1]}, dead: {dead}")

# Trait activity
trait_active = (inputs.iloc[:, 23:39].sum() > 0).sum()
print(f"Trait dims active: {trait_active}/16")

# Dedup
unique_prompts = llm.context_text.nunique() if "context_text" in llm.columns else "?"
print(f"Unique prompts: {unique_prompts}")

# Time estimate
n = llm.key.nunique()
print(f"\nLLM estimate ({n} calls):")
for workers in [1, 2, 4, 8]:
    mins = n * 8 / workers / 60
    print(f"  {workers} workers: ~{mins:.0f} min")
