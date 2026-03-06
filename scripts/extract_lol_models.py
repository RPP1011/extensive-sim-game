#!/usr/bin/env python3
"""Batch-convert extracted LoL champion assets to glTF using lol2gltf CLI.

Prerequisites:
  1. Install lol2gltf CLI: https://github.com/Crauzer/lol2gltf/releases
  2. Extract champion WAD files using Obsidian:
     - Open: League of Legends/Game/DATA/FINAL/Champions/<Champion>.wad.client
     - Extract: assets/characters/<champion>/skins/base/ -> extracted/<champion>/
     Expected structure per champion:
       extracted/<champion>/
         <champion>.skn
         <champion>.skl
         <champion>.dds  (or .png texture)
         animations/
           <champion>_idle1.anm
           <champion>_run.anm
           <champion>_attack1.anm
           <champion>_spell1.anm
           <champion>_spell2.anm
           <champion>_spell3.anm
           <champion>_spell4.anm
           <champion>_death1.anm
           ...

Usage:
  python3 scripts/extract_lol_models.py --input extracted/ --output assets/lol_models/
  python3 scripts/extract_lol_models.py --input extracted/ --output assets/lol_models/ --champion Ahri
"""

import argparse
import os
import subprocess
import sys
import json
from pathlib import Path


# Animation files we want to include (common across all champions).
# lol2gltf includes ALL .anm files from the -a folder, so we filter by
# copying only the ones we need to a temp dir, or we just include all.
# Including all is fine — Bevy loads clips by name and we only play the ones
# we need.

DESIRED_ANIMS = {
    "idle1", "idle2", "run", "attack1", "attack2",
    "spell1", "spell2", "spell3", "spell4",
    "death1", "dance", "recall", "recall_winddown",
    "channel", "channel_wndup",
}


def find_champion_dirs(input_dir: Path) -> list[tuple[str, Path]]:
    """Find all champion directories with .skn files."""
    results = []
    for entry in sorted(input_dir.iterdir()):
        if not entry.is_dir():
            continue
        # Look for .skn file
        skn_files = list(entry.glob("*.skn"))
        if skn_files:
            results.append((entry.name, entry))
    return results


def find_file(directory: Path, extension: str, prefer_base: bool = True) -> Path | None:
    """Find a file with the given extension, preferring 'base' skin files."""
    candidates = list(directory.glob(f"*.{extension}"))
    if not candidates:
        # Check skins/base/ subdirectory
        base_dir = directory / "skins" / "base"
        if base_dir.exists():
            candidates = list(base_dir.glob(f"*.{extension}"))
    if not candidates:
        return None
    if prefer_base and len(candidates) > 1:
        # Prefer files without skin number suffix
        base = [f for f in candidates if "_skin" not in f.stem.lower()]
        if base:
            return base[0]
    return candidates[0]


def find_anim_dir(champion_dir: Path) -> Path | None:
    """Find the animations directory for a champion."""
    # Common locations
    for subdir in ["animations", "anims", "skins/base/animations"]:
        d = champion_dir / subdir
        if d.exists() and any(d.glob("*.anm")):
            return d
    # Check for .anm files directly in champion dir
    if any(champion_dir.glob("*.anm")):
        return champion_dir
    return None


def find_texture(champion_dir: Path) -> Path | None:
    """Find the primary texture file."""
    for ext in ["png", "dds", "tga"]:
        tex = find_file(champion_dir, ext)
        if tex:
            return tex
    return None


def convert_champion(
    name: str,
    champion_dir: Path,
    output_dir: Path,
    lol2gltf_bin: str,
    dry_run: bool = False,
) -> bool:
    """Convert a single champion to .glb format."""
    skn = find_file(champion_dir, "skn")
    skl = find_file(champion_dir, "skl")
    if not skn or not skl:
        print(f"  SKIP {name}: missing .skn or .skl")
        return False

    output_path = output_dir / f"{name}.glb"

    cmd = [
        lol2gltf_bin, "skn2gltf",
        "-m", str(skn),
        "-s", str(skl),
        "-g", str(output_path),
    ]

    # Add animations if available
    anim_dir = find_anim_dir(champion_dir)
    if anim_dir:
        cmd.extend(["-a", str(anim_dir)])

    # Add texture if available
    tex = find_texture(champion_dir)
    if tex:
        # lol2gltf needs --materials and --textures to match
        # Use the material name from the skn (usually the champion name)
        material_name = skn.stem
        cmd.extend(["--materials", material_name, "--textures", str(tex)])

    if dry_run:
        print(f"  DRY RUN {name}: {' '.join(cmd)}")
        return True

    print(f"  Converting {name}...", end=" ", flush=True)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"FAIL: {result.stderr.strip()[:200]}")
            return False
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"OK ({size_mb:.1f} MB)")
        return True
    except subprocess.TimeoutExpired:
        print("FAIL: timeout")
        return False
    except FileNotFoundError:
        print(f"FAIL: lol2gltf binary not found at '{lol2gltf_bin}'")
        return False


def build_manifest(output_dir: Path):
    """Write a manifest of all converted models."""
    models = {}
    for glb in sorted(output_dir.glob("*.glb")):
        name = glb.stem
        size = glb.stat().st_size
        models[name] = {
            "file": glb.name,
            "size_bytes": size,
        }

    manifest_path = output_dir / "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"models": models, "count": len(models)}, f, indent=2)
    print(f"\nManifest written: {manifest_path} ({len(models)} models)")


def main():
    parser = argparse.ArgumentParser(description="Batch convert LoL champion models to glTF")
    parser.add_argument("--input", "-i", required=True, help="Directory with extracted champion folders")
    parser.add_argument("--output", "-o", default="assets/lol_models", help="Output directory for .glb files")
    parser.add_argument("--champion", "-c", help="Convert only this champion (case-insensitive)")
    parser.add_argument("--lol2gltf", default="lol2gltf", help="Path to lol2gltf binary")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--list", action="store_true", help="List found champions and exit")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: input directory '{input_dir}' does not exist")
        sys.exit(1)

    champions = find_champion_dirs(input_dir)
    if not champions:
        print(f"No champion directories found in '{input_dir}'")
        print("Expected: extracted/<ChampionName>/<champion>.skn")
        sys.exit(1)

    if args.list:
        print(f"Found {len(champions)} champions:")
        for name, path in champions:
            skn = find_file(path, "skn")
            skl = find_file(path, "skl")
            anim_dir = find_anim_dir(path)
            anim_count = len(list(anim_dir.glob("*.anm"))) if anim_dir else 0
            tex = find_texture(path)
            print(f"  {name:20s}  skn={'Y' if skn else 'N'}  skl={'Y' if skl else 'N'}  anims={anim_count:3d}  tex={'Y' if tex else 'N'}")
        return

    if args.champion:
        target = args.champion.lower()
        champions = [(n, p) for n, p in champions if n.lower() == target]
        if not champions:
            print(f"Champion '{args.champion}' not found in '{input_dir}'")
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(champions)} champion(s) -> {output_dir}/")
    print(f"Using lol2gltf: {args.lol2gltf}")
    print()

    success = 0
    fail = 0
    for name, path in champions:
        if convert_champion(name, path, output_dir, args.lol2gltf, args.dry_run):
            success += 1
        else:
            fail += 1

    print(f"\nDone: {success} converted, {fail} failed")

    if not args.dry_run and success > 0:
        build_manifest(output_dir)


if __name__ == "__main__":
    main()
