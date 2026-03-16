#!/usr/bin/env python3
"""Batch-caption room images using Qwen3-VL via vLLM OpenAI-compatible API.

Usage:
    # Start vLLM server first (separate terminal):
    vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8200 --max-model-len 4096

    # Run captioning:
    uv run python training/caption_rooms.py \
        --input generated/rooms.jsonl \
        --images generated/room_images/ \
        --output generated/rooms_captioned.jsonl

    # Or with a custom VLM endpoint:
    uv run python training/caption_rooms.py --vlm-url http://localhost:8200
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx


CAPTION_PROMPT = """\
You are describing tactical room layouts for a squad-based combat game.
Given the top-down image and computed metrics, write a 1-3 sentence
description of the room's tactical properties. Focus on:
- Room shape and proportions
- Cover layout and density
- Chokepoints and corridors
- Elevation advantages
- Flanking routes
- Asymmetries between the two spawn sides (green=player, red=enemy)

Room type: {room_type}
Dimensions: {width}×{depth}
Metrics:
  Blocked: {blocked_pct:.0%}
  Chokepoints: {chokepoint_count}
  Cover density: {cover_density:.2f}
  Elevation zones: {elevation_zones}
  Flanking routes: {flanking_routes}
  Aspect ratio: {aspect_ratio:.2f}

Describe this room's tactical layout:"""


def encode_image(path: Path) -> str:
    """Read a PNG and return base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def caption_room(
    client: httpx.Client,
    vlm_url: str,
    model: str,
    image_b64: str,
    prompt: str,
    max_retries: int = 3,
) -> str:
    """Send image + prompt to VLM and return caption text."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    for attempt in range(max_retries):
        try:
            resp = client.post(
                f"{vlm_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 256,
                    "temperature": 0.3,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Failed after {max_retries} retries: {e}", file=sys.stderr)
                return ""


def main():
    parser = argparse.ArgumentParser(description="Caption room images using VLM")
    parser.add_argument("--input", default="generated/rooms.jsonl",
                        help="Input JSONL from roomgen export")
    parser.add_argument("--images", default="generated/room_images",
                        help="Directory containing room PNG images")
    parser.add_argument("--output", default="generated/rooms_captioned.jsonl",
                        help="Output JSONL with caption field added")
    parser.add_argument("--vlm-url", default="http://localhost:8200",
                        help="VLM server base URL (OpenAI-compatible)")
    parser.add_argument("--model", default="default",
                        help="Model name for the API (default: 'default')")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Concurrent requests (default: 1)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of rooms to caption (0=all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rooms that already have captions in output")
    args = parser.parse_args()

    input_path = Path(args.input)
    images_dir = Path(args.images)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    # Load existing captions if resuming
    existing_seeds = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("caption"):
                        existing_seeds.add(rec["seed"])
                except json.JSONDecodeError:
                    pass
        print(f"Resuming: {len(existing_seeds)} rooms already captioned", file=sys.stderr)

    # Check VLM health
    client = httpx.Client()
    try:
        health = client.get(f"{args.vlm_url}/health", timeout=5.0)
        if health.status_code != 200:
            print(f"VLM server not healthy at {args.vlm_url}", file=sys.stderr)
            sys.exit(1)
    except httpx.HTTPError:
        print(f"Cannot connect to VLM at {args.vlm_url}. Is vLLM running?", file=sys.stderr)
        sys.exit(1)

    # Process rooms
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    out_f = open(output_path, mode)
    count = 0
    skipped = 0
    t_start = time.time()

    with open(input_path) as f:
        for line_num, line in enumerate(f):
            if args.limit > 0 and count >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            seed = record["seed"]
            room_type = record["room_type"]

            if seed in existing_seeds:
                skipped += 1
                continue

            # Find image file
            image_path = images_dir / f"{room_type}_{seed}.png"
            if not image_path.exists():
                print(f"  Image not found: {image_path}", file=sys.stderr)
                continue

            # Build prompt
            metrics = record.get("metrics", {})
            prompt = CAPTION_PROMPT.format(
                room_type=room_type,
                width=record["width"],
                depth=record["depth"],
                blocked_pct=metrics.get("blocked_pct", 0),
                chokepoint_count=metrics.get("chokepoint_count", 0),
                cover_density=metrics.get("cover_density", 0),
                elevation_zones=metrics.get("elevation_zones", 0),
                flanking_routes=metrics.get("flanking_routes", 0),
                aspect_ratio=metrics.get("aspect_ratio", 1.0),
            )

            # Encode image and caption
            image_b64 = encode_image(image_path)
            caption = caption_room(client, args.vlm_url, args.model, image_b64, prompt)

            record["caption"] = caption
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            count += 1

            if count % 100 == 0:
                elapsed = time.time() - t_start
                rate = count / elapsed if elapsed > 0 else 0
                print(
                    f"  Captioned {count} rooms ({rate:.1f}/s, "
                    f"~{(args.limit or line_num) / rate / 60:.0f}m remaining)",
                    file=sys.stderr,
                )

    out_f.close()
    elapsed = time.time() - t_start
    print(
        f"Done: captioned {count} rooms in {elapsed:.0f}s "
        f"({count / elapsed:.1f}/s), skipped {skipped}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
