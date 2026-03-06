#!/usr/bin/env python3
"""
Generate dungeon/map concept art using Gemini image models.

Usage example:
  GEMINI_API_KEY=... python3 scripts/gemini_mapgen.py \
    --prompt "Top-down fantasy dungeon map, guild outpost branch, 3 lanes, boss chamber"
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request


DEFAULT_MODEL = "gemini-3-pro-image-preview"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a map image via Gemini API."
    )
    parser.add_argument(
        "--prompt",
        help="Direct text prompt for map generation.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Path to a text file containing the prompt.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Gemini model name. "
            "Examples: gemini-3-pro-image-preview, gemini-2.5-flash-image"
        ),
    )
    parser.add_argument(
        "--out",
        default="generated/maps/map.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--save-text",
        action="store_true",
        help="Save any text response next to the image as a .txt file.",
    )
    return parser.parse_args()


def load_dotenv_if_present(path: pathlib.Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt and args.prompt_file:
        raise ValueError("Use either --prompt or --prompt-file, not both.")
    if args.prompt:
        return args.prompt.strip()
    if args.prompt_file:
        prompt_path = pathlib.Path(args.prompt_file)
        return prompt_path.read_text(encoding="utf-8").strip()
    raise ValueError("You must provide either --prompt or --prompt-file.")


def call_gemini(model: str, prompt: str, api_key: str) -> dict:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    body = json.dumps(payload).encode("utf-8")
    url = f"{API_ROOT}/{model}:generateContent"
    req = urllib.request.Request(
        url=url,
        method="POST",
        data=body,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc


def extract_parts(response_json: dict) -> tuple[str | None, bytes | None]:
    candidates = response_json.get("candidates", [])
    if not candidates:
        return None, None

    parts = candidates[0].get("content", {}).get("parts", [])
    text_chunks: list[str] = []
    image_bytes: bytes | None = None

    for part in parts:
        part_text = part.get("text")
        if part_text:
            text_chunks.append(part_text)
            continue

        inline_data = part.get("inlineData") or part.get("inline_data")
        if inline_data and inline_data.get("data"):
            image_bytes = base64.b64decode(inline_data["data"])

    text = "\n".join(chunk.strip() for chunk in text_chunks if chunk.strip()) or None
    return text, image_bytes


def ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_outputs(
    out_path: pathlib.Path, image_bytes: bytes, text: str | None, save_text: bool
) -> None:
    ensure_parent(out_path)
    out_path.write_bytes(image_bytes)
    print(f"Saved image: {out_path}")

    if save_text and text:
        text_path = out_path.with_suffix(out_path.suffix + ".txt")
        text_path.write_text(text + "\n", encoding="utf-8")
        print(f"Saved text:  {text_path}")


def main() -> int:
    args = parse_args()
    load_dotenv_if_present(pathlib.Path(".env"))
    try:
        prompt = load_prompt(args)
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Missing GEMINI_API_KEY environment variable.", file=sys.stderr)
        return 2

    print(
        f"[{dt.datetime.now().isoformat(timespec='seconds')}] "
        f"Generating map with model '{args.model}'..."
    )
    response_json = call_gemini(args.model, prompt, api_key)
    text, image_bytes = extract_parts(response_json)

    if not image_bytes:
        print("No image returned by model.", file=sys.stderr)
        if text:
            print("Model text response:", file=sys.stderr)
            print(text, file=sys.stderr)
        return 1

    write_outputs(pathlib.Path(args.out), image_bytes, text, args.save_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
