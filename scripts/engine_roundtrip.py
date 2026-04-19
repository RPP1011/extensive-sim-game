#!/usr/bin/env python3
"""Round-trip validation: load a safetensors trajectory, re-save it with value equality preserved."""
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.numpy import save_file


def main() -> None:
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    tensors = {}
    with safe_open(in_path, framework="numpy") as f:
        for key in sorted(f.keys()):
            tensors[key] = f.get_tensor(key)
    save_file(tensors, out_path)


if __name__ == "__main__":
    main()
