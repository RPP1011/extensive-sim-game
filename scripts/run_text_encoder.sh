#!/bin/bash
export LIBTORCH="/home/ricky/.cache/uv/environments-v2/bench-vllm-8a2c635b0debffbf/lib/python3.12/site-packages/torch"
export LIBTORCH_CXX11_ABI=1
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/12:$LIBRARY_PATH"
cargo run -p ability-vae --release --bin train-text-encoder "$@"
