#!/bin/bash
# Lessons 1-2: decode byte floor, and pricing launch overhead by removing graph capture.
# Protocol used everywhere: 4 runs, discard run 1 as warmup, median of runs 2-4.
MODEL=Qwen/Qwen2.5-7B-Instruct
for i in 1 2 3 4; do
  vllm bench serve --model $MODEL --dataset-name random \
    --random-input-len 512 --random-output-len 256 \
    --num-prompts 20 --max-concurrency 1
done
# Then restart the server with --enforce-eager and repeat.
# NOTE: --enforce-eager is not a single-variable flag. It also disables
# torch.compile/inductor and swaps in hand-written kernels, so the delta is
# NOT attributable to launch overhead alone.
