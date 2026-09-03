#!/bin/bash
# Lesson 3: prefill (TTFT, compute-bound) vs decode (ITL, memory-bound).
# Prompt length must be the only variable, or the TTFT split stops working.
MODEL=Qwen/Qwen2.5-7B-Instruct
for LEN in 128 512 2048; do
  for R in 1 2 3; do
    vllm bench serve --model $MODEL --dataset-name random \
      --random-input-len $LEN --random-output-len 32 \
      --num-prompts 20 --max-concurrency 1
  done
done
# ms per 1k = (TTFT@2048 - TTFT@512) / 1.536
# Differencing cancels the fixed per-request overhead.
