#!/bin/bash
# Lesson 8: two independent dials. Name which one you turned or the claim is empty.
#   --quantization fp8      shrinks WEIGHT bytes    -> pays in ms of ITL
#   --kv-cache-dtype fp8    shrinks KV per token    -> pays in tokens in flight
# On sm89 the KV dial needs --attention-backend TRITON_ATTN (FA3 is sm90-only).
MODEL=Qwen/Qwen2.5-7B-Instruct
for B in 1 64; do
  vllm bench serve --model $MODEL --dataset-name random \
    --random-input-len 512 --random-output-len 200 \
    --num-prompts $((B*8)) --max-concurrency $B
done
# Fidelity is a DIFF WITH A FLOOR, never "quality seemed fine":
#   1. run the 16-bit server against ITSELF (temperature=0, fixed seed) -> the noise floor
#   2. then 16-bit vs fp8
# Compare PER TOKEN. Whole-completion equality over 200 tokens is too coarse to inform.
