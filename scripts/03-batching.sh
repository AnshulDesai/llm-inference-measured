#!/bin/bash
# Lesson 5: the batching knee, and proving it moves with context length.
# The knee moving left as C grows is what names the pinned resource.
MODEL=Qwen/Qwen2.5-7B-Instruct
for C in 512 3840; do          # 3840 + 256 output = exactly max-model-len 4096
  for B in 1 16 64; do
    vllm bench serve --model $MODEL --dataset-name random \
      --random-input-len $C --random-output-len 256 \
      --num-prompts $((B*8)) --max-concurrency $B
  done
done
# Predicted knee: B* = 265,000 / C   (weight_bytes / kv_bytes_per_token / C)
