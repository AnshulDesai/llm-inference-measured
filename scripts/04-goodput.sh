#!/bin/bash
# Lesson 6: open-loop arrivals. NO --max-concurrency -- that is the whole point.
# A closed loop caps waiting at the pinned in-flight count, so p99 cannot diverge.
MODEL=Qwen/Qwen2.5-7B-Instruct
for R in 4 6 8 10 12; do
  vllm bench serve --model $MODEL --dataset-name random \
    --random-input-len 1024 --random-output-len 128 \
    --num-prompts 200 --request-rate $R \
    --percentile-metrics ttft,itl,e2el --metric-percentiles 50,99 \
    --save-result --result-filename rate_$R.json
done
# Goodput = highest rate holding p99 TTFT < 1000 ms AND p99 ITL < bound.
# Check vllm:num_preemptions_total: nonzero at your goodput rate means back off.
