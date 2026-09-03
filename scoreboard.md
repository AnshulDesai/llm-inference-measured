# Scoreboard

**Rig, frozen for all lessons:** 1× NVIDIA L40S (46,068 MiB = 44.99 GiB, 864 GB/s, sm89) on AWS `g6e.xlarge`, us-east-2.
**Model:** `Qwen/Qwen2.5-7B-Instruct`, BF16, `--max-model-len 4096 --gpu-memory-utilization 0.90 --no-enable-prefix-caching`.
**Server:** vLLM 0.27.1, torch 2.13.0.
**Protocol:** `vllm bench serve --dataset-name random --random-input-len 512 --random-output-len 256 --num-prompts 20 --max-concurrency 1`. Four runs, discard run 1 as warmup, median of runs 2–4.

**The constants everything divides by:**
```
weight bytes = 7,615,616,512 params × 2 bytes = 15.23 GB  (14.19 GiB)
byte floor   = 15.23 GB ÷ 864 GB/s            = 17.6 ms/step
KV per token = 2 × 28 layers × 4 KV heads × 128 head_dim × 2 bytes = 57,344 B = 56 KiB
```

---

| # | Quantity | Predicted | Measured | Ratio (MBU) | What explained the gap |
|---|---|---|---|---|---|
| 0 | weights on GPU (GiB) | 14.19 | 14.57 | 1.03 | measured value bundles non-torch overhead with weights |
| 0 | GPU KV cache size (tokens) | ~465,104 | 470,400 | 1.01 | self-consistent: 25.12 GiB ÷ 56 KiB = 470,377, block-aligned to 470,400 |
| 0 | median ITL @ conc 1 (ms) | — | 20.23 | — | instrument reading, no prediction yet |
| 1 | median ITL @ conc 1 (ms) | 17.6 | 20.23 | **0.87** | CLOSED — graph capture already applied by default; 2.63 ms residual is streaming efficiency + non-matmul tails, **not** host overhead |
| 2 | ITL, `--enforce-eager` (ms) | 23.64 | **20.57** | **0.856** | CLOSED — prediction missed 10×. Launch overhead is hidden behind GPU execution, not additive to wall-clock. Visible cost only 0.34 ms total (1 µs/launch). |

| 3 | prompt-phase ms per 1k tokens | 42.1 | **69.5** | **0.61** | efficiency haircut on the math clock — achieving 61% of peak dense BF16 FLOP/s |
| 3 | TTFT @ 2048 prompt (ms) | 130–260 (floor 86.2) | **153.4** | 1.78× floor | in predicted range |
| 3 | ITL flat across 128/512/2048 | flat | 20.26 / 20.24 / **20.40** | 0.8% spread | flat as predicted; the +0.16 ms at 2048 is the KV-cache read becoming barely visible (Lesson 4) |

| 4a | KV bytes per token | 57,344 (56 KiB) | 57,344 | 1.00 | derived from config.json: `2 × 28 layers × 4 kv_heads × 128 head_dim × 2 bytes` |
| 4b | KV token-slots in pool | ~453,000 | **470,400** | 1.04 | pass — within 15% |
| 4c | crossover, tokens in flight | **265,000** | — | — | *computed, not measured* — `15.24e9 ÷ 57,344` |
| 4d | tokens in flight @ 32 conc × 2048 | 65,536–69,632 | **69,013** | 1.00 | `kv_cache_usage_perc 0.1467 × 470,400` — gauge matches arithmetic |

| 5a | C=512, B=1 | 17.6 ms / 57 | 20.26 ms / 49 | 0.87 | matches Lesson 1 exactly |
| 5b | C=512, B=16 | 18.1 ms / 882 | 21.99 ms / 666 | 0.82 | 85% of ideal 16× scaling |
| 5c | C=512, B=64 | 19.8 ms / 3238 | 29.51 ms / **1703** | 0.67 | 64% of ideal 4× scaling from B=16 |
| 5d | C=3840, B=1 | 17.9 ms / 56 | 20.60 ms / 46 | 0.87 | — |
| 5e | C=3840, B=16 | 21.9 ms / 729 | 26.09 ms / 374 | 0.84 | 51% of ideal 16× |
| 5f | C=3840, B=64 | 35.0 ms / 1829 | 46.89 ms / **479** | 0.75 | **32% of ideal — past the knee** |
| 5g | knee position B* | 518 (C=512) / 69 (C=3840) | knee moved left ~7.5× | ✅ | **names the pinned resource: KV bandwidth, not math** |
| 5h | $/1M output tokens @ $1.86/hr | — | **$10.56 → $0.30** | 35× | batching, same GPU/hour/weights |

| 6a | rate 4 /s | occupancy 0.29 | p99 TTFT 318 / p99 ITL **139** | — | FAIL (ITL) |
| 6b | rate 6 /s | occupancy 0.43 | p99 TTFT 470 / p99 ITL **150** | — | FAIL (ITL) |
| 6c | rate 8 /s | occupancy 0.58 | p99 TTFT 628 / p99 ITL **165** | — | FAIL (ITL); best under a 200 ms bound |
| 6d | rate 10 /s | occupancy 0.72 | p99 TTFT **1298** / p99 ITL 208 | — | FAIL (both) |
| 6e | rate 12 /s | occupancy 0.86 | p99 TTFT **3135** / p99 ITL 234 | — | FAIL (both) |
| 6f | goodput @ p99 ITL<50ms, TTFT<1000ms | 6 /s | **0 /s** | — | bound unachievable at any rate on this config |
| 6g | goodput @ p99 ITL<200ms, TTFT<1000ms | — | **8 /s** | — | 885 tok/s, p99 TTFT 628 ms — the shippable number |

| 7a | concurrency: reserve-max vs paged @600 tok | 110 vs 755 | — | ~7× | **computed, not measured — vLLM has no unpaged mode** |
| 7b | TTFT @2048, cold → warm (ms) | ~153 → ~15 | **144.5 → 31.0** | **4.7×** | run-1 (524 ms) discarded as post-restart warmup, not cold |
| 7c | prefix cache hit rate, 3 identical runs | 66.1% | **66.1%** | 1.00 | exact: 4064 hits / 6144 queries |
| 7d | invalidation: alter first token | back to cold | **144.5 ms**, then 31.0 warm | ✅ | one token at position 0 kills all 128 chunks |

| 8a | ITL @ conc 1, 16-bit → fp8 (ms) | 11.43 (1.77×) | **20.26 → 14.08 (1.44×)** | 0.81 | 2.65 ms unaccounted = scale-apply cost + non-weight work |
| 8b | ITL @ conc 64, 16-bit → fp8 (ms) | win collapses to ~1.0-1.15× | **29.36 → 18.46 (1.59×)** | ✗ | **lesson claim failed** — B=64 is far below the C=512 knee of 518, so weights still dominate |
| 8c | KV pool, 16-bit → fp8 weights (tokens) | — | **437,200 → 564,592 (+29%)** | — | freed weight bytes flow into the pool; **fp8 KV-dtype pool still unmeasured** |
| 8d | % byte-identical, self-diff floor | 96-100% | **100.0%** | ✅ | seed+temperature properly pinned; clean floor |
| 8e | % byte-identical, fp8 vs 16-bit | 60-80% | **0.0%** | ✗ | **bad metric, my design error** — whole-200-token exact match; needs per-token agreement |
| 8f | $/1M output @ conc 64, $1.86/hr | — | **$0.32 → $0.22** | 1.48× | fp8 throughput 1,613 → 2,392 tok/s |

| 9a | sharding: ITL floor at TP=N | `17.6/N` + fixed per-layer sync tax → ~1.4-1.7× at N=2 | — | **COMPUTED, NOT MEASURED** — one card. Ceiling N=4 (must divide kv_heads=4) |
| 9b | spec decoding: accepted tokens/verify @ a=0.7, k=4 | 2.77 minus drafting cost | — | **COMPUTED, NOT MEASURED** — no second model served |
| 9c | sizing: 300 req/s, 2k in / 300 out, p99 TTFT <800ms | **~77 GPUs, $0.44/1M** | — | **COMPUTED** from rows 1-8 |

---

## Reasoning log

**Lesson 1.** Floor is 17.6 ms; measured 20.23 ms; MBU 0.87. The course expected ~0.57 with a ~13.4 ms gap, on the assumption that host launch overhead would still be present. It isn't — vLLM 0.27.1 enables CUDA graph capture by default. MBU 0.87 sits at or above the top of the 0.70–0.85 band that read+compute kernels typically achieve against copy-only peak bandwidth, so there is very little left to win on this axis. Per the triage rule (*near the floor → the model is the lever; far from the floor → the software is*), the remaining levers here are model-side: fewer bytes per token (Lesson 8) or more tokens per pass (Lesson 5).

**Lesson 2 prediction.** 28 layers × ~12 kernels ≈ 340 launches per token, at ~10 µs of host CPU time each ≈ 3.4 ms. That 3.4 ms **cannot fit inside the measured 2.63 ms gap**, which proves by arithmetic alone — no profiler needed — that launch overhead was already eliminated. Predicting eager ITL = 20.23 + 3.4 = 23.64 ms → MBU 0.74. Closing this row requires restarting with `--enforce-eager` and rerunning the identical benchmark, one variable changed.

**Lesson 2 result — the prediction missed by 10×, and the miss is the finding.**

Measured eager ITL 20.57 ms (runs 2-4: 20.57 / 20.57 / 20.57, zero spread), vs 23.64 ms predicted. Delta from graph-captured was **+0.34 ms, not +3.4 ms**. Backed out: 0.34 ms ÷ 340 launches = **1 µs of visible cost per launch**. MBU moved only 0.87 → 0.856.

The broken assumption was not the ~10 µs figure — it was treating launch cost as **additive to wall-clock latency**. It isn't:

```
WRONG:  added latency = launches × launch_cost
RIGHT:  added latency = Σ max(0, launch_cost − gpu_time_per_kernel)
```

The CPU pushes kernels onto a queue and the GPU pops from them **concurrently**. While the GPU executes kernel N, the CPU is already queuing N+1 and N+2, so launch cost hides behind execution time. Overhead becomes latency only when the queue *drains* — when the host cannot stay ahead.

On this rig it never drains: 20.57 ms ÷ ~340 kernels ≈ **60 µs of GPU work per kernel** against ~1-10 µs to launch one. The CPU is 6-60× faster per kernel and stays far ahead.

**Counter-intuitive consequence: a slower GPU hides launch overhead better.** The L40S's 864 GB/s makes each kernel slow enough to mask the host. An H100 at ~3,350 GB/s finishes kernels ~4× sooner, host slack shrinks, and launch overhead starts to bite. **Launch overhead is a fast-GPU problem** — which is why the course's reference machine measured +4 ms where this one measured +0.34 ms.

Also noted: eager did **not** return the 0.54 GiB of graph memory to the KV pool (25.11 GiB eager vs 25.12 GiB captured, 470,144 vs 470,400 tokens). Graph memory comes out of a different budget bucket.

**Interview line:** *"Launch overhead is only visible when the host can't keep the queue full. On an L40S with ~60 µs kernels it's hidden; on an H100 it isn't."*

**Measured side-finding — `nvidia-smi` utilization is not evidence.** Sampled 10× during eager decode: `utilization.gpu` and `utilization.memory` both read **100%** every single sample, while MBU was 0.856 and compute utilization was ~0.24%. The same 100% appeared at MBU 0.87 (captured) and 0.856 (eager). A metric identical across two different states cannot distinguish them. `utilization.gpu` reports only *whether at least one kernel was resident during the sampling window* — a presence check, not a productivity check; a memory-stalled kernel still counts. SM clocks held steady at 2040 MHz throughout, ruling out thermal throttling.

**⚠ Attribution caveat — `--enforce-eager` is not a single-variable flag (found Aug 25 by reading the engine config).** Comparing the two startup logs:

| | captured | `--enforce-eager` |
|---|---|---|
| `compilation mode` | `VLLM_COMPILE` (inductor) | **`NONE`** |
| `custom_ops` | `['none']` | **`['all']`** |
| `fuse_norm_quant` | False | **True** |
| `cudagraph_capture_sizes` | 51 buckets (1→512) | `[]` |

The flag disables `torch.compile`/inductor **and** switches to vLLM's hand-written CUDA kernels — so the eager run executed *different kernels*, not the same kernels with more launches. The claim "same kernels, same bytes, you deleted overhead not work" is false for this flag in vLLM 0.27.1.

**Consequence: the +0.34 ms delta cannot be attributed to launch overhead alone.** It is the net of (launch overhead added) + (inductor-compiled kernels swapped for hand-written ones), and those may partially cancel. The Lesson 2 *concepts* stand; the *attribution* does not. A clean single-variable test would need a way to disable capture while keeping inductor compilation.

**Graph capture buckets measured on this rig:** `[1, 2, 4, 8, 16, 24, ... 256 (step 8), 272 ... 512 (step 16)]` — 51 graphs, `max_cudagraph_capture_size=512`, `cudagraph_mode=FULL_AND_PIECEWISE`. Spacing is fine at small batch and coarse at large batch because padding waste is proportional: batch 5 rounds to the 8-graph (37% wasted lanes) while batch 100 rounds to 104 (4%).

**Lesson 3 — the two clocks, measured.** Sweep at input 128 / 512 / 2048, output pinned at 32, median of 3 runs each.

| prompt | TTFT (ms) | ITL (ms) | compute floor | which floor binds |
|---|---|---|---|---|
| 128 | 32.92 | 20.26 | 5.4 | bytes → 17.6 ms |
| 512 | 46.74 | 20.24 | 21.6 | math |
| 2048 | 153.44 | 20.40 | 86.2 | math |

All three predictions held: TTFT@2048 landed at 153.4 ms inside the predicted 130–260 ms band; ITL was flat within 0.8% across a **16× change in prompt length**; and 128 was only 1.4× cheaper than 512 despite having 4× fewer tokens.

**Prefill efficiency: 69.5 measured ÷ 42.1 floor = 1.65× haircut → 61% of peak dense BF16 FLOP/s.** This is the prefill analogue of MBU — fraction of the *binding* resource in use. Decode binds on bandwidth (0.87); prefill binds on compute (0.61).

**Crossover confirmed from an unplanned direction.** Marginal cost per 1k prompt tokens is not constant: 36.0 ms/1k across 128→512 versus 69.5 ms/1k across 512→2048. Adding prompt tokens is roughly half price below the predicted ~418-token crossover, because prompt length is not a term in the byte floor and the byte floor is what binds down there. Above the crossover it goes fully compute-limited and linear.

**The crossover equals the ridge, and that is algebra not luck:** crossover N solves `bytes/bandwidth = 2·P·N/FLOPs`; with bytes = 2P the P cancels and `N = FLOPs/bandwidth` = 419. This identity only holds at exactly 2 bytes per parameter — at fp8 the crossover halves (Lesson 8).

**Key constant to quote from memory: 42.1 ms per 1k prompt tokens** on this card (`2 × 7.62e9 × 1000 ÷ 362e12`). Real-world rate with the efficiency haircut: **~70 ms/1k**. A 3k-token system preamble in front of every request therefore costs ~210 ms of unavoidable TTFT — a prompt-shortening problem, not a faster-GPU problem.

**Lesson 4 — the second term.** `bytes/step = weight_bytes + Σ over live requests (56 KiB × its context length)`. Weights are shared and frozen; each request's KV cache is private and **re-read in full every step**.

Derived from `config.json` (28 layers, 28 attention heads, **4 KV heads**, hidden_size 3584, no `head_dim` key so derived as 3584/28 = 128, bfloat16):
```
2 (K and V) × 28 layers × 4 kv_heads × 128 head_dim × 2 bytes = 57,344 B = 56 KiB/token
```
**The 7× trap:** using `num_attention_heads` (28) instead of `num_key_value_heads` (4) gives 401,408 B = 392 KiB — it would turn a 115-session box into a 16-session box. 28 query heads share each stored pair in groups of 7.

**Capacity, from the measured pool of 470,400 slots:** ~115 sessions at 4,096-token context, but only **~14** at 32,768. Eight times the context, eight times fewer users, perfectly linear — predictable from a config file before it ever happens in production.

**A full box on this rig is already past the crossover.** 470,400 slots ÷ 265,000 crossover = **1.77×**. At full pool the step reads 15.24 GB of weights *plus* 26.97 GB of KV cache = 42.2 GB, so the floor becomes 42.2/864 = **48.8 ms — 2.8× the 17.6 ms used for three lessons.** Nothing redeployed; the second term simply grew until it owned step time.

**Throughput collapse is multiplicative:** 4k chats give ~115 tokens per step; 32k documents give ~14 — at the same ~49 ms step cost. Roughly 2,350 tok/s versus 290 tok/s. Same full pool, same busy GPU, 8× less output.

**Gauge verified live.** Metric name in this version is **`vllm:kv_cache_usage_perc`**, not the `gpu_cache_usage_perc` the lesson cites. Under 32 concurrent × 2048 input it read 0.1467 → `0.1467 × 470,400 = 69,013` tokens in flight, against an arithmetic expectation of 65,536–69,632. That is 26% of crossover, which is why ITL stayed near 20 ms. Also watch **`vllm:num_requests_waiting_by_reason{reason="capacity"}`** — nonzero means the pool is full and requests are being deferred. `nvidia-smi` is blind to all of this: the pool is claimed at boot, so free VRAM reads flat at 0 or 115 sessions.

**Operational note learned the hard way:** the server was killed by a stray `SIGTERM` mid-session and a subsequent benchmark ran against a dead server, silently reporting `ITL 0.00`. **Always confirm `/health` returns 200 before trusting any benchmark number.** Run the server under a process manager (systemd/tmux/`setsid`) rather than bare `nohup`.

**Lesson 5 — batching, and which ceiling bends the curve.** `step_time = (W + B×C×kv) / bandwidth`. Per token that is `W/(B×bandwidth) + (C×kv)/bandwidth` — the first term amortizes like 1/B, the second never shrinks. The knee is where they cross: `B* = 265,000 / C`.

**Central claim confirmed — the knee moved.** Batching efficiency measured as fraction of ideal scaling:

| | B 1→16 (ideal 16×) | B 16→64 (ideal 4×) |
|---|---|---|
| C=512 | 13.6× = 85% | **2.56× = 64%** |
| C=3840 | 8.1× = 51% | **1.28× = 32%** |

Predicted knees were 518 at C=512 (so B=64 sits far below it, still gaining) and 69 at C=3840 (so B=64 sits right at it, flattened). Both held. **The knee moving left when context grew — with weights identical — is the proof that this box pins on KV cache bandwidth, not on the 419 FLOP/byte math ceiling.** Arithmetic intensity pins at 265,000/C ≈ 69 at long context; it never approaches 419, so the math ceiling is unreachable here regardless of batch size.

**Cost, at the real $1.86/hr rate:** $10.56/1M output tokens at B=1 → **$0.30 at B=64. 35× cheaper**, same GPU, same hour, same weights. Long context costs 3.6× more per token at identical batch size ($1.08 vs $0.30) because bandwidth goes to saved state instead of to more users. Formula: `$/1M = hourly ÷ (tok/s × 3600) × 1e6`.

**⚠ The equation is optimistic on magnitude.** ITL increment from B=1 to B=64: measured +9.25 ms vs +2.14 ms predicted at C=512 (**4.3× off**), and +26.29 ms vs +16.05 ms at C=3840 (1.64× off). There is real per-request cost beyond KV bytes — attention kernel overhead and per-sequence scheduler work — that the model ignores, and it dominates at short context where KV bytes are small. **Trust `265,000/C` for where the knee is; do not trust the equation for how bad ITL gets there.**

**Measurement rule learned by falsifying a hypothesis.** I predicted the early knee at C=512 was prefill contaminating the measurement, and ran one-wave controls (`--num-prompts B` instead of `B*8`) to test it. **Median ITL was identical** — 29.51 vs 29.49 at C=512, 46.89 vs 46.22 at C=3840 — so the hypothesis was wrong for the median. But **mean ITL diverged hard**: 124.58 (8 waves) vs 79.54 (1 wave) at C=3840, against a median of ~46. Conclusion: **report median ITL; prefill interference is real but lives entirely in the tail, and a large mean/median gap is the diagnostic signal that prefill is stealing decode time.**

**Sweep bug worth remembering:** the lesson's own command asks for `--random-input-len 4096 --random-output-len 256` against a server pinned at `--max-model-len 4096`. 4096 + 256 = 4352 > 4096, so **all three long-context runs were rejected and silently reported `ITL 0.00`**. Rerun used input 3840 + output 256 = exactly 4096. Always check for zero-throughput rows before trusting a sweep.

**Lesson 6 — open-loop arrivals, and the third ceiling.** Rows 1–5 all used `--max-concurrency`, which is a **closed loop**: the client only sends when a request completes, so when the server slows the client sends slower and the load quietly shrinks. Waiting is bounded by the pinned in-flight count, so p99 can never diverge. **Any p99 quoted from a closed-loop run is a best case production will never see — delete `--max-concurrency` from anything you plan to quote.** Rows 1–5 remain valid for step time and throughput, which don't need a queue; they simply cannot produce a p99.

Sweep: `--request-rate {4,6,8,10,12}`, Poisson-spaced, no concurrency pin, 1024 in / 128 out held fixed (so `TTFT = queue delay + prefill` isolates queue delay), 200 prompts each.

**The third ceiling — step-time occupancy.** The engine has 1.0 s of step time per second of wall clock; each arrival spends ~72 ms of it on prefill (1024 tokens × ~70 ms/1k from Lesson 3). `occupancy = rate × 0.072` → 0.29 / 0.43 / 0.58 / 0.72 / 0.86 across the sweep. The wall is `1 ÷ 0.072 = 13.9 req/s`, where prefill alone consumes the machine. **This is neither the byte ceiling (Lesson 5) nor the slot ceiling (Lesson 4).**

**⚠ Prediction failed structurally: goodput is 0 req/s under the stated SLO, not 6.** p99 ITL is **139 ms at rate 4** — the lightest load tested — against a 50 ms bound. The bound is unachievable on this configuration at any arrival rate.

Why: **p99 ITL is 5.4× p50 ITL** (139 vs 26 ms at rate 4). This is Lesson 5's mean/median finding amplified. Poisson arrivals bunch, so several prefills land on the same step, and chunked prefill lets prompt work preempt token generation — a user mid-stream simply stops receiving tokens for ~139 ms. **Rate barely moves p99 ITL** (139 → 234 ms across a 3× rate increase), which proves it is a *scheduling* problem, not a rate problem. The lesson's reference machine reported p99 ITL of 38 ms at rate 4; this rig measures 139. The measurement wins.

**Shippable goodput: 8 req/s** under p99 ITL < 200 ms and p99 TTFT < 1000 ms — 885 tok/s, p99 TTFT 628 ms.

**Textbook overload signature, rate 10 → 12:** offered rate +20%, throughput **+4%** (998 → 1039 tok/s), p99 TTFT **+142%** (1298 → 3135 ms). Throughput flattens while the tail explodes. Peak throughput without a latency bound is a vanity metric — you reach it by letting the queue grow forever, and above capacity the number you report is a function of test duration, which means it is not a number.

**Confirmed step-time bound, not slots.** Zero preemptions at every rate. Little's Law at rate 8: residence ≈ 250 ms + 127 × 33.65 ms ≈ 4.5 s, so in-flight ≈ 8 × 4.5 = **36 requests against ~393 available slots** at this 1,152-token shape. Ran out of step time with ~10× the slot headroom spare.

**Operational fix applied:** vLLM was being killed by `SIGTERM` twice when subsequent SSM commands ran — `nohup`/`setsid` did not reliably escape the teardown. Installed it as a **systemd unit** (`/etc/systemd/system/vllm.service`, `Restart=on-failure`), after which it survived the entire 25-minute sweep. Long-running benchmarks are launched via `systemd-run --unit=... --collect` for the same reason.

**Lesson 7 — paged KV cache and refcounted block sharing.** A naive allocator reserves `max_model_len` per request because it cannot know the final length at admission. That caps concurrency at `pool ÷ max_model_len` regardless of real traffic: `453,000 ÷ 4,096 = 110`. Charging only what a request uses (~600 tokens) gives `453,000 ÷ 600 = 755`. **~7× of the pool lost to an allocator policy, not to hardware — and Lesson 4's ceiling of 110 was that counterfactual all along.** The ratio is pool-independent (`max_model_len ÷ actual_need`). **vLLM has no unpaged mode, so 7a is computed; never imply it was A/B'd.**

The fix: chop the pool into uniform **16-token blocks**, hand them out **on first touch** rather than at admission, and give each request a **block table** mapping logical token position → physical block index (`i >> 4`, `i & 15`). Blocks need not be adjacent. Remaining waste is only the partially-filled last block, **bounded at 15 slots**. This is OS paging: blocks = frames, block table = PTE array, tail waste = internal fragmentation — except **translation happens in software inside the attention kernel**, there is no MMU and no swap. Lesson 6's preemption is *not* a page fault: it evicts a whole request at request granularity, while paging is an addressing scheme at block granularity. **Say "it's a page table with refcounted frames and no swap."**

**Sharing falls out for free.** Blocks are interchangeable and a written block is immutable, so two requests' tables can point at the same physical block. Each block carries a **reference count**; identical leading token sequences share those blocks and the second request skips computing them. **Never call it copy-on-write** — a written KV block is never mutated, so there is no write to copy on. Three names kept separate: *prefix caching* = the flag, *refcounted block sharing* = the mechanism, *prefix reuse* = the measured effect.

Sharing requires **exact token IDs** (not similar text), **starting at position 0**, in **whole blocks**. Measured: `4,064 hits / 6,144 queries = 66.1%` against a predicted 66.1% — exact. Hits were `2 warm runs × 127 blocks × 16 tokens`; **127 not 128 because the engine prunes the last matching block** so there is always work to run, which is why warm TTFT is 31 ms rather than ~0.

**⚠ Run 1 was not a clean cold measurement.** It read 524 ms against Lesson 3's 153 ms for the same 2048-token prompt — 3.4× slower because it was the first request after a systemd restart and absorbed one-time warmup (JIT, lazy kernel load, allocator init). The altered-first-token run gave **144.5 ms**, consistent with Lesson 3, and that is the honest cold number. **Quote 4.7× (144.5 → 31.0), not 16×** — the larger figure is mostly this server's startup cost, and "was that the first request after boot?" would expose it.

**Invalidation confirmed by control:** changing only the **first** token pushed TTFT back to 144.5 ms, then 31.0 ms on repeat. One differing token at position 0 invalidates all 128 blocks. **Practical consequence: order prompts stable-text-first, varying-text-last.** Putting a username ahead of a 2,000-token preamble yields a permanent zero hit rate.

**Prefix reuse moves TTFT only, never tokens/sec** — it deletes prefill work already done and leaves the decode clock untouched. But since prefill occupancy is exactly what capped goodput at 8 req/s in Lesson 6, **a shared preamble raises goodput with no hardware change.** Rows 1–6 were measured with caching off and remain valid: those benchmarks sent random prompts sharing no prefixes, so caching could not have hit.

**Server config changed for this lesson:** dropped `--no-enable-prefix-caching`, added `--block-size 16`. Pool size unchanged at ~470k slots — **sharing costs zero memory, it is a refcount.**

**Lesson 8 — quantization, the first dial that is not output-preserving.** Every earlier dial (graph capture, batching, paging, prefix reuse) produced identical tokens bit-for-bit. This one does not, which is why it needs a fidelity row. Mechanism: 1 byte per weight instead of 2, with one shared multiplier (**scale**) per block — `code × scale ≈ original`. vLLM converts at load, reading 16-bit files from disk and writing 8-bit into VRAM; nothing on disk changes. sm89 (Ada) is a tuned path for 8-bit weights *and* 8-bit math, not a fallback.

**Two independent dials — naming which one you turned is the whole claim.** `--quantization fp8` shrinks weight bytes (15.24 → 7.62 GB, floor 17.6 → 8.8 ms). `--kv-cache-dtype fp8` shrinks KV bytes per token (56 → 28 KiB) and on sm89 needs `--attention-backend TRITON_ATTN`, since only FlashAttention's FA3 build reads an fp8 cache and FA3 is sm90-only.

**8a — prediction 1.77×, measured 1.44×.** Predicted `8.8 + 2.63 = 11.43 ms`; measured **14.08 ms**. The **2.65 ms shortfall is the scale-apply cost** — an extra multiply per code on the way into the math units — plus non-weight work that never halves. Note the prediction was *arithmetically* correct; the model simply omitted a term. Also worth noting: a well-tuned baseline makes the predicted ratio look better than it can be. MBU 0.87 means the gap is only 2.63 ms, so the naive model predicts 1.77×, while the lesson's slower reference machine (13.4 ms gap) predicts 1.40×. **The measured 1.44× sits inside the stated 1.3–1.6× band; 1.77× was never achievable.**

**⚠ 8b — the lesson's central claim failed on this rig.** It predicted the fp8 win would collapse to ~1.0–1.15× at concurrency 64, on the reasoning that past the knee one weight pass serves the whole batch so halving it buys nothing. **Measured 1.59× — better than at concurrency 1, not worse.** The explanation was already on this scoreboard: at **C=512 the knee is B≈518** (row 5g), so B=64 is nowhere near it. Per-token weight term at B=64 is `17.6/64 = 0.275 ms` against a KV term of `0.034 ms` — weights still dominate 8:1, so halving them still pays. **The claim is right in principle but was tested at the wrong context length.** Reproducing the collapse requires B=64 at ~4k context, where the knee is 65.

**8c — free capacity bonus.** fp8 weights released 7.6 GB straight into the KV pool: **437,200 → 564,592 token slots (+29%)**. fp8 buys latency *and* concurrency. **Gap: the `--kv-cache-dtype fp8` pool size was never captured** — the collection grep only read that line from inside the benchmark suite, which was deliberately skipped for that config. `L08-kvfp8.log` persists on the EBS volume; read it next session. Expected ~2× the 16-bit pool, and Lesson 4's crossover moves 265k → ~530k with the KV dial alone, but **stays at 265k if both dials are on**, since numerator and denominator halve together.

**8d/8e — fidelity: one clean number, one bad metric.** The 16-bit server against itself scored **100.0% byte-identical** over 19 prompts — a perfect noise floor, confirming `temperature=0` and `seed=42` were properly pinned, so any fp8 disagreement is real signal rather than dice. But **fp8 vs 16-bit scored 0.0%, and that is a measurement-design error on my part**: I compared whole 200-token completions for exact equality, so a single divergent token anywhere fails the entire line. The number is true and nearly uninformative — it cannot distinguish "diverged at token 3" from "diverged at token 199." The lesson's expected 60–80% only makes sense for much shorter outputs or a per-token comparison. **Re-measure with per-token agreement or first-divergence position.**

**8f — cost:** 1,613 → 2,392 tok/s at concurrency 64, so **$0.32 → $0.22 per 1M output tokens** at $1.86/hr.

**Operational note:** long-running servers must be supervised (systemd, `Restart=on-failure`) rather than backgrounded with `nohup` — a stray `SIGTERM` killed the server twice mid-session and a subsequent benchmark ran against a dead server, silently reporting `ITL 0.00`. **Always confirm `/health` returns 200 before trusting a benchmark number, and check every sweep for zero-throughput rows.**

**Lesson 9 — the routing table and the sizing drill.** No new measurements. The hireable skill is a lookup: **symptom in, pinned resource out, with a number attached.**

| # | What you see | What's pinned | Evidence |
|---|---|---|---|
| 1 | ITL near the byte floor | nothing — done | MBU 0.87 |
| 2 | ITL far above floor, launch gaps | host, not the card | L2 |
| 3 | batching knee slides left as context grows | KV cache bandwidth | L5 — moved 7.5× |
| 4 | cache occupancy high, **preemptions climbing** | capacity / admission | L4, L6 |
| 5 | p99 TTFT rising **at fixed prompt length** | queueing | L6 — goodput 8/s |
| 6 | fp8 helps at low batch, not high | weight bytes, unamortized | L8 |

**Discriminating questions matter more than the table.** Pick the follow-up that *splits your two candidates*, not one merely related:
- Rows 4 vs 5 → **"what's the preemption count?"** Nonzero = out of KV space; zero = out of step time. L6 measured **zero at every rate**, including the 3.1 s p99 — ran out of time with ~10× slots free.
- TTFT up with ITL **flat** → prompt length grew (L3 measured ITL flat across a 16× prompt change). TTFT up with ITL **also up** → load (L6 measured ITL 26 → 46 ms as rate rose 4 → 12).

**The sizing drill — 300 req/s, 2k in / 300 out, p99 TTFT < 800 ms:**
```
cache/request  2150 × 56 KiB                     = 123 MB
knee B*        15.24 GB / 123 MB ≈ 124           → use B = 120
fit check      120 × 2150 = 258,000 of 470,400   → fits, 1.8× headroom
decode         (15.24 + 120×0.123)/864 = 34.7 ms → 300 × 34.7/120 = 87 ms/req
prefill        2 × 7.62e9 × 2000 / 362e12        = 84 ms/req   (matches L3's 2048 row)
add them       84 + 87 = 171 ms                  → 5.8 req/s per GPU at saturation
haircut        L6 goodput 8 / divergence 12 = 0.67 → 5.8 × 0.67 = 3.9 req/s per GPU
answer         300 / 3.9                         = ~77 GPUs, $0.44 per 1M output
```
**Size to the knee (120), not the memory ceiling (211).** Memory tells you what's *possible*; the knee tells you what's *shippable*. Packing to 211 would break the 800 ms p99 that was the actual constraint — same distinction as peak-vs-goodput in L6.

The lesson's worked example lands on **103 GPUs** because its reference machine had goodput 6. **This rig measured 8, so it gives 77. The procedure is portable; the digits are yours** — quoting 103 would be quoting someone else's machine.

**Two mechanisms computed and never run — say "computed, not measured" out loud.**
- **Sharding:** floor divides by N, minus two syncs per layer per step. Those messages are a few KB, so it is a **fixed latency tax, not a bandwidth problem** — hence ~1.4-1.7× at N=2, never 2×. Hard ceiling **N=4**, because N must divide `num_key_value_heads`=4; that comes from the model file, not the budget. *Trap: interconnect GB/s is the wrong axis.*
- **Speculative decoding:** works by spending the ~400× of math that sits idle during a weight read (1 FLOP/byte against a 419 ridge). **But batching already sold that capacity** — at batch 64 it goes negative, because drafting costs real work with no spare compute to recover it. **Batching and speculative decoding compete for the same resource.** *Trap: it preserves the output distribution, not the token stream — never claim bitwise identity.*

**Still true and unmeasured:** the ~1 µs figure is *visible* cost per launch, not actual host CPU cost per launch. Separating those requires measuring host CPU time directly — a profiler — which this course deliberately excludes. Do not claim the real launch cost is 1 µs.
