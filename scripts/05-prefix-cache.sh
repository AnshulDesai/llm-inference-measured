#!/bin/bash
# Lesson 7: refcounted block sharing. Requires prefix caching ON (default).
# Only WHOLE blocks share, so build a prefix that is an exact multiple of 16 tokens.
python3 - <<'PY'
import json, urllib.request, time
M="Qwen/Qwen2.5-7B-Instruct"
def post(p,o):
    r=urllib.request.Request("http://localhost:8000"+p, data=json.dumps(o).encode(),
                             headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r))
ids=post("/tokenize",{"model":M,"prompt":"the quick brown fox jumps over the lazy dog. "*400})["tokens"][:2048]
text=post("/detokenize",{"model":M,"tokens":ids})["prompt"]
body={"model":M,"prompt":text,"max_tokens":1,"temperature":0}
for i in (1,2,3):
    t0=time.time(); post("/v1/completions",body); print(f"run{i} {(time.time()-t0)*1000:.1f} ms")
PY
# Run 1 after a server restart absorbs one-time warmup -- it is NOT a cold measurement.
# Get the honest cold number by altering the FIRST token, which invalidates every block.
curl -s localhost:8000/metrics | grep -i prefix_cache   # counters are in TOKENS, cumulative
