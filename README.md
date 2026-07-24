# vllm-lite

An LLM inference server built from scratch: paged KV cache, continuous batching,
chunked prefill, prefix caching, speculative decoding, and an OpenAI-compatible
HTTP API.

The interesting part of LLM serving isn't the model, it's everything around it.
A transformer forward pass is a few hundred lines. Making thousands of
concurrent requests share one copy of the weights and one pool of KV memory
without wasting either is where the actual engineering is. This implements that
part, and measures it against the naive approach.

```
pip install -r requirements.txt
python -m vllmlite.api.server          # OpenAI API on :8000
python bench/bench.py --requests 48    # the numbers below
pytest                                 # 90 tests
```

## Results

48 requests, prompt and output lengths drawn from a lognormal (17-251 prompt
tokens, 20-192 output tokens), 6 layers / d_model 384 / 8 heads / 4 KV heads,
best of 3 runs on an M-series CPU. Every row uses the same weights, the same
sampling and the same KV memory budget, so the only variables are the memory
layout and the scheduler.

| config | tok/s | TTFT p50 | TTFT p99 | e2e p99 | fwd passes | KV slot use |
|---|---|---|---|---|---|---|
| sequential (batch=1) | 649 | 2849 ms | 4802 ms | 4865 ms | 3179 | 68% |
| static batch=4 | 1030 | 1792 ms | 2970 ms | 3088 ms | 1418 | 35% |
| static batch=16 | 1811 | 736 ms | 1375 ms | 1744 ms | 497 | **21%** |
| **paged + continuous** | **2028** | **520 ms** | **1179 ms** | **1543 ms** | **260** | **94%** |

**2.9x throughput and 5.5x better TTFT than sequential; 1.4x better TTFT than
the best static batch.** The number that matters most is the last column.
Static batching has to reserve `max_model_len` of KV per slot because it can't
know how long a sequence will get, so at a 16-wide batch only 21% of the
reserved cache ever holds real tokens. Paging allocates in 16-token blocks on
demand and runs at 94%. That 4.5x is 4.5x more concurrent requests out of the
same memory, which is what actually sets throughput on a real GPU.

Note that static batch=16 needs 497 forward passes to do what the paged engine
does in 260. Both process the same tokens. The difference is entirely
head-of-line blocking: a static batch runs until its *longest* member finishes,
so slots sit idle decoding padding.

### Prefix caching

Same benchmark with a 256-token system prompt shared across all 48 requests,
which is what production traffic actually looks like:

| config | tok/s | TTFT p50 | e2e p99 |
|---|---|---|---|
| static batch=16 | 910 | 1557 ms | 3611 ms |
| paged + continuous | 867 | 1398 ms | 3822 ms |
| **+ prefix caching** | **1289** | **775 ms** | **2567 ms** |

**+49% throughput and 1.8x better TTFT** from not recomputing the same system
prompt 48 times.

### Speculative decoding

4 repetitive prompts, 96 output tokens each, n-gram proposer:

| k | forward passes | tok/s | acceptance | tokens/pass |
|---|---|---|---|---|
| 0 (off) | 96 | 5301 | - | 1.00 |
| 2 | 58 | 7118 | 84.5% | 3.83 |
| 4 | 58 | 7190 | 76.3% | 3.88 |
| 8 | 58 | 7089 | 65.6% | 3.90 |

**1.66x fewer forward passes.** Output is token-for-token identical to
non-speculative greedy decoding at every k, which the test suite asserts.

Raw JSON in [`results/`](results/), regenerate with `--json`.

## How it works

```
HTTP request
    |
AsyncLLMEngine        one background task drives the loop, one queue per request
    |
LLMEngine.step()      exactly one forward pass, then sample and check stops
    |
    +-- Scheduler     picks who runs this pass, under a token budget
    |     +-- BlockManager    block tables, copy-on-write, prefix reuse
    |           +-- BlockAllocator   free list, refcounts, LRU cache of full blocks
    |
    +-- ModelRunner   owns the KV tensors, flattens the batch, calls the model
          +-- TinyLlama    RMSNorm / RoPE / GQA / SwiGLU in numpy
                +-- paged_attention   gather through the block table
```

### Paged KV cache

The cache is one flat tensor per layer, chopped into fixed 16-token blocks. A
sequence doesn't own contiguous memory, it owns a *block table*, and attention
gathers through it. Exactly the virtual-memory trick, applied to attention keys
and values:

```python
# vllmlite/attention.py
slots = (block_ids[:, None] * block_size + arange(block_size)).reshape(-1)[:ctx]
k = cache_k[slots]
```

That one indirection is what buys the 21% -> 94% utilization, because a
sequence can grow one block at a time instead of reserving its worst case up
front.

Blocks are reference counted, so two sequences can point at the same physical
block. Writing into a shared block triggers copy-on-write, which the runner
performs before the forward pass.

### Prefix caching

Every full block gets a hash of *all* tokens up to and including it, chained
through the previous block's hash so identical content in a different context
never collides. Full blocks are published to a hash table **the moment they
fill**, not when they're freed, so two requests sent seconds apart share the
same system prompt blocks while both are still running. Unreferenced cached
blocks stay around in an LRU pool and get evicted only under pressure.

### Continuous batching

The batch is rebuilt from scratch on every forward pass. A finished sequence
frees its slot immediately and a queued request takes it on the very next pass.
Running sequences get scheduled before waiting ones, oldest first.

Prefill and decode share a single token budget (chunked prefill), so a 2000
token prompt gets sliced across several passes instead of stalling everyone
else's decoding. Splitting a prompt this way is not supposed to change its
output, and the tests check that across chunk sizes of 8, 33, 64 and 4096.

### Preemption

When the cache runs dry, the scheduler evicts the newest sequences, drops their
KV, and pushes them back to the front of the queue to be recomputed. Recompute
beats swapping here because the prefix cache usually still holds most of the
evicted blocks. Sequences are evicted newest-first so the oldest request is
always guaranteed to make progress, which means the loop can't deadlock.

A prompt longer than the entire cache is retired with `cache_full` instead of
spinning forever. That was a real bug the tests caught.

### Speculative decoding

Decoding is memory bound: a forward pass drags every weight through cache to
produce one token. Verifying k guesses costs nearly the same pass, so good
guesses are close to free.

The proposer is n-gram lookup, not a draft model: search the context for the
last few tokens and copy whatever followed last time. No second set of weights,
no second KV cache, and it works well on the workloads people actually run
(long documents, code, RAG, anything where the output quotes the input). A
draft model implements the same `Proposer` interface.

Acceptance is the greedy rule: keep a guess while it matches what the real
model would have picked, take the model's own token at the first mismatch, and
collect a free bonus token if every guess held. Rejected guesses leave junk KV
behind, which costs nothing because `num_filled` only advances by the accepted
count and the next pass overwrites those slots.

## Being straight about what this is

The point of the project is the serving layer, so:

- **The weights are random.** The generated text is nonsense. That's on
  purpose, not a bug: random weights do the same arithmetic and move the same
  bytes as trained ones, which is what the scheduler and the cache are being
  measured on. Point `ModelConfig.weights_path` at an `.npz` to run real ones.
- **It's numpy on CPU, not CUDA.** `paged_attention` is a gather plus two batch
  matmuls where a real system has one fused kernel. Ratios between the
  configurations hold because every row runs the identical code path; absolute
  tok/s numbers mean nothing outside this machine.
- **Attention loops over sequences in Python.** Everything else (all the
  projections, the MLP, the LM head) is batched across the whole step. That
  loop is the part a GPU kernel would flatten.
- **Not implemented:** tensor parallelism, CUDA graphs, quantization, LoRA,
  swap-to-CPU preemption, beam search.

Grouped-query attention never materializes the repeated KV heads. Instead of
`np.repeat(k, n_rep)` the query heads fold into the batch dimension, so the
saving GQA exists for is actually realized:

```python
qg = q.transpose(1, 0, 2).reshape(n_kv_heads, n_rep * q_len, head_dim)
scores = qg @ k.transpose(1, 2, 0)
```

## Tests

90 tests, `pytest`. The ones that carried their weight are the invariance
tests, because paging, chunked prefill, prefix caching, preemption and batch
composition are all supposed to be invisible from outside. Greedy decoding has
exactly one right answer, so any of those knobs moving a token is a bug:

- output is identical across block sizes 1, 4, 16, 64
- identical with prefix caching on and off
- identical across chunked prefill budgets of 8, 33, 64, 4096
- identical whether a request runs alone or batched with 7 others
- identical after forced preemption and recompute
- identical with speculative decoding at k = 1, 2, 4, 8
- paged attention matches a dense reference to 1e-4, on deliberately scattered
  and out-of-order block tables, across three GQA ratios
- every block is returned to the allocator once the engine goes idle

## API

OpenAI-compatible, so any OpenAI client works by pointing `base_url` at it.

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "hello"}],
  "max_tokens": 32, "stream": true
}'
```

- `POST /v1/completions` and `POST /v1/chat/completions`, both with SSE streaming
- `GET /v1/models`, `GET /health`
- `GET /metrics` for live block usage, prefix hit rate, preemptions,
  acceptance rate
- usage reports `cached_prompt_tokens` so you can see prefix caching working

Client disconnects abort the request and free its blocks rather than letting it
run to completion into a dead socket.

Configuration is by environment variable (`VLLMLITE_NUM_BLOCKS`,
`VLLMLITE_BLOCK_SIZE`, `VLLMLITE_MAX_SEQS`, `VLLMLITE_MAX_BATCHED`,
`VLLMLITE_PREFIX_CACHE`, ...), see `vllmlite/api/server.py`.

## Layout

```
vllmlite/
  block.py          physical blocks, refcounts, LRU cache of full blocks
  block_manager.py  block tables, copy-on-write, prefix reuse
  scheduler.py      continuous batching, chunked prefill, preemption
  engine.py         step(), sampling, stop conditions, detokenization
  runner.py         owns the KV tensors, builds the flat batch
  model.py          RMSNorm / RoPE / GQA / SwiGLU decoder
  attention.py      paged attention and RoPE
  speculative.py    n-gram proposer and greedy verification
  sampling.py       temperature, top-k, top-p, repetition penalty
  tokenizer.py      byte-level, so nothing has to be downloaded
  async_engine.py   one engine loop, many HTTP connections
  api/              OpenAI-compatible FastAPI server
bench/              benchmark plus the static-batching baseline it competes with
tests/              90 tests
```
