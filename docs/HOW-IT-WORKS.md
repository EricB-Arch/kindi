# How Kindi works

## Hardware measurements

The Quadro K420 is one GK107 multiprocessor (SMX). It has 192 FP32 lanes at 875 MHz, 48 KB of shared memory, a
63-register limit per thread, and 1 GB of DDR3. We measured these values on the card:

| | |
|---|---|
| DRAM read bandwidth, 128-bit loads | 24 GB/s (32-bit loads: only 14 GB/s) |
| conflict-free shared-memory table lookup | 1.05 clocks per warp |
| integer shift / multiply, int-to-float conversion | 32 per clock (1/6 of FP32) |

The model streams ~500 MB of weights per token, so the bandwidth ceiling is ~45-50 tokens/s. In practice, the limit is
the SMX's **load/store pipe**. Every shared-memory lookup, warp shuffle and global load shares this pipe. So the floor
for this weight format is ~25 ms of matrix work per token, or ~30 tokens/s with perfect kernels. Kindi reaches about
70 % of that.

## The model

Ternary-Bonsai-2-2B is Qwen3.5-2B with ternary weights {-1, 0, +1} and one fp16 scale per 128 weights.
It has 24 layers. Of these, 18 layers use Gated-DeltaNet (linear attention with a 16 x 128 x 128 recurrent state).
The other 6 are full-attention layers (8 query heads, 2 KV heads, head dim 256). It also has a SwiGLU FFN (2048 -> 6144) and a
248,320-token vocabulary with a tied embedding / output head. Prism ML stores every matrix with a block Hadamard
rotation (512-wide, with random signs) folded in. The GGUF carries the sign vectors.

## The techniques

### 1. Ternary GEMV by table lookup, conflict-free (`k_g4`)

Kindi never unpacks the weights. Four ternary weights are one byte (`t0 + 3 t1 + 9 t2 + 27 t3`, 81 values). For each
group of 4 input values, Kindi precomputes the 81 possible signed sums once per token. A weight byte is then only an
index: one byte, one shared-memory load, one add per 4 weights.

- **Per-lane tables.** In a naive layout all 32 lanes of a warp look up random entries of the same table and collide
  in shared-memory banks (3-4-way conflicts). Kindi gives every lane its own tables in the lane's own bank column
  (entry `e` of lane `l` at word `e*64 + l`). So lookups can never conflict.
- **The table build is a ternary recursion.** Kindi builds each table level by level. Each level is three shifted
  copies of the level below, one per value of the next trit (`b3 = (t3-1)x3`, `b2 = b3 + (t2-1)x2`, ... ). This is
  the same self-similar construction as the ternary Cantor set. It takes 27 stores per thread with no division.
- **One instruction per address.** The entry stride is 256 bytes. So one `__byte_perm` instruction puts the lane
  offset into bits 0-7 and a weight byte into bits 8-15. The result *is* the byte address.
- **Interleaved rows.** Kindi stores the weights as 4 rows per 16 bytes. So each lane issues one 128-bit load per 4
  rows. This is the load width that reaches 24 GB/s on this card.
- **Scale after a partial reduction.** The butterfly that sums lanes runs xor-1, -2, -4 first. These first steps
  combine the lanes that share a 128-column scale group. So Kindi applies the fp16 scale once per 2 rows instead of
  once per row. Kindi also stores the scales that a warp needs contiguously (one coalesced 128-byte load).
- **Straight-line multi-row butterflies.** Kindi reduces 16 rows across 32 lanes in 16 shuffles. It does not index
  arrays dynamically, because that would spill to local memory.
- **Prefetching.** Kindi loads the inputs of the next phase while the current phase computes.
- **Fusion.** Up to three matrices that share an input (q/k/v, gate/up, qkv/gate) run in one launch.

### 2. Glue kernels

- **fp64 sums at warp-shuffle speed.** The norms in llama.cpp accumulate in double precision. Kindi does this too
  (it is one part of what keeps Kindi token-exact). But Kindi reduces with warp shuffles, one warp per row, instead of
  a 1,024-thread tree.
- **Fused norm + Hadamard fold** (`k_rmsfold`) and **fused SiLU-gate + fold** (`k_smfold`). Prism folded a rotation
  into the weights. Kindi applies that rotation to the activations with an in-place fast Walsh-Hadamard transform. It
  does this in the same kernel as the operation before it.
- **GDN glue**: gates, causal conv + SiLU, L2 norms and the delta-rule update in four kernels.

### 3. Attention

- **fp16 K/V cache** (as llama.cpp uses by default). This halves the memory, which lets a 24,576-token context fit in
  1 GB. It also halves the attention traffic.
- **GQA-aware kernels.** Kindi reads each K/V row once for its 4 query heads. The score kernel keeps q in registers.
  In it, a warp processes 4 positions per pass and reduces 16 sums with one butterfly. The value kernel splits
  positions into chunks and combines them in order.

### 4. Output head

- **Exact head**: the full 248,320-row GEMV and an argmax, on the GPU.
- **Fast head** (optional): Kindi clusters the head rows once with spherical k-means into 1,024 clusters. It stores
  them in cluster order, padded to 16-row groups, and the embedding lookup follows the permutation. For each token,
  Kindi scores the centroids and runs the GEMV only over the row groups of the best 256 clusters (~13 % of the rows).
- **Sampling without a copy of the vocabulary**: Kindi finds the top-k threshold on the GPU by binary search. It copies
  only the candidates to the host.

### 5. Prompt reading and caching

- **No head while Kindi reads a prompt**: Kindi computes logits only for the last prompt token (+26 % prompt speed).
- **Recurrent-state checkpoints**, **parked conversations** and **saved openings** (see the README) make prefix reuse
  exact for a model whose recurrent layers cannot be rewound.
- **A tokenizer equal to llama.cpp's.** Kindi reproduces the Qwen3.5 pre-tokenizer exactly, with regex alternation
  semantics. It splits out added tokens first. The result matches `llama-tokenize` on chat prompts, tool calls, code
  and Unicode.

## Ideas that did not help on this card

- **5 trits per byte (243-entry tables).** 20 % fewer lookups, but the bigger tables shrink each phase to 160 columns
  and triple the cross-lane reductions - more load/store work in total.
- **Batched prompt reads** (one weight read for 4-16 tokens). They are correct, but slower per token than the lookup
  GEMV. The 63-register limit caps the batch. Also, Kepler's slow integer units make the decode of base-3 bytes cost
  more than the shared load saves. On Maxwell and later GPUs it should win.
- **Activation sparsity / fuzzy alpha-cut thresholds, speculative decoding, a fuzzy controller for speculation
  depth.** Kindi does not use them. The SwiGLU activations are not truly sparse. Speculation needs a batched forward
  pass that is cheaper than k single steps, and this card does not provide one.
- **A software pipeline for the GEMV.** No gain: the load/store pipe, not latency, is the bound.

## Accuracy

With the exact head, greedy output equals llama.cpp's for the same GGUF, token for token (`kindi verify`). This is
because the arithmetic order of every reduction that matters is either the same or exact (fp64 sums, in-order chunk
combines). The pack is also a lossless re-layout of the file.
