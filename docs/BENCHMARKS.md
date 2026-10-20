# Benchmarks

## Kindi 0.1.0 on a Quadro K420

Hardware: NVIDIA Quadro K420 (GK107, 1 GB DDR3, driver 470), Intel Xeon W3503 host. Model: Ternary-Bonsai-2-2B
(PTQ1_0 GGUF). Context 24,576 tokens. Measured with `kindi bench` (single stream; tokens/s = 1 / time per token).
Project page: [amirhusain.com/kindi](https://amirhusain.com/kindi/).

| | exact head | fast head |
|---|---|---|
| generation at position 100 | 20.8 | 22.9 (position 2,000) |
| generation at 8,000 tokens of context | 18.6 | 20.9 |
| generation at 16,000 | 16.8 | 18.6 |
| generation at 24,000 | 15.4 | 17.0 |
| prompt read | 26.5 | 26.5 |

Through the server, with sampling, a short reply runs at 19-21 tokens/s (exact) and 23-24 tokens/s (fast).
`kindi verify` reports ~22 tokens/s over its run (prompt + 40 generated tokens per prompt).

GPU memory with the 24,576-token context: ~865 MB used, ~115 MB free of 980.

## Compared with llama.cpp on the same card

Same machine, same K420, same model family. The comparison uses Prism ML's llama.cpp build with the Vulkan backend
and all layers on the GPU (`-ngl 99`, Q2_0 file of the same model):

| | llama.cpp (Vulkan) | Kindi 0.1.0 (exact) | |
|---|---|---|---|
| generation | 2.84 tokens/s | 20.8 tokens/s | 7.3x |
| prompt read | 2.99 tokens/s | 26.5 tokens/s | 8.9x |

The baseline is the Vulkan build of llama.cpp because the CUDA build cannot run on this card. The K420 has compute
capability 3.0 (sm_30). CUDA 11 and later toolkits cannot compile code for compute capability 3.0, and current
llama.cpp CUDA builds need one of these toolkits. CUDA 10.2 is the last toolkit for this card, and current llama.cpp
does not build with it. The Vulkan backend runs on the K420 with all layers on the GPU. So the Vulkan build is
the fair comparison on this card.

An earlier CuPy port of the reference arithmetic ran at 3.7 tokens/s on the same card.
[HOW-IT-WORKS.md](HOW-IT-WORKS.md) describes the engine work that took it to 21-23 tokens/s.

## Other published results

We looked for any published measurement of a modern language model that runs on a Quadro K420. We also looked for
one on any Kepler GPU of the sm_30 generation (GT 640/730, K420/K600). We found none. General benchmark collections and llama.cpp
discussions cover newer cards, and recent CUDA releases dropped Kepler support. To our knowledge, Kindi is
the fastest way to run a current-generation LLM on this card.
We have not had it independently verified; if you know of a faster result, please tell us.

How to reproduce: `kindi verify` (correctness) and `kindi bench` (speed) on a K420 with the model file named in the
README.
