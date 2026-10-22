# Benchmarks

Each table compares Kindi and llama.cpp on the same card, with the same model file and the same context. "Read" is
the prompt reading speed: an N-token prompt divided by its read time. "Write" is the generation speed at that depth.
All figures are tokens per second, from one stream, greedy, with Kindi's exact head. When only one engine runs at a
context, the table says why the other does not ("out of memory", "crashed", "not measured").

Unless a table says otherwise, the method is the same for the two engines. They get the same token ids. The prompt
is read in segments up to each N, and the read time of N tokens is the sum of the segments. At each N, 32 tokens are
generated. Project page: [amirhusain.com/kindi](https://amirhusain.com/kindi/).

## Quadro K420, 1 GB · Ternary-Bonsai-2-2B

2026-10-21. Kindi: dev commit bb2d506, ctx 24,576, f16 K/V. llama.cpp: the prebuilt Vulkan release,
`-ngl 99 -fa on -c 24640 -b 2048 -ub 512`, f16 K/V. Model file sha256 36061ad4….

| Context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---:|---:|---:|---:|
| 512 | 33.3 | 1.2 | 20.8 | 1.65 |
| 8,192 | 29.7 | crashed (Vulkan ErrorDeviceLost after 17 minutes) | 18.5 | crashed |
| 24,000 | 24.2 | not measured (it crashed at 8,192) | 15.3 | not measured |
| 32,768 | Kindi: does not fit (its maximum on 1 GB is 24,576) | loads; not measured | does not fit | not measured |
| 65,536 | does not fit | does not load (an 805 MB allocation fails) | does not fit | does not load |

## Quadro K1200, 4 GB, one card · 2B, 4B and 9B

2026-10-22. Kindi: dev commit 7fb99ff, ctx 24,576, f16 K/V, chunked prompt reading. llama.cpp: Prism ML's fork,
Vulkan build, `-ngl 99 -fa on -c 24640 -b 2048 -ub 512 --cache-ram 0`, f16 K/V. Each model had a 1-hour limit.

| Model | Context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---|---:|---:|---:|---:|
| 2B | 512 | 194.5 | 26.7 | 81.0 | 20.8 |
| 2B | 8,192 | 178.7 | 28.1 | 73.3 | 21.9 |
| 2B | 24,000 | 152.3 | 26.5 | 58.8 | 18.6 |
| 4B | 512 | 77.4 | 11.1 | 39.2 | 8.7 |
| 4B | 8,192 | 70.2 | 10.8 | 34.3 | 8.1 |
| 4B | 24,000 | 58.9 | 10.2 | 26.4 | 6.9 |
| 9B | 512 | 41.7 | 5.8 | 23.4 | 5.9 |
| 9B | 8,192 | 39.2 | 5.7 | 21.5 | 5.6 |
| 9B | 24,000 | 35.6 | not finished (the 1-hour limit ended during this prompt) | 18.1 | not measured |

The largest context on the card:

| Model | Kindi | llama.cpp |
|---|---|---|
| 2B | 131,072 (all that was asked) | loads at 65,536 (larger was not tried) |
| 4B | 86,272 | loads at 65,536 (larger was not tried) |
| 9B | 47,360 | loads at 65,536 (Kindi does not fit there) |

## Two Quadro K1200 cards · Ternary-Bonsai-2-27B

2026-10-21. Kindi: dev commit d62501b, the tensor-parallel engine, q8_0 K/V. llama.cpp: Prism ML's fork, Vulkan
build, `-ngl 99 -sm layer -fa 1 -ctk q8_0 -ctv q8_0`, with its default layer split (an even split does not load).

| Context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---:|---:|---:|---:|
| short (generation at a position below 150) | - | - | 13.0 | 1.69 |
| 512 (the same 512 token ids) | 21.0 | 1.63 | - | - |
| 1,000 | not measured | does not run (a 508 MB Vulkan allocation fails) | not measured | does not run |

The 512-token read was measured on 2026-10-22 with the 0.2.0 engine code (dev commit 7fb99ff) and llama-server
`-ngl 99 -sm layer -fa on -ctk q8_0 -ctv q8_0 -c 1024`, with the token ids sent to `/completion`.

At 1.3 tokens/s, a 24,000-token prompt takes about 5 hours on llama.cpp, so there are no deeper rows.

## GeForce GTX 1080, 8 GB · Ternary-Bonsai-2-27B

2026-10-21. Kindi: dev commit a9b627e, one-card plan engine, 5-trit weights, int8 chunked prompt reading, ctx
65,536, q8_0 K/V. llama.cpp: Prism ML's fork, CUDA build for sm_61, `-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -b 2048
-ub 512 -c 49280`. Model file sha256 53107f53….

| Context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---:|---:|---:|---:|
| 512 | 250.9 | 98.1 | 27.8 | 19.4 |
| 8,192 | 224.6 | 96.2 | 25.7 | 18.1 |
| 32,768 | 167.5 | 84.9 | 19.7 | 14.3 |
| 49,152 | 143.4 | 78.6 | 17.2 | 12.6 |
| 64,000 | 127.0 | out of memory (it does not load at 64K) | 15.4 | out of memory |

## GeForce RTX 3080, 10 GB · Ternary-Bonsai-2-27B

2026-10-22, one physical card (cluster GPU 7), both engines. The cluster cards have a power limit of 190 W; 380 W is
the default of the card. Kindi: the tensor-core engine of 0.2.0, development build of 2026-10-22,
`KINDI_TC=1 KINDI_FD=1 KINDI_ATT_TC=1`, exact head, q8_0 K/V. llama.cpp: Prism ML's fork 1180918, CUDA 12.4 build
for sm_86. llama-bench `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -r 1`: read pp512 / 3000 / 12000 / 22000 from an empty
cache, write tg32 at a depth of 1K / 32K / 64K / 90K. No CPU limit (`--cpus 4.0`).

| Context | Kindi read 190 W | llama.cpp read 190 W | Kindi read 380 W | llama.cpp read 380 W |
|---|---:|---:|---:|---:|
| 512 | 1,069 | 960 | 1,361 | 1,217 |
| 3,000 | 1,005 | 1,006 | 1,282 | 1,265 |
| 12,000 | 924 | 931 | 1,171 | 1,170 |
| 22,000 | 838 | 853 | 1,050 | 1,074 |

| Depth | Kindi write 190 W | llama.cpp write 190 W | Kindi write 380 W | llama.cpp write 380 W |
|---|---:|---:|---:|---:|
| 1,000 | 62.4 | 52.9 | 66.4 | 56.5 |
| 32,768 | 55.6 | 41.2 | 60.1 | 47.0 |
| 65,536 | 48.7 | 29.5 | 54.7 | 37.9 |
| 90,000 | 43.6 | 24.7 | 51.2 | 32.9 |

With one CPU core (`--cpus 1.0`, llama.cpp `-t 1 -tb 1`), the two engines stay within 5% of these values. Kindi
uses about 0.02 CPU cores per stream; llama.cpp uses about 0.77.

Agent prompts: 30 greedy and 20 sampled (temperature 0.7) pi prompts of 1,100 to 10,500 tokens, 128 tokens out,
through the servers (llama-server `-fa on -ctk q8_0 -ctv q8_0 -c 81920 -np 1 --cache-ram 0`). Write speed, greedy /
sampled:

| Engine | 190 W | 380 W |
|---|---:|---:|
| llama.cpp | 50.9 / 57.4 | 53.2 / 60.8 |
| Kindi, exact | 61.9 / 55.7 | 66.9 / 58.6 |
| Kindi + the 2B draft model (`KINDI_SPEC=draft2b`) | 100.3 / 85.3 | 107.9 / 90.9 |
| llama.cpp + the 2B draft model (`-md`) | does not load | does not load |

With the draft model, Kindi's greedy output is the same as without it (30 of 30 prompts). llama.cpp 1180918 does
not load the 2B as a draft model for the 27B: the two files have different Hadamard metadata.

## Why llama.cpp runs with Vulkan on the Kepler and Maxwell cards

The K420 has compute capability 3.0 (sm_30). CUDA 11 and later cannot compile code for it, and current llama.cpp
does not build with CUDA 10.2, the last toolkit for this card. The Vulkan backend runs on it with all layers on the
GPU, so the Vulkan build is the fair comparison. On the K1200 (sm_50) host, only the Vulkan build was available.

## Other published results

We looked for a published measurement of a modern language model on a Quadro K420, and on any Kepler GPU of the
sm_30 generation (GT 640/730, K420/K600). We found none. To our knowledge, Kindi is the fastest way to run a
current-generation LLM on this card. We have not had it independently verified. If you know of a faster result,
please tell us.

## How to reproduce

`kindi verify` checks the output against llama.cpp. `kindi bench` gives the write speed at several positions and
the prompt reading speed. [CARDS.md](CARDS.md) gives the settings for each card.
