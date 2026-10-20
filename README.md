<p align="center"><img src="assets/kindi-banner.svg" alt="Kindi - a modern 2B-parameter LLM at 20+ tokens/s on a 2014 NVIDIA Quadro K420 with 1 GB" width="100%"></p>

# Kindi 0.1.0

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Commercial licenses available](https://img.shields.io/badge/commercial%20licenses-available-green.svg)](COMMERCIAL-LICENSE.md)
[![Release v0.1.0](https://img.shields.io/badge/release-v0.1.0-lightgrey.svg)](https://github.com/ahtx/kindi/releases/tag/v0.1.0)

Kindi is a custom inference engine. It runs **Ternary-Bonsai-2-2B**, a modern 2-billion-parameter language
model with ternary weights. It runs this model on the **NVIDIA Quadro K420** (Kepler, 1 GB) at **20-23 tokens per
second**. It has an **OpenAI-compatible server** for chat clients, scripts and agents that write code. Its default
output is token-for-token identical to the output of llama.cpp for the same model file, and it is about 7x faster on
this card. The one-page summary is
[WHY.md](WHY.md).

<p align="center"><img src="assets/kindi-demo.gif" alt="The pi coding agent driving Kindi on a Quadro K420: it runs nvidia-smi, writes and runs a Python script, and checks disk space" width="100%"></p>

<p align="center"><sub>Recorded live: the <a href="https://pi.dev">pi</a> coding agent (with a Kindi colour theme)
connected to <code>kindi serve</code> on one Quadro K420, default settings. Token streaming is real time. We
shortened pauses longer than 2.5 s. The <a href="https://github.com/ahtx/kindi/releases/download/v0.1.0/kindi-demo-untrimmed.mp4">full
video</a> (123 s, no pauses shortened) and its <a href="https://github.com/ahtx/kindi/releases/download/v0.1.0/kindi-demo.cast">asciicast</a> are in the release.</sub></p>

<p align="center"><img src="assets/kindi-speed.svg" alt="Speed on one Quadro K420: llama.cpp Vulkan 2.84 tokens/s, Kindi 20.8 (exact) and 22.9 (fast); prompt reading 2.99 vs 26.5" width="100%"></p>

By [Amir Husain](https://www.amirhusain.com). Watch the [launch video](https://youtu.be/ddAnghj8HpE) on YouTube.

- [Requirements](#requirements)
- [Install](#install) - with Docker, or natively
- [Get the model and convert it](#get-the-model-and-convert-it)
- [Run the server](#run-the-server)
- [Use it](#use-it) - curl, Python, streaming, tools, agents
- [Exact and fast output heads](#exact-and-fast-output-heads)
- [The prompt cache](#the-prompt-cache)
- [Commands](#commands)
- [Configuration](#configuration)
- [Performance](#performance)
- [Agent benchmark](#agent-benchmark)
- [Limits](#limits)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Requirements

| | |
|---|---|
| GPU | NVIDIA Quadro K420 (GK107, 1 GB). We tuned Kindi's kernels for its single multiprocessor; we did not test other Kepler cards in 0.1.0. |
| Driver | NVIDIA **470** series (the last driver branch with Kepler support) |
| OS | Linux x86-64 (developed on Ubuntu 24.04 with Docker; the image itself is Ubuntu 18.04) |
| RAM / disk | 4 GB RAM or more; ~1.5 GB disk for the model and pack (+ ~4 GB for the Docker image) |
| Software | Docker with the NVIDIA Container Toolkit - **or** CUDA 10.2 + Python 3.6 + CuPy 9.6 (see below) |

The K420 must be free: Kindi uses about 870 MB of its 980 MB with the default 24,576-token context. A desktop
session on the same card takes a few MB and is fine.

## Install

### With Docker (recommended)

CUDA 10.2 and CuPy 9.6 are the last releases that support Kepler, and they need an old userland; the image packages
exactly that.

```sh
git clone https://github.com/ahtx/kindi.git && cd kindi
docker build -t kindi:0.1.0 -f docker/Dockerfile .   # downloads the CUDA 10.2 toolkit (~2.5 GB), ~15 minutes
```

You can also use the release archive. Download `kindi-0.1.0.tar.gz` from the
[v0.1.0 release](https://github.com/ahtx/kindi/releases/tag/v0.1.0), then run
`tar xzf kindi-0.1.0.tar.gz && cd kindi-0.1.0` and build the image with the same `docker build` command.

Check that Docker sees the GPU (create `~/.kindi` first, so it belongs to you and not to root):

```sh
mkdir -p ~/.kindi
docker run --rm --gpus all -u $(id -u):$(id -g) -v ~/.kindi:/data kindi:0.1.0 info
```

More in [docs/DOCKER.md](docs/DOCKER.md).

### Natively

You need CUDA 10.2 (toolkit with NVRTC and cuBLAS), Python 3.6 or later that CuPy 9.6 supports, and:

```sh
pip install "numpy==1.19.5" "fastrlock==0.8.3" "cupy-cuda102==9.6.0" "fastapi==0.79.1" "uvicorn==0.16.0" "pydantic==1.9.2"
```

Then run Kindi from the repository root with `python3 -m kindi ...` (put the repository on `PYTHONPATH` to run it
from elsewhere).

## Get the model and convert it

Download Prism ML's PTQ1_0 file (430 MB, Apache 2.0) from Hugging Face:

```sh
mkdir -p ~/.kindi && cd ~/.kindi
wget https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/resolve/main/Ternary-Bonsai-2-2B-PTQ1_0.gguf
```

Convert it once into Kindi's weight pack (about a minute; exact - the same weights, re-laid out):

```sh
# Docker
docker run --rm -u $(id -u):$(id -g) -v ~/.kindi:/data kindi:0.1.0 convert /data/Ternary-Bonsai-2-2B-PTQ1_0.gguf
# native
python3 -m kindi convert ~/.kindi/Ternary-Bonsai-2-2B-PTQ1_0.gguf          # writes ~/.kindi/pack
```

Check the engine against llama.cpp's reference output:

```sh
docker run --rm --gpus all -u $(id -u):$(id -g) -v ~/.kindi:/data kindi:0.1.0 verify
```

```
prompt 1 'The sky is blue because'          PASS  40/40 tokens match
prompt 2 'The capital of France is'         PASS  40/40 tokens match
prompt 3 'Water boils at a lower temper...' PASS  40/40 tokens match
prompt 4 'def add_two_numbers('             PASS  40/40 tokens match
ALL PASS: token-exact with llama.cpp - 22.1 tokens/s over the run
```

The speed in the last line includes the prompt tokens, so it is not the generation speed. It also depends on the
fast-head file of the next step. Without this file, `verify` uses a plain layout of the exact head and shows about 20
tokens/s. With the file, `verify` shows about 22 tokens/s. In both cases, the output is token-exact. `kindi bench`
measures the generation speed (see [Performance](#performance)).

Optional - build the cluster file for the fast output head (about 2 minutes):

```sh
docker run --rm --gpus all -u $(id -u):$(id -g) -v ~/.kindi:/data kindi:0.1.0 fast-head
```

## Run the server

```sh
docker run -d --name kindi --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
    -p 127.0.0.1:8093:8093 -v ~/.kindi:/data kindi:0.1.0
# native:  python3 -m kindi serve
```

The first start compiles Kindi's CUDA kernels for the K420 (about 1.5 minutes). Kindi caches them under
`state_dir`, and later starts take about 10 seconds. Then:

```sh
curl -s localhost:8093/health
```

The server listens on `127.0.0.1:8093` (natively) or on the port you publish (Docker). To serve your network, publish
`-p 8093:8093` (Docker) or set `host = 0.0.0.0`, and set an `api_key`.

## Use it

The endpoint is `http://localhost:8093/v1`, the model id `bonsai2-2b` (configurable).

```sh
curl -s localhost:8093/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "bonsai2-2b",
  "messages": [{"role": "user", "content": "Write a bash one-liner that counts the .txt files here."}]
}'
```

Python (`pip install openai`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8093/v1", api_key="none")   # your api_key if you set one
r = client.chat.completions.create(model="bonsai2-2b",
                                   messages=[{"role": "user", "content": "What is the capital of Italy?"}])
print(r.choices[0].message.content)
```

Kindi supports these request fields: `messages` (system / user / assistant / tool, text content), `tools` and
`tool_choice` (`"none"` disables tools), and `stream` and `stream_options.include_usage`. It also supports
`temperature` (0 = greedy), `top_p`, `top_k`, `max_tokens` / `max_completion_tokens`, `stop` and `seed`. It also
supports `chat_template_kwargs.enable_thinking` (off by default). When thinking is on, the thinking comes back as
`reasoning_content` (also streamed as `delta.reasoning_content`). `thinking_budget_tokens` caps the thinking. At the
budget, the server closes the thinking and the model then gives its answer.
Responses carry OpenAI `usage` (with `prompt_tokens_details.cached_tokens`) and a `timing` block.

**Tool calls** come back as OpenAI `tool_calls` with `finish_reason: "tool_calls"`; send the results back as
`role: "tool"` messages.

**Agents and chat UIs.** Point any OpenAI-compatible client at `http://localhost:8093/v1` with model `bonsai2-2b`.
Two settings matter for this card - see [docs/AGENTS.md](docs/AGENTS.md):
- Give the client a **long request timeout**. An agent's first prompt of 5,000+ tokens takes a few minutes to read the
  first time. The cache makes later turns fast. Do not use automatic retries, because they queue behind the current
  request.
- Tell the client that the context window is **24,576** tokens (or what you configured).

`kindi chat` is a terminal chat that needs no server.

## Exact and fast output heads

The output layer scores all 248,320 vocabulary entries for every generated token - about 9 ms of a 46 ms step.

- **exact** (default): the full head. Greedy output is token-for-token identical to llama.cpp.
- **fast**: a shortlist head. `kindi fast-head` clusters the vocabulary once. For each token, Kindi scores the 1,024
  cluster centres. Then it computes exact scores only for the 256 clusters with the best scores (~13 % of the
  vocabulary). Generation is about 15 % faster. On 310 measured steps, it chose the same token as the full head
  99.4 % of the time.

To change the head while the server runs, with no restart, use `kindi head fast` / `kindi head exact` (Docker:
`docker exec kindi python3 -m kindi head fast`). `/health` shows the current mode.

## The prompt cache

Bonsai mixes attention with recurrent Gated-DeltaNet layers, whose state cannot be rewound. Kindi caches anyway,
exactly:

- **checkpoints** of the recurrent state every 256 tokens. A request that shares a start with the previous request
  resumes at the last checkpoint before the first difference.
- **parked conversations**: Kindi keeps up to 3 other conversations in host RAM. It resumes each conversation when its
  next turn comes. (Agents alternate between their main prompt and side prompts.)
- **saved openings** on disk. When two requests share an opening of 512+ tokens and then differ, Kindi saves the
  state of that opening under `state_dir`. Typically, this opening is a system prompt with tool definitions. Later
  sessions load it in a fraction of a second, also after restarts.

Resumed output is identical to the output when Kindi reads the prompt from scratch.

## Commands

| | |
|---|---|
| `kindi convert MODEL.gguf [--out DIR] [--force]` | build the weight pack (default: the configured `pack`) |
| `kindi fast-head [--clusters N]` | build the fast-head cluster file (stop the server first: needs ~300 MB of GPU memory) |
| `kindi verify` | check greedy output against llama.cpp's reference |
| `kindi bench` | speed at several context positions, prompt read speed, GPU memory |
| `kindi serve [--host H] [--port P] [--api-key K] [--head exact\|fast]` | the OpenAI-compatible server |
| `kindi chat [--system TEXT]` | terminal chat |
| `kindi head exact\|fast` | set the output head while the server runs |
| `kindi info` | configuration, pack and GPU |

Global options: `--config FILE`, `--pack DIR`, `--state-dir DIR`, `--ctx N`.

## Configuration

Defaults, then a config file (`--config`, `$KINDI_CONFIG`, `./kindi.conf` or `~/.kindi/kindi.conf`), then
environment variables `KINDI_<KEY>`, then command-line flags. Copy [kindi.conf.example](kindi.conf.example) and edit
it. The file [docs/CONFIGURATION.md](docs/CONFIGURATION.md) describes the keys: paths, server address, API key, context length,
head mode, cache sizes and generation defaults.

## Performance

Quadro K420, Ternary-Bonsai-2-2B, `kindi bench`:

| | exact head | fast head |
|---|---|---|
| generation, short context | 20.8 tokens/s | 22.9 tokens/s |
| generation at 8,000 tokens of context | 18.6 | 20.9 |
| generation at 16,000 | 16.8 | 18.6 |
| generation at 24,000 | 15.4 | 17.0 |
| prompt read speed | 26.5 tokens/s | 26.5 |

<p align="center"><img src="assets/kindi-context.svg" alt="Generation speed by position in a 24,576-token context: exact head from 20.8 down to 15.4 tokens/s, fast head from 22.9 to 17.0; llama.cpp 2.8 at short context" width="100%"></p>

For comparison, llama.cpp runs the same model on the same card (Vulkan build, all layers on the GPU). It generates
2.8 tokens/s and reads prompts at 3.0 tokens/s. Method, comparison and context: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
How Kindi works: [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md).

## Agent benchmark

[kbench](https://github.com/ahtx/kindi-bench) has 65 agent tasks with hidden tests and a runner. It also has
the results of our runs with Kindi on the K420. kbench works with any OpenAI-compatible server.

## Limits

- **One request at a time.** Concurrent requests queue (the card holds one engine).
- **Kindi reads prompts at about 26 tokens/s**, so a long new prompt takes time. A new prompt of 5,000 tokens takes
  about 3 minutes. The cache makes repeated prompts fast. Kindi reads text that is not in the cache (for example, a
  large file or long command output) at this speed.
- **One model.** 0.1.0 runs Ternary-Bonsai-2-2B (the PTQ1_0 GGUF) only.
- **One GPU.** Tuned for the K420's single multiprocessor; larger Kepler or Maxwell cards would run but use one
  multiprocessor only.
- **A 2B model.** Good at short code, shell and factual questions, tool calls and summaries; it makes the mistakes of
  its size - check what it writes.

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). The usual causes are:

- Another program uses the K420 (out of memory at start).
- The driver is not the 470 branch.
- A client timed out while it waited for a long first prompt.

## About

[Amir Husain](https://www.amirhusain.com). The [Deep Tech with Amir](https://www.youtube.com/@DeepTechWithAmir)
series on VectorFall covers this work. The [Kindi launch video](https://youtu.be/ddAnghj8HpE) shows Kindi on the K420. More broadly, it also shows how to get the most from inexpensive GPUs.

## License

**AGPL-3.0; commercial licenses available.**

Kindi is free software under the [GNU Affero General Public License v3.0](LICENSE). If you distribute Kindi or a
changed version, the AGPL says that you must give the source code to the recipients. This is also true when users
use your version over a network. For use in a closed product or a hosted service, a paid commercial license is
available. See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).

The model is Prism ML's Ternary-Bonsai-2-2B under Apache 2.0. You download it from its publisher. The third-party
notices are in [NOTICE](NOTICE).

## Acknowledgements

Kindi exists because of the model it runs. Thank you to [Prism ML](https://huggingface.co/prism-ml) for
[Ternary-Bonsai-2-2B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf) and for its release under
Apache 2.0. Thank you to the Qwen team at Alibaba Cloud for Qwen3.5-2B, the base model of Ternary-Bonsai-2-2B. Thank
you also to the [llama.cpp](https://github.com/ggml-org/llama.cpp) authors. Their implementation is the reference
that we check Kindi's output against.
