# Configuration

Kindi reads its settings in this order - later sources win:

1. built-in defaults;
2. a config file: `--config FILE`, else `$KINDI_CONFIG`, else `./kindi.conf`, else `~/.kindi/kindi.conf` (INI, one
   `[kindi]` section). Unknown keys are an error, so typos do not pass silently;
3. environment variables `KINDI_<KEY>` (for example `KINDI_PORT=9000`, `KINDI_API_KEY=...`);
4. command-line flags (`--pack`, `--state-dir`, `--ctx`, and for `serve`: `--host`, `--port`, `--api-key`, `--head`).

`kindi info` prints the final values and the file that each value came from. The Docker image sets
`KINDI_PACK=/data/pack`, `KINDI_STATE_DIR=/data/state` and `KINDI_HOST=0.0.0.0`.

## Keys

### Paths

| key | default | |
|---|---|---|
| `pack` | `~/.kindi/pack` | the converted model (`kindi convert` writes it) and, optionally, `head_clusters.npz` |
| `state_dir` | `~/.kindi/state` | the kernel cache (`kernels/`), saved prompt openings (`openings/`) and the head mode set by `kindi head` |

### Server

| key | default | |
|---|---|---|
| `host` | `127.0.0.1` | address to listen on; `0.0.0.0` serves your network - then also set `api_key` |
| `port` | `8093` | |
| `api_key` | (empty) | if set, every `/v1` request needs `Authorization: Bearer <api_key>`; `/health` stays open |
| `model_name` | `bonsai2-2b` | the model id `/v1/models` reports; clients may send any model name |

### Engine

| key | default | |
|---|---|---|
| `ctx` | `24576` | context length in tokens (prompt + answer). For GPU memory, see the notes below the table |
| `head` | `exact` | `exact` = full output head (token-exact with llama.cpp); `fast` = shortlist head (needs `kindi fast-head`). `kindi head exact\|fast` overrides it at run time |
| `fast_clusters` | `256` | clusters the fast head scores exactly (of 1,024). Fewer is faster and less exact (see the notes below the table) |

Notes:

- `ctx`: GPU memory is ~12 KiB per token for the fp16 K/V cache, on top of ~570 MB for weights and buffers.
- `ctx`: 24,576 leaves ~115 MB of the K420's 980 MB free. Lower it if something else needs the card.
- `fast_clusters`: 128 clusters found the full head's top token 98.1 % of the time, 256 clusters 99.4 %.

### Prompt cache

| key | default | |
|---|---|---|
| `checkpoint_every` | `256` | interval between recurrent-state checkpoints (host RAM, ~19 MB each). Smaller = less to read again after a change, more RAM |
| `prefix_max` | `8` | saved openings kept on disk (each ~12 KiB per token + 19 MB; ~80 MB for a 5,000-token opening) |
| `prefix_min` | `512` | the shortest opening that Kindi saves |
| `park_max` | `3` | other conversations kept in host RAM (each: its checkpoints + its K/V cache) |

### Generation defaults

Used when a request does not set them: `max_tokens` (`1024`), `temperature` (`0.7`; `0` = greedy), `top_p`
(`0.8`), `top_k` (`20`). These are the model publisher's recommended non-thinking settings.
