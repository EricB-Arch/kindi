# The one-command installer

`install.sh` installs Kindi with one command. It finds your cards, selects the model and the settings, downloads
and checks the model, converts it, checks the output against llama.cpp and starts the server as a service.

```bash
curl -fsSL https://amirhusain.com/kindi/install.sh | bash
```

To see each step and each command before anything changes, do a dry run first:

```bash
curl -fsSL https://amirhusain.com/kindi/install.sh | bash -s -- --dry-run
```

## Install with a checksum check

Download the script, compare its sha256 with the value below, read it, and then run it:

```bash
curl -fsSLO https://amirhusain.com/kindi/install.sh
echo "e20573338e7d8048387265ef5958cc70ad3c23862e709f6127e32bc424ad22fb  install.sh" | sha256sum -c -
bash install.sh --dry-run
bash install.sh
```

| File | sha256 |
|---|---|
| `install.sh` (Kindi 0.2.0) | `e20573338e7d8048387265ef5958cc70ad3c23862e709f6127e32bc424ad22fb` |

If `sha256sum -c` does not print `install.sh: OK`, do not run the script. The same script is in the repository root
and in the 0.2.0 release.

## What it does

| Step | What it does | sudo |
|---|---|---|
| 1. Check the machine | Linux x86-64, RAM, free disk, the NVIDIA driver and each GPU with its compute capability. No GPU: the CPU engine. | no |
| 2. Docker and GPU access | Finds Docker and the NVIDIA Container Toolkit. If one is missing, it asks before it installs it (apt). Without the toolkit, it gives the container the GPU devices and the driver libraries directly. | only to install |
| 3. Select the model | Uses the profiles of [CARDS.md](CARDS.md) (table below). `--model` and `--ctx` override them. | no |
| 4. Build the image | Builds `kindi:0.2.0` from the release source. A re-run with the same source does not build again. | no |
| 5. Download the model | Downloads the GGUF from Prism ML on Hugging Face. The sha256 must match the value in the script (table below). | no |
| 6. Convert and verify | `kindi convert`, then `kindi verify`: the output must be token-for-token identical to llama.cpp. | no |
| 7. Start the server | A Docker container with `--restart unless-stopped` on `127.0.0.1:8093`. It waits until `/health` answers. | no |

At the end it shows the URL, the model, the context, the speed of the check, and the settings for OpenAI clients
and for pi. It writes no secrets and asks for no keys. The server listens on 127.0.0.1 only.

If your user is not in the `docker` group, the script uses `sudo docker`.

## Model selection

| Profile | Cards | Model | Context |
|---|---|---|---|
| cpu | no NVIDIA GPU, or `--gpu cpu` | 2B | 24,576 |
| kepler | sm_30-sm_37 (K420 1 GB, K420 2 GB) | 2B | 24,576; 65,536 at 1.5 GB or more |
| maxwell | sm_50-sm_75 with one card | 9B at 3.8 GB or more, 4B at 1.9 GB or more, else 2B | the engine fits it |
| k1200x2 | two cards of 3.8 GB or more, below sm_61 (2x K1200) | 27B, tensor-parallel | 24,576 |
| pascal8 | sm_61 or newer, 7.5 GB or more (GTX 1080) | 27B | 65,536 |
| tc | sm_80 or newer, 9.5 GB or more (RTX 3080) | 27B, tensor cores + 2B draft | 131,072 asked; about 85,000 after the fit |

The engine fits the context to the free memory and reports what it got.

## Model files

The script pins these files. It does not use a file with a different sha256.

| Model | File | Size (bytes) | sha256 |
|---|---|---|---|
| 2B | `Ternary-Bonsai-2-2B-PTQ1_0.gguf` | 429,183,456 | `36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa` |
| 4B | `Ternary-Bonsai-2-4B-PTQ1_0.gguf` | 949,549,920 | `56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60` |
| 9B | `Ternary-Bonsai-2-9B-PTQ1_0.gguf` | 1,997,486,976 | `3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b` |
| 27B | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5,946,648,928 | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |

## Options

| Option | What it does |
|---|---|
| `--dry-run` | Shows each step and each command. It changes nothing. |
| `--yes`, `-y` | Answers yes to each question (for scripts). |
| `--model M` | `2b`, `4b`, `9b` or `27b`. |
| `--ctx N` | The context length in tokens. |
| `--gpu G` | `N` (one card), `N,M` (two cards), `all`, or `cpu`. |
| `--port P` | The server port on 127.0.0.1 (default 8093). |
| `--dir D` | The data folder (default `~/.kindi`). |
| `--name N` | The container name (default `kindi`). |
| `--threads N` | The CPU engine: the number of threads. |
| `--memory M` | The container memory limit, for example `8g`. |
| `--source S` | The Kindi source: a folder, or a `.tar.gz` path or URL (default: the 0.2.0 release). |
| `--uninstall` | Removes the container and the image. The models stay. |
| `--uninstall --purge` | Also removes the data folder. |

## Upgrade, repair and remove

Run the script again to upgrade or repair. It keeps the steps that are already correct: the image (same source),
the model (sha256 already checked) and the pack (made from the same file). It always runs `kindi verify` again and
starts a new container.

```bash
bash install.sh --uninstall          # the container and the image; the models stay in ~/.kindi
bash install.sh --uninstall --purge  # also ~/.kindi
```

## Tested

| Card | Profile | Result |
|---|---|---|
| CPU, 4 threads | cpu: 2B, 24,576 | PASS: verify 4 of 4, 11.7 tokens/s |
| Quadro K420 2 GB | kepler: 2B, 65,536 | PASS: verify 4 of 4, 21.4 tokens/s |
| Quadro K420 1 GB | kepler: 2B, 24,576 | PASS: verify 4 of 4, 20.4 tokens/s |
| Quadro K1200 4 GB | maxwell: 9B, 24,576 | PASS: verify 4 of 4, 23.9 tokens/s |
| 2x Quadro K1200 | k1200x2: 27B, 24,576 | PASS: verify 4 of 4, 13.4 tokens/s |
| GeForce GTX 1080 8 GB, without the toolkit | pascal8: 27B, 65,536 | PASS: verify 4 of 4, 30.4 tokens/s |
| GeForce RTX 3080 10 GB | tc: 2B installed; the 27B selection in a dry run | PASS: verify 4 of 4, 147 tokens/s (2B) |

On the RTX 3080 we tested the installer with the 2B and with a dry run of the 27B selection. The full 27B path
on the 3080 (tensor cores and the 2B draft model, 131,072 tokens) has its own gate on the production cluster.
