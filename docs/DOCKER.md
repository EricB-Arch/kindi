# Running Kindi with Docker

To install Kindi with one command (the image, the model and a service), see [INSTALL.md](INSTALL.md). This page
tells how to do the same steps by hand.

The image (`docker/Dockerfile`) is Ubuntu 18.04 with the CUDA 10.2 toolkit, Python 3.6, CuPy 9.6, FastAPI and
Kindi. Everything Kindi reads or writes lives in one volume mounted at `/data`:

| path in the container | |
|---|---|
| `/data/pack` | the converted model (`kindi convert`) and `head_clusters.npz` (`kindi fast-head`) |
| `/data/state` | kernel cache, saved prompt openings, head mode |
| `/data/kindi.conf` | optional config file (pass `-e KINDI_CONFIG=/data/kindi.conf`) |

## Host prerequisites

1. NVIDIA driver **470** (`nvidia-smi` shows `Driver Version: 470.x`). Newer branches do not support Kepler; on a
   distribution that would upgrade it, hold the package.
2. Docker and the **NVIDIA Container Toolkit** (`docker run --rm --gpus all ubuntu nvidia-smi` must show the K420).

## Build

```sh
docker build -t kindi:0.1.0 -f docker/Dockerfile .
```

## Commands

```sh
D="docker run --rm --gpus all -u $(id -u):$(id -g) -v $HOME/.kindi:/data kindi:0.1.0"
$D info
$D convert /data/Ternary-Bonsai-2-2B-PTQ1_0.gguf
$D verify
$D fast-head
$D bench
```

## The server

```sh
docker run -d --name kindi --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
    -p 127.0.0.1:8093:8093 -v $HOME/.kindi:/data kindi:0.1.0
docker logs -f kindi                                  # request log
docker exec kindi python3 -m kindi head fast          # switch the output head
```

The option `-p 127.0.0.1:8093:8093` keeps the server on this machine. To serve the network, use `-p 8093:8093` and
set an API key: `-e KINDI_API_KEY=<long random string>`.

Options go in as environment variables (`-e KINDI_CTX=16384`) or in `/data/kindi.conf`.
