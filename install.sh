#!/usr/bin/env bash
# Kindi installer: one command from a bare Linux machine to a verified, running Kindi server.
#
#   curl -fsSL https://amirhusain.com/kindi/install.sh | bash             # install (or upgrade / repair)
#   curl -fsSL https://amirhusain.com/kindi/install.sh | bash -s -- --dry-run
#   bash install.sh --uninstall [--purge]
#
# What it does, in order (every step prints what it is about to do and why):
#   1. checks the machine: Linux x86-64, RAM, disk, the NVIDIA driver and every GPU (or none: the CPU engine);
#   2. checks Docker and the NVIDIA container toolkit; offers to install what is missing (sudo only for that);
#      without the toolkit it passes the GPU devices and driver libraries to the container itself;
#   3. picks the model and the engine settings for the card (docs/CARDS.md);
#   4. builds the Kindi image from the release source;
#   5. downloads the model GGUF from Prism ML on Hugging Face and checks its sha256;
#   6. converts it (kindi convert) and checks the output against llama.cpp's reference (kindi verify);
#   7. starts the server as a service (Docker, restart unless-stopped) on 127.0.0.1 and prints how to use it.
# Run it again to upgrade or repair; the downloads and the converted model are kept. It never asks for or stores a
# password or a key. Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only.
set -euo pipefail

KINDI_VERSION="0.2.0"
SOURCE_URL="https://github.com/ahtx/kindi/archive/refs/tags/v${KINDI_VERSION}.tar.gz"
HF="https://huggingface.co/prism-ml"
# model: GGUF file, sha256 (= the Hugging Face file hash), size in bytes
declare -A GGUF SHA SIZE
GGUF[2b]="Ternary-Bonsai-2-2B-PTQ1_0.gguf";  SHA[2b]="36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa"; SIZE[2b]=429183456
GGUF[4b]="Ternary-Bonsai-2-4B-PTQ1_0.gguf";  SHA[4b]="56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60"; SIZE[4b]=949549920
GGUF[9b]="Ternary-Bonsai-2-9B-PTQ1_0.gguf";  SHA[9b]="3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b"; SIZE[9b]=1997486976
GGUF[27b]="Ternary-Bonsai-2-27B-PTQ1_0.gguf"; SHA[27b]="53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"; SIZE[27b]=5946648928

DIR="${KINDI_HOME:-$HOME/.kindi}"
PORT=8093
NAME=kindi
MODEL=auto
CTX=auto
GPU=auto
SOURCE=""
DRY=0
YES=0
MODE=install
PURGE=0
THREADS=""
MEMORY=""

usage() {
  cat <<EOF
Kindi ${KINDI_VERSION} installer

  install.sh [options]          install, upgrade or repair Kindi
  install.sh --uninstall        remove the Kindi service and image (keeps the models in ${DIR})
  install.sh --uninstall --purge   also remove ${DIR}

Options:
  --dry-run        print every step and command, change nothing
  --yes, -y        answer yes to every question (for scripts)
  --model M        2b | 4b | 9b | 27b (default: the best model for the card)
  --ctx N          context length in tokens (default: the profile's; the engine fits it to the free memory)
  --gpu G          N (one card), N,M (two cards), all, or cpu (default: choose)
  --port P         server port on 127.0.0.1 (default ${PORT})
  --dir D          data folder (default ${DIR})
  --name N         container name (default ${NAME})
  --threads N      CPU engine: the number of threads (default: the cores of one socket)
  --memory M       the container memory limit, for example 8g (default: the profile's)
  --source S       Kindi source: a folder or a .tar.gz URL/path (default: the ${KINDI_VERSION} release)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --yes|-y) YES=1 ;;
    --uninstall) MODE=uninstall ;;
    --purge) PURGE=1 ;;
    --model) MODEL="$2"; shift ;;
    --ctx) CTX="$2"; shift ;;
    --gpu) GPU="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --dir) DIR="$2"; shift ;;
    --name) NAME="$2"; shift ;;
    --source) SOURCE="$2"; shift ;;
    --threads) THREADS="$2"; shift ;;
    --memory) MEMORY="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
  shift
done

# ------------------------------------------------------------------------------------------------ helpers
B=$(printf '\033[1m'); G=$(printf '\033[32m'); Y=$(printf '\033[33m'); R=$(printf '\033[31m'); N=$(printf '\033[0m')
[ -t 1 ] || { B=""; G=""; Y=""; R=""; N=""; }
STEP=0
step() { STEP=$((STEP + 1)); echo; echo "${B}[$STEP] $1${N}"; [ -n "${2:-}" ] && echo "    $2"; return 0; }
info() { echo "    $*"; }
ok() { echo "    ${G}ok${N}  $*"; }
warn() { echo "    ${Y}note${N} $*"; }
die() { echo; echo "${R}stopped:${N} $*"; exit 1; }
exec 3>&1         # run() prints to the terminal also when the command's output goes to /dev/null
run() {   # print, then run unless --dry-run
  echo "    \$ $*" >&3
  [ "$DRY" = 1 ] || "$@"
}
ask() {   # ask "question" -> 0 for yes
  [ "$YES" = 1 ] && { info "$1 yes (--yes)"; return 0; }
  [ "$DRY" = 1 ] && { info "$1 (dry run: no)"; return 1; }
  local a=""
  if [ -r /dev/tty ]; then read -r -p "    $1 [y/N] " a < /dev/tty || true; else info "$1 (no terminal: no)"; fi
  [ "$a" = y ] || [ "$a" = Y ] || [ "$a" = yes ]
}
SUDO=""
need_sudo() {
  [ "$(id -u)" = 0 ] && { SUDO=""; return 0; }
  command -v sudo >/dev/null || die "this step needs root, and sudo is not installed"
  SUDO="sudo"
}
human() { awk -v b="$1" 'BEGIN { split("B KB MB GB TB", u); i = 1; while (b >= 1024 && i < 5) { b /= 1024; i++ } printf "%.1f %s", b, u[i] }'; }

DOCKER="docker"
STATE="$DIR/install.json"

# ------------------------------------------------------------------------------------------------ uninstall
if [ "$MODE" = uninstall ]; then
  step "Remove the Kindi service" "Stops and removes the container '${NAME}'. The models in ${DIR} stay unless --purge."
  command -v docker >/dev/null || die "Docker is not installed: nothing to remove"
  docker info >/dev/null 2>&1 || DOCKER="sudo docker"
  if $DOCKER ps -a --format '{{.Names}}' | grep -x "$NAME" >/dev/null; then run $DOCKER rm -f "$NAME" >/dev/null; ok "container removed"
  else info "no container named ${NAME}"; fi
  img="kindi:${KINDI_VERSION}"                 # only the image this installer built (other Kindi images stay)
  [ -f "$STATE" ] && img=$(sed -n 's/.*"image": "\([^"]*\)".*/\1/p' "$STATE" | sed -n 1p)
  if $DOCKER image inspect "$img" >/dev/null 2>&1 && [ -z "$($DOCKER ps -a --filter ancestor="$img" --format '{{.Names}}')" ] \
     && ask "Remove the image ${img}?"; then if run $DOCKER rmi "$img" >/dev/null 2>&1; then ok "image removed"; else warn "could not remove the image ${img}"; fi; fi
  if [ "$PURGE" = 1 ] && [ -d "$DIR" ]; then
    info "${DIR} holds $(du -sh "$DIR" 2>/dev/null | cut -f1): the models, the converted packs and the cache."
    if ask "Delete ${DIR}?"; then run rm -rf "$DIR"; ok "deleted"; fi
  fi
  echo; echo "${G}Kindi is removed.${N}"
  exit 0
fi

echo "${B}Kindi ${KINDI_VERSION} installer${N}$([ "$DRY" = 1 ] && echo " (dry run: nothing changes)")"
PREV=""
[ -f "$STATE" ] && PREV=$(cat "$STATE") && info "found an earlier install (${STATE}): this run upgrades or repairs it"

# ------------------------------------------------------------------------------------------------ 1. machine
step "Check the machine" "Kindi runs on 64-bit Linux, on an NVIDIA GPU from the Quadro K420 (Kepler) up, or on the CPU."
[ "$(uname -s)" = Linux ] || die "Kindi needs Linux (found $(uname -s))"
[ "$(uname -m)" = x86_64 ] || die "Kindi needs an x86-64 CPU (found $(uname -m))"
OS_ID=$(. /etc/os-release 2>/dev/null && echo "${ID:-unknown}") || OS_ID=unknown
RAM_MB=$(awk '/MemTotal/ {print int($2 / 1024)}' /proc/meminfo)
AVAIL_MB=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
CORES=$(nproc)
[ "$DRY" = 1 ] || mkdir -p "$DIR" 2>/dev/null || die "cannot create ${DIR}"
d="$DIR"; while [ ! -d "$d" ]; do d=$(dirname "$d"); done      # the nearest folder that exists (a dry run makes none)
FREE_DISK=$(df -Pk "$d" | awk 'NR == 2 {print $4 * 1024}')
ok "Linux ($OS_ID), ${CORES} CPU threads, $((RAM_MB / 1024)) GB RAM, $(human "$FREE_DISK") free for ${DIR}"

# GPUs: index, name, memory (MiB), compute capability
declare -a GIDX GNAME GMEM GCC
DRIVER=""
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n 1p)
  if out=$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null); then :; else
    out=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | sed 's/$/, ?/'); fi
  while IFS=',' read -r i n m c; do
    n=$(echo "$n" | sed 's/^ *//; s/ *$//'); c=$(echo "$c" | tr -d ' ')
    if [ "$c" = "?" ] || [ -z "$c" ]; then   # older drivers (Kepler's 470) do not report it: from the name
      case "$n" in
        *K420*|*K600*|*K2000*|*K4000*|*K5000*|*K6000*|*"GTX 6"*|*"GTX 7"[0-4]*|*"GT 7"*) c=3.0 ;;
        *K620*|*K1200*|*K2200*|*"GTX 75"*|*"GTX 9"*|*M[0-9]*) c=5.0 ;;
        *"GTX 10"*|*"Quadro P"*|*"TITAN X"*|*Tesla\ P*) c=6.1 ;;
        *"RTX 20"*|*"GTX 16"*|*"Quadro RTX"*|*T4*) c=7.5 ;;
        *"RTX 30"*|*A[0-9]*) c=8.6 ;;
        *"RTX 40"*|*L4*) c=8.9 ;;
        *) c=6.1 ;;
      esac
    fi
    GIDX+=("$(echo "$i" | tr -d ' ')"); GNAME+=("$n"); GMEM+=("$(echo "$m" | tr -d ' ')"); GCC+=("$c")
  done <<< "$out"
fi
NG=${#GIDX[@]}
if [ "$NG" = 0 ]; then
  warn "no NVIDIA GPU found (or no driver): Kindi will use its CPU engine"
else
  for k in $(seq 0 $((NG - 1))); do ok "GPU ${GIDX[$k]}: ${GNAME[$k]}, ${GMEM[$k]} MiB, sm_${GCC[$k]/./}"; done
  info "NVIDIA driver ${DRIVER}"
  case "${GCC[0]}" in 3.*) [ "${DRIVER%%.*}" -le 470 ] || warn "Kepler cards need the NVIDIA 470 driver branch (found ${DRIVER})" ;; esac
fi

# ------------------------------------------------------------------------------------------------ 2. docker
step "Check Docker and GPU access" "Kindi runs in a Docker container with CUDA 10.2 inside, so the host needs only the driver."
if ! command -v docker >/dev/null; then
  warn "Docker is not installed"
  case "$OS_ID" in
    ubuntu|debian|linuxmint|pop)
      if ask "Install Docker (apt package docker.io) and add ${USER} to the docker group?"; then
        need_sudo; run $SUDO apt-get update -qq; run $SUDO apt-get install -y -qq docker.io
        run $SUDO usermod -aG docker "$USER"; DOCKER="$SUDO docker"
      else die "Kindi needs Docker: https://docs.docker.com/engine/install/"; fi ;;
    *) die "install Docker for ${OS_ID} (https://docs.docker.com/engine/install/), then run this again" ;;
  esac
fi
if [ "$DRY" = 0 ] && ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1 || { need_sudo; $SUDO docker info >/dev/null 2>&1; }; then
    DOCKER="sudo docker"; warn "your user cannot use Docker yet (log out and in after 'usermod -aG docker'); using sudo docker"
  else die "Docker does not answer (is the daemon running? 'sudo systemctl start docker')"; fi
fi
ok "Docker $(docker --version 2>/dev/null | awk '{print $3}' | tr -d ,)"

GPU_MODE=none
if [ "$NG" -gt 0 ] && [ "$GPU" != cpu ]; then
  if $DOCKER info 2>/dev/null | grep -i 'runtimes.*nvidia' >/dev/null || command -v nvidia-ctk >/dev/null || command -v nvidia-container-toolkit >/dev/null; then
    GPU_MODE=toolkit; ok "NVIDIA container toolkit found: the container gets the GPUs with --gpus"
  else
    warn "the NVIDIA container toolkit is not installed"
    if [ "$OS_ID" = ubuntu ] || [ "$OS_ID" = debian ]; then
      if ask "Install the NVIDIA container toolkit (NVIDIA's apt repository)?"; then
        need_sudo
        run sh -c "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        run sh -c "curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null"
        run $SUDO apt-get update -qq; run $SUDO apt-get install -y -qq nvidia-container-toolkit
        run $SUDO nvidia-ctk runtime configure --runtime=docker; run $SUDO systemctl restart docker
        GPU_MODE=toolkit
      fi
    fi
    if [ "$GPU_MODE" = none ]; then
      GPU_MODE=passthrough; info "Kindi passes the GPU devices and the driver libraries to the container itself instead"
    fi
  fi
fi

# ------------------------------------------------------------------------------------------------ 3. profile
step "Choose the model and the settings" "From docs/CARDS.md: the largest model that runs well on this card, and its engine settings."
ENV=()            # -e settings for the server
PACKS=()          # models to convert (the first is the served one)
CARDS=""          # CUDA_VISIBLE_DEVICES
MEM=""            # container memory limit
CPUS=""           # container CPU limit (the CPU engine)
PROFILE=""
pick_card() {     # the GPU with the most memory (first on a tie)
  local best=0
  for k in $(seq 0 $((NG - 1))); do [ "${GMEM[$k]}" -gt "${GMEM[$best]}" ] && best=$k; done
  echo $best
}
if [ "$NG" = 0 ] || [ "$GPU" = cpu ]; then
  PROFILE=cpu; M=${MODEL/auto/2b}
  th=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | awk -F, '$2 == 0' | sort -u | wc -l); [ "$th" -gt 0 ] || th=$CORES
  [ -n "$THREADS" ] && th=$THREADS
  ENV+=(-e KINDI_DEVICE=cpu -e OMP_NUM_THREADS="$th" -e OMP_PROC_BIND=close -e OMP_PLACES=cores)
  C=${CTX/auto/24576}; MEM=8g; CPUS="$th"; info "CPU engine, ${th} threads"
else
  if [ "$GPU" = auto ] || [ "$GPU" = all ]; then
    # two cards of 3.8 GB or more below sm_61: the 27B on both (tensor parallel)
    if [ "$NG" -ge 2 ] && [ "${GMEM[0]}" -ge 3800 ] && [ "${GMEM[1]}" -ge 3800 ] && [ "${GCC[0]/./}" -lt 61 ] && [ "${GCC[1]/./}" -lt 61 ] \
       && { [ "$MODEL" = auto ] || [ "$MODEL" = 27b ]; }; then
      CARDS="${GIDX[0]},${GIDX[1]}"
    else
      CARDS="${GIDX[$(pick_card)]}"
    fi
  else CARDS="$GPU"; fi
  k=0; for j in $(seq 0 $((NG - 1))); do [ "${GIDX[$j]}" = "${CARDS%%,*}" ] && k=$j; done
  V=${GMEM[$k]}; CC=${GCC[$k]}; ccn=${CC/./}
  if [[ "$CARDS" == *,* ]]; then
    PROFILE=k1200x2; M=${MODEL/auto/27b}; C=${CTX/auto/24576}; MEM=12g
    ENV+=(-e KINDI_CARDS=2 -e KINDI_REPEAT_STOP=1 -e KINDI_CHECKPOINT_BUDGET_MB=4096)
  elif [ "$ccn" -ge 80 ] && [ "$V" -ge 9500 ]; then
    PROFILE=tc; M=${MODEL/auto/27b}; C=${CTX/auto/131072}; MEM=14g
    ENV+=(-e KINDI_CARDS=1 -e KINDI_REPEAT_STOP=1 -e KINDI_CHECKPOINT_BUDGET_MB=1280 -e KINDI_PARK_MAX=1
          -e KINDI_DENSE_SPACING=2048 -e KINDI_PREFIX_MAX=2 -e KINDI_MAX_TOKENS=65536)
    if [ "$M" = 27b ]; then   # the profile of the RTX 3080 cluster servers (docs/CARDS.md)
      ENV+=(-e KINDI_TC=1 -e KINDI_FD=1 -e KINDI_ATT_TC=1 -e KINDI_CKPT_VRAM=1
            -e KINDI_SPEC=draft2b -e KINDI_SPEC_K=3 -e KINDI_SPEC_PACK=/data/pack-2b -e KINDI_SPEC_WINDOW=8192
            -e KINDI_TC_TMAX=4 -e KINDI_MEM_RESERVE_MB=160)
      PACKS+=(2b)
    fi
  elif [ "$ccn" -ge 61 ] && [ "$V" -ge 7500 ]; then
    PROFILE=pascal8; M=${MODEL/auto/27b}; C=${CTX/auto/65536}; MEM=14g
    ENV+=(-e KINDI_CARDS=1 -e KINDI_REPEAT_STOP=1 -e KINDI_MAX_TOKENS=65536)
  elif [ "$ccn" -lt 50 ]; then
    PROFILE=kepler; M=${MODEL/auto/2b}
    if [ "$V" -ge 1500 ]; then C=${CTX/auto/65536}; else C=${CTX/auto/24576}; fi
  else
    PROFILE=maxwell; C=${CTX/auto/24576}
    if [ "$V" -ge 3800 ]; then M=${MODEL/auto/9b}; elif [ "$V" -ge 1900 ]; then M=${MODEL/auto/4b}; else M=${MODEL/auto/2b}; fi
  fi
  ENV+=(-e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="$CARDS")   # the card numbers of nvidia-smi
  # blocking sync frees a CPU core per busy server but costs ~2.5% decode (GTX 1080): only the tc profile (many
  # servers per host on the cluster) keeps it; one server on its own card waits in a loop
  [ "$PROFILE" = tc ] || ENV+=(-e KINDI_BLOCKING_SYNC=0)
fi
[ -n "${GGUF[$M]:-}" ] || die "unknown model '${M}' (2b, 4b, 9b or 27b)"
PACKS=("$M" ${PACKS[@]+"${PACKS[@]}"})
OWN=0   # a re-run: this installer's own server still holds RAM now; it stops before the new load
if [ "$DRY" != 1 ] && $DOCKER ps --format '{{.Names}}' | grep -x "$NAME" >/dev/null; then OWN=1; fi
if [ "$M" = 27b ] && [ "$PROFILE" = k1200x2 ] && [ "$AVAIL_MB" -lt 5000 ] && [ "$OWN" = 1 ]; then
  warn "the 27B on two cards needs about 5 GB of free RAM while it loads; ${NAME} holds some now and stops first"
elif [ "$M" = 27b ] && [ "$PROFILE" = k1200x2 ] && [ "$AVAIL_MB" -lt 5000 ] && [ "${KINDI_HOST_RAM_CHECK:-1}" = 1 ]; then
  die "the 27B on two cards needs about 5 GB of free RAM while it loads (free now: $((AVAIL_MB / 1024)) GB of $((RAM_MB / 1024)) GB).
  Stop the other model servers and large programs, then run the installer again (KINDI_HOST_RAM_CHECK=0 skips this check)."
elif [ "$M" = 27b ] && [ "$PROFILE" != k1200x2 ] && [ "$AVAIL_MB" -lt 14000 ]; then
  warn "the 27B needs about 14 GB of free RAM for its first load (free now: $((AVAIL_MB / 1024)) GB of $((RAM_MB / 1024)) GB)"
fi
[ -n "$MEMORY" ] && MEM="$MEMORY"
need=0; for p in ${PACKS[@]+"${PACKS[@]}"}; do need=$((need + ${SIZE[$p]} * 3)); done
[ "$FREE_DISK" -gt "$need" ] || warn "the model, its pack and its cache need about $(human $need) of disk (free: $(human "$FREE_DISK"))"
MODEL_NAME="bonsai2-${M}"
ok "model Ternary-Bonsai-2-${M^^} (${MODEL_NAME}), context ${C} tokens$([ -n "$CARDS" ] && echo ", GPU(s) ${CARDS}"), profile ${PROFILE}"
info "(the engine fits the context to the free memory and reports what it got)"

# GPU arguments for docker run
GARGS=()
if [ "$GPU_MODE" = toolkit ]; then GARGS=(--gpus all)
elif [ "$GPU_MODE" = passthrough ]; then
  for d in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do [ -e "$d" ] && GARGS+=(--device "$d"); done
  L=$(ldconfig -p | awk '/libcuda\.so\.1 .*x86-64/ && !f {print $NF; f = 1}')
  [ -n "$L" ] || die "the NVIDIA driver library libcuda.so.1 is not on this host"
  GARGS+=(-v "$(readlink -f "$L"):/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro")
  for lib in libnvidia-ptxjitcompiler.so.1 libnvidia-nvvm.so.4; do
    p=$(ldconfig -p | awk -v l="$lib" '$1 == l && /x86-64/ && !f {print $NF; f = 1}')
    [ -n "$p" ] && GARGS+=(-v "$(readlink -f "$p"):/usr/lib/x86_64-linux-gnu/${lib}:ro")
  done
fi
UIDGID="$(id -u):$(id -g)"

# ------------------------------------------------------------------------------------------------ 4. image
IMAGE="kindi:${KINDI_VERSION}"
step "Build the Kindi image (${IMAGE})" "The image holds CUDA 10.2, CuPy 9.6 and Kindi. The first build downloads about 3 GB and takes 10-15 minutes."
src_id=""
if [ -n "$SOURCE" ] && [ -d "$SOURCE" ]; then SRC="$SOURCE"
else
  SRC="$DIR/src-${KINDI_VERSION}"
  tarball="${SOURCE:-$SOURCE_URL}"
  if [ ! -f "$SRC/docker/Dockerfile" ]; then
    run mkdir -p "$SRC"
    if [ -f "$tarball" ]; then run tar -xzf "$tarball" -C "$SRC" --strip-components=1
    else run sh -c "curl -fsSL '$tarball' | tar -xz -C '$SRC' --strip-components=1"; fi
  fi
fi
[ "$DRY" = 1 ] || src_id=$(cd "$SRC" && find kindi docker -type f | sort | xargs cat | sha256sum | cut -c1-12)
have=$($DOCKER image inspect -f '{{index .Config.Labels "kindi.source"}}' "$IMAGE" 2>/dev/null || true)
if [ -n "$have" ] && [ "$have" = "$src_id" ]; then ok "${IMAGE} is up to date (source ${src_id})"
else
  [ -n "$have" ] && info "the image is from other source (${have}): upgrading"
  run $DOCKER build -q --label "kindi.source=${src_id}" -t "$IMAGE" -f "$SRC/docker/Dockerfile" "$SRC" >/dev/null
  ok "built ${IMAGE}"
fi

# ------------------------------------------------------------------------------------------------ 5. download
step "Download the model" "From Prism ML on Hugging Face. The sha256 of each file must match the value in this script."
[ "$DRY" = 1 ] || mkdir -p "$DIR/models"
for p in ${PACKS[@]+"${PACKS[@]}"}; do
  f="$DIR/models/${GGUF[$p]}"
  if [ -f "$f" ] && [ "$(stat -c %s "$f")" = "${SIZE[$p]}" ] && [ -f "$f.sha256-ok" ]; then ok "${GGUF[$p]} is here (checked before)"; continue; fi
  info "${GGUF[$p]}: $(human "${SIZE[$p]}")"
  run curl -fL --retry 5 --retry-delay 5 -C - --progress-bar -o "$f.part" "$HF/Ternary-Bonsai-2-${p^^}-gguf/resolve/main/${GGUF[$p]}"
  if [ "$DRY" = 0 ]; then
    got=$(sha256sum "$f.part" | cut -d' ' -f1)
    [ "$got" = "${SHA[$p]}" ] || { rm -f "$f.part"; die "${GGUF[$p]}: sha256 ${got} is not the expected ${SHA[$p]} (deleted; run again)"; }
    mv "$f.part" "$f"; touch "$f.sha256-ok"
  fi
  ok "${GGUF[$p]}: sha256 ${SHA[$p]:0:16}... matches"
done

# ------------------------------------------------------------------------------------------------ 6. convert + verify
step "Convert and verify the model" "kindi convert builds Kindi's weight pack; kindi verify checks the output against llama.cpp, token for token."
STATE_DIR="state-${M}"
BASE=(-u "$UIDGID" -v "$DIR:/data" -e HOME=/tmp -e "CUPY_CACHE_DIR=/data/${STATE_DIR}/cupy" -e "CUDA_CACHE_PATH=/data/${STATE_DIR}/cudacache"
      -e "KINDI_STATE_DIR=/data/${STATE_DIR}" -e KINDI_CPU_CACHE=/data/cpu)
for p in ${PACKS[@]+"${PACKS[@]}"}; do
  pk="$DIR/pack-$p"
  if [ -f "$pk/kindi_pack.json" ] && grep -q "${SHA[$p]}" "$pk/kindi_pack.json"; then ok "pack-$p is up to date"; continue; fi
  run $DOCKER run --rm ${BASE[@]+"${BASE[@]}"} "$IMAGE" convert "/data/models/${GGUF[$p]}" --out "/data/pack-$p" --force
  ok "pack-$p converted"
done
if $DOCKER ps --format '{{.Names}}' | grep -x "$NAME" >/dev/null; then
  info "the running '${NAME}' holds the GPU: stopping it for the check (it is started again below)"
  run $DOCKER stop "$NAME" >/dev/null
fi
[ "$M" = 27b ] && [ "$PROFILE" != k1200x2 ] && info "the first 27B start writes its 5-trit weights into the pack (5.6 GB, up to 15 minutes)"
VERIFY_OUT="$DIR/verify.log"
SPEED="?"
if [ "$DRY" = 1 ]; then info "\$ $DOCKER run --rm ${BASE[*]} ${GARGS[*]:-} ${ENV[*]:-} -e KINDI_PACK=/data/pack-$M $IMAGE verify"
else
  VX=(); [ -n "$CPUS" ] && VX=(--cpus "$CPUS")
  if $DOCKER run --rm ${VX[@]+"${VX[@]}"} ${BASE[@]+"${BASE[@]}"} ${GARGS[@]+"${GARGS[@]}"} ${ENV[@]+"${ENV[@]}"} -e "KINDI_PACK=/data/pack-$M" -e KINDI_HEAD=exact "$IMAGE" verify > "$VERIFY_OUT" 2>&1; then
    grep -E "PASS|FAIL" "$VERIFY_OUT" | sed 's/^\[[^]]*\] /    /' | sed -n 1,6p
    SPEED=$(grep -o '[0-9.]* tokens/s over the run' "$VERIFY_OUT" | awk '{print $1}')
    ok "token-exact with llama.cpp (${SPEED} tokens/s over the check, prompt tokens included)"
  else
    tail -15 "$VERIFY_OUT" | sed 's/^/    /'
    die "kindi verify did not pass (log: ${VERIFY_OUT}). The server is not started."
  fi
fi

# ------------------------------------------------------------------------------------------------ 7. service
step "Start the server" "A Docker container with restart=unless-stopped: it starts again after a reboot. Only this machine can reach it (127.0.0.1)."
$DOCKER ps -a --format '{{.Names}}' | grep -x "$NAME" >/dev/null && run $DOCKER rm -f "$NAME" >/dev/null
SARGS=(-d --name "$NAME" --restart unless-stopped ${BASE[@]+"${BASE[@]}"} ${GARGS[@]+"${GARGS[@]}"} ${ENV[@]+"${ENV[@]}"} -p "127.0.0.1:${PORT}:8093"
       -e "KINDI_PACK=/data/pack-$M" -e "KINDI_CTX=$C" -e KINDI_HEAD=exact -e "KINDI_MODEL_NAME=${MODEL_NAME}" -e KINDI_HOST=0.0.0.0)
[ -n "$MEM" ] && SARGS+=(--memory "$MEM")
[ -n "$CPUS" ] && SARGS+=(--cpus "$CPUS")
run $DOCKER run ${SARGS[@]+"${SARGS[@]}"} "$IMAGE" serve >/dev/null
HEALTH=""
if [ "$DRY" = 0 ]; then
  printf "    waiting for the server"
  for i in $(seq 1 450); do
    HEALTH=$(curl -fs -m 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null || true)
    echo "$HEALTH" | grep -q '"ok":true' && break
    $DOCKER ps --format '{{.Names}}' | grep -x "$NAME" >/dev/null || { echo; $DOCKER logs --tail 20 "$NAME" 2>&1 | sed 's/^/    /'; die "the server stopped (see above)"; }
    [ $((i % 10)) = 0 ] && printf "."
    sleep 2
  done
  echo
  echo "$HEALTH" | grep -q '"ok":true' || die "the server did not answer in 15 minutes ($DOCKER logs ${NAME})"
fi
GOT_CTX=$(echo "$HEALTH" | sed -n 's/.*"ctx":\([0-9]*\).*/\1/p'); GOT_CTX=${GOT_CTX:-$C}
GOT_VER=$(echo "$HEALTH" | sed -n 's/.*"kindi": *"\([^"]*\)".*/\1/p'); GOT_VER=${GOT_VER:-$KINDI_VERSION}
GOT_SHA=$(echo "$HEALTH" | sed -n 's/.*"commit": *"\([0-9a-f]*\)".*/\1/p')
ok "the server answers on http://127.0.0.1:${PORT}"

[ "$DRY" = 1 ] || cat > "$STATE" <<EOF
{"version": "${KINDI_VERSION}", "source": "${src_id}", "profile": "${PROFILE}", "model": "${M}", "ctx": ${GOT_CTX},
 "container": "${NAME}", "image": "${IMAGE}", "port": ${PORT}, "gpus": "${CARDS}", "gpu_access": "${GPU_MODE}",
 "installed": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF

# ------------------------------------------------------------------------------------------------ done
URL="http://127.0.0.1:${PORT}/v1"
cat <<EOF

${G}${B}Kindi is running.${N}

  Server     ${URL}   (OpenAI-compatible; status: http://127.0.0.1:${PORT}/health)
  Kindi      ${GOT_VER}${GOT_SHA:+ (commit ${GOT_SHA})}
  Model      ${MODEL_NAME} (Ternary-Bonsai-2-${M^^})
  Context    ${GOT_CTX} tokens
  Speed      ${SPEED} tokens/s in the check
  Service    docker container '${NAME}' (restart unless-stopped); logs: ${DOCKER} logs -f ${NAME}

Try it:
  curl ${URL}/chat/completions -H 'Content-Type: application/json' \\
    -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

OpenAI clients:
  export OPENAI_BASE_URL=${URL}  OPENAI_API_KEY=none

pi (~/.pi/agent/models.json):
  {"providers": {"kindi": {"baseUrl": "${URL}", "api": "openai-completions", "apiKey": "none",
    "compat": {"supportsDeveloperRole": false, "supportsReasoningEffort": false,
               "thinkingFormat": "qwen-chat-template", "thinkingTokenBudgetField": "thinking_budget_tokens"},
    "models": [{"id": "${MODEL_NAME}", "name": "${MODEL_NAME}", "reasoning": true, "input": ["text"],
                "contextWindow": ${GOT_CTX}, "maxTokens": 4096}]}}}

Run this installer again to upgrade or repair. Remove Kindi: bash install.sh --uninstall [--purge]
EOF
