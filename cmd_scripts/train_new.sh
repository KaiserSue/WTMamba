#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_root"

fail() { echo "train_new.sh: $*" >&2; exit 2; }
help() {
    cat <<'HELP'
Usage: bash cmd_scripts/train_new.sh --data-dir ROOT [options]
  --dataset Imagenet|cifar100|tinyimagenet   (default Imagenet; case sensitive)
  --data-dir ROOT                          Required dataset root
  --gpus N                                 Visible CUDA GPUs, default 1
  --model NAME                             WaveletTinyViM_S/B/L or TinyViM_S/B/L
  --batch-size N                           Per GPU, default 256 / 200 / 200
  --epochs N                               Default 300
  --warmup-epochs N                        Default 5 / 30 / 30; less than epochs
  --distillation-type none|hard|soft        Default none
  --teacher-path PATH_OR_HTTPS              Required for hard/soft training
  --eval --resume CHECKPOINT                Evaluation; no teacher required
Training options below override launcher defaults. CUDA_VISIBLE_DEVICES is honored.
HELP
    python main.py --help
}

parse_launch_args() {
    dataset=Imagenet
    data_dir=''
    gpus=1
    training_argv=()
    for argument in "$@"; do
        if [[ "$argument" == --help || "$argument" == -h ]]; then help; exit 0; fi
    done
    while (($#)); do
        case "$1" in
            --dataset|--data-dir|--gpus)
                (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || fail "$1 requires a value"
                case "$1" in
                    --dataset) dataset=$2 ;;
                    --data-dir) [[ -z "$data_dir" ]] || fail 'duplicate --data-dir'; data_dir=$2 ;;
                    --gpus) gpus=$2 ;;
                esac
                shift 2 ;;
            --dataset=*|--data-dir=*|--gpus=*)
                option=${1%%=*}; value=${1#*=}
                [[ -n "$value" ]] || fail "$option requires a value"
                case "$option" in
                    --dataset) dataset=$value ;;
                    --data-dir) [[ -z "$data_dir" ]] || fail 'duplicate --data-dir'; data_dir=$value ;;
                    --gpus) gpus=$value ;;
                esac
                shift ;;
            --data-path|--data-path=*|--data-set|--data-set=*|--world_size|--world_size=*|--world-size|--world-size=*|--dist_url|--dist_url=*)
                fail "$1 conflicts with dataset/GPU routing" ;;
            --device)
                (($# >= 2)) && [[ "$2" == cuda ]] || fail '--device must be cuda'
                shift 2 ;;
            --device=*) [[ "$1" == --device=cuda ]] || fail '--device must be cuda'; shift ;;
            *) training_argv+=("$1"); shift ;;
        esac
    done
    case "$dataset" in
        Imagenet) entry=(main.py); data_set=IMNET; batch=256; warmup=5 ;;
        cifar100) entry=(--module supportcode.main_cifar100); data_set=CIFAR; batch=200; warmup=30 ;;
        tinyimagenet) entry=(--module supportcode.main_tinyimagenet); data_set=IMNET; batch=200; warmup=30 ;;
        *) fail "unknown dataset: $dataset" ;;
    esac
    [[ -n "$data_dir" ]] || fail '--data-dir is required'
    [[ "$gpus" =~ ^[1-9][0-9]*$ ]] && ((${#gpus} <= 4)) || fail '--gpus must be a positive integer'
    data_dir=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$data_dir")
    [[ -d "$data_dir" && -r "$data_dir" ]] || fail "data-dir is not a readable directory: $data_dir"
}

launch_training() {
    local visible
    visible=$(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')
    ((gpus <= visible)) || fail "--gpus=$gpus exceeds available CUDA devices ($visible)"
    exec python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$gpus" \
        "${entry[@]}" --model WaveletTinyViM_S --data-set "$data_set" --data-path "$data_dir" \
        --batch-size "$batch" --epochs 300 --warmup-epochs "$warmup" \
        --distillation-type none --teacher-path '' --device cuda "${training_argv[@]}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    parse_launch_args "$@"
    launch_training
fi
