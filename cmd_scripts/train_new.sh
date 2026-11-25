#!/usr/bin/env bash
set -euo pipefail

# 修改以下配置后，直接运行 bash cmd_scripts/train_new.sh。
python_bin="$HOME/anaconda3/envs/Experiment/bin/python"
dataset=cifar100  # Imagenet、cifar100 或 tinyimagenet
data_dir="$HOME/dl_project/data/cifar100"
gpus=1  # 使用 CUDA_VISIBLE_DEVICES 可见的 GPU
model=WaveletTinyViM_S
batch_size=128  # 每张 GPU 的 batch size
epochs=300
warmup_epochs=30
distillation_type=none  # none、hard 或 soft
teacher_path=''  # hard / soft 蒸馏训练时填写权重路径或 HTTPS URL

training_args=(
    # --output-dir checkpoints
    # --resume /path/to/checkpoint.pth
    # --eval  # 仅评估时与 --resume 一起启用
)

fail() { echo "train_new.sh: $*" >&2; exit 2; }

launch_training() {
    (($# == 0)) || fail '请在脚本顶部修改训练配置，然后直接运行 bash cmd_scripts/train_new.sh'

    local repo_root data_set visible
    local -a entry
    repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
    cd -- "$repo_root"

    case "$dataset" in
        Imagenet) entry=(main.py); data_set=IMNET ;;
        cifar100) entry=(--module supportcode.main_cifar100); data_set=CIFAR ;;
        tinyimagenet) entry=(--module supportcode.main_tinyimagenet); data_set=IMNET ;;
        *) fail "不支持的 dataset：$dataset" ;;
    esac
    [[ -n "$data_dir" && -d "$data_dir" && -r "$data_dir" ]] || fail "data_dir 不是可读目录：$data_dir"
    [[ "$gpus" =~ ^[1-9][0-9]*$ ]] && ((${#gpus} <= 4)) || fail 'gpus 必须是正整数'
    command -v "$python_bin" >/dev/null || fail "找不到 python_bin：$python_bin"

    visible=$("$python_bin" -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')
    ((gpus <= visible)) || fail "gpus=$gpus，当前可用 CUDA 设备数为 $visible"

    exec "$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$gpus" \
        "${entry[@]}" --model "$model" --data-set "$data_set" --data-path "$data_dir" \
        --batch-size "$batch_size" --epochs "$epochs" --warmup-epochs "$warmup_epochs" \
        --distillation-type "$distillation_type" --teacher-path "$teacher_path" --device cuda \
        "${training_args[@]}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    launch_training "$@"
fi
