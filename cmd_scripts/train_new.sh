#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
dataset=cifar100  # Imagenet、cifar100 或 tinyimagenet
data_dir="/home/apulis-dev/adhub/CIFAR100/0.1/cifar100"
output_dir=checkpoints  # 输出根目录，相对路径以仓库根目录为基准
gpus=1  # 使用 CUDA_VISIBLE_DEVICES 可见的 GPU
model=WaveletTinyViM_S
batch_size=128  # 每张 GPU 的 batch size
epochs=300
warmup_epochs=20
distillation_type=none  # none、hard 或 soft
teacher_path=''  # hard / soft 蒸馏训练时填写权重路径或 HTTPS URL

# 额外训练参数。命令行传入的参数会放在最后，可覆盖脚本中的同名选项。
training_args=(
    # --lr 0.001
    # --resume /path/to/checkpoint.pth
    # --eval  # 仅评估时与 --resume 一起启用
)

fail() { echo "train_new.sh: $*" >&2; exit 2; }

launch_training() {
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
        "${entry[@]}" --model "$model" --data-set "$data_set" --data-path "$data_dir" --output-dir "$output_dir" \
        --batch-size "$batch_size" --epochs "$epochs" --warmup-epochs "$warmup_epochs" \
        --distillation-type "$distillation_type" --teacher-path "$teacher_path" --device cuda \
        "${training_args[@]}" "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    launch_training "$@"
fi
