<div align="center">
<h1>TinyViM </h1>
<h3>TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba</h3>

[Xiaowen Ma](https://scholar.google.com/citations?hl=zh-CN&user=UXj8Q6kAAAAJ),  [Zhenliang Ni](https://scholar.google.com/citations?user=2urTmpkAAAAJ&hl=zh-CN&oi=sra), [Xinghao Chen](https://scholar.google.com/citations?user=tuGWUVIAAAAJ&hl=zh-CN&oi=ao)

Huawei Noah’s Ark Lab

 [[Paper Link](https://arxiv.org/abs/2411.17473)]

</div>

## 🔥 News

- **`2025/06/26`**: **TinyViM is accepted by ICCV'25 and the checkpoints are available.**
- **`2024/11/29`**: **Code is open.**
- **`2024/11/27`**: **TinyViM is available at [Arxiv](https://arxiv.org/abs/2411.17473).**

## 📷 Introduction

<img src="fig/comp.png"  />

<img src="fig/whole.png"  />

We build a series of tiny hybrid vision Mamba called **TinyViM** by integrating mobile-friendly convolution and efficient Laplace mixer. The proposed TinyViM achieves impressive performance on several downstream tasks including image classification, semantic segmentation, object detection and instance segmentation. In particular, TinyViM outperforms Convolution, Transformer and Mamba-based models with similar scales, and the throughput is about 2-3 times higher than that of other Mamba-based models.



## 🏆 Performance

### 1️⃣ Classification

| Model |                           Type                           | Params (M) | GMACs | Throughput (im/s) | Top-1(300e/1000e) |
| :------------: | :----------------------------------------------------------: | ---------- | :-------: | :-------: | --------- |
|   TinyViM-S   | CNN-Mamba | 5.6     |    0.9    |    2563    |   79.2/[80.3](https://drive.google.com/drive/folders/1cLgX_Mlk9W5_nMtgwmLC71ap-RlvT7eW?usp=sharing)   |
|  TinyViM-B  | CNN-Mamba | 11.0     |    1.5    |    1851    |   [81.2](https://drive.google.com/drive/folders/1eo9RBT186E11pJFOO8uOi9S5GoR0PosS?usp=drive_link)/[82.1](https://drive.google.com/drive/folders/1nZAz0nt8Ztk7YqXbrXW8BZyPZZ49HmVx?usp=sharing)   |
|   TinyViM-L   | CNN-Mamba | 31.7    |    4.7    |    843    |   [83.3](https://drive.google.com/drive/folders/1BtxDPRyll8FVbZfbdjH-oDJU7dYKaSfY?usp=sharing)/[83.6](https://drive.google.com/drive/folders/1-nfZiEMix8Xd625M2UVFXrujIIV-K2_2?usp=sharing)   |

**We have provided the checkpoints and training logs in the table above!**

### 2️⃣  Detection & Instance Segmentation

| Model |                           Head                           | AP-box | AP-mask |
| :------------: | :----------------------------------------------------------: | :----------: | ---------- |
|     TinyViM-B     | Mask RCNN |     42.3     | 38.7       |
| TinyViM-L | Mask RCNN |     44.5     | 40.7    |

### 3️⃣ Semantic Segmentation

|   Model   | Head | Throughput | mIoU |
| :-------: | :--: | :--------: | ---- |
| TinyViM-B | FPN  |    180     | 41.9 |
| TinyViM-L | FPN  |    111     | 44.2 |



## 📚 Use example

- **Environment**

  ```shell
  conda create --name tinyvim python=3.9.11 -y
  conda activate tinyvim
  conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
  pip install timm==0.5.4
  python -m pip install 'https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2%2Bcu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl'
  ```

  The `mamba-ssm` wheel above provides **selective_scan_cuda** and matches Python 3.9, PyTorch 2.0, CUDA 11.x, and the non-CXX11 ABI PyTorch build used in this environment.
  
  Please refer to  [mmdetection-2.28.2](https://github.com/open-mmlab/mmdetection/tree/v2.28.2) and [mmsegmentation-0.30.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0) for environments and data preparation of detection and segmentation, respectively. 
  
- **Train**

  ```shell
  bash train.sh
  ```
  
- **Test**

  ```shell
  bash eval.sh
  ```
  
- **speed**

  ```shell
  python speed_gpu.py --model TinyViM_S --resolution 224 --batch 2048
  ```

- **Detection & Instance Segmentation**

  ```shell
  cd detection
  bash train.sh # for train
  bash eval.sh # for eval
  ```

- **Semantic Segmentation**

  ```shell
  cd segmentation
  bash train.sh # for train
  bash eval.sh # for eval
  ```

  

## WaveletTinyViM 分类实验

新增 `WaveletTinyViM_S`、`WaveletTinyViM_B`、`WaveletTinyViM_L` 三个独立注册名，
实现位于 `model/new_model.py` 和 `model/wavelet_block.py`。原 TinyViM 定义和默认选择保持现状。
新模型保留原 Stem、四级通道与深度、三次 Embedding、归一化和分类头；每个 Block 执行
Haar WT → LL 的行优先 Mamba2 / 三个高频子带的 FFT–RepConv–IFFT → 双门控 → IWT。
奇数尺寸在右侧和底部复制补齐，重建后裁回；没有添加 Block 外残差或 FFN。
蒸馏训练返回双头元组，推理返回均值；`num_classes<=0` 时保留 Identity head 行为。

依赖清单为根目录 `request.txt`，针对已有 `Experiment` 环境的 Python 3.10、Linux x86_64、
CUDA 12.1 和官方 PyTorch 2.2.2+cu121 wheel。2026-11-22 同时通过 conda 与 pip 核对：
已有 pip、packaging、setuptools、wheel 和 pip 安装的 dwconv2d；torch 等实验依赖尚未安装。
不需要小波库。安装由用户执行：

```shell
conda activate Experiment
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
hash -r
nvcc --version
python -m pip install -r request.txt
```

版本组合来自蓝图，并采用官方 [PyTorch 配套版本](https://pytorch.org/get-started/previous-versions/)、
[Mamba 2.2.2 发布包](https://github.com/state-spaces/mamba/releases/tag/v2.2.2) 和
[causal-conv1d 1.4.0 发布包](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.4.0)。
两个扩展显式使用 cp310 / cu122 / torch2.2 / ABI FALSE 的预编译 wheel；
其官方安装脚本将 CUDA 12.x（包括 PyTorch cu121）统一映射到 cu122 发布包，未提供独立 cu121 wheel。
不要用于其他 Python 或 Torch ABI。清单是待安装验证的候选环境，尚未完成 CUDA 运行验证。

安装完成后，原分类入口可选择新模型，例如（其余数据与训练参数沿用已有配置）：

```shell
python main.py --model WaveletTinyViM_S --distillation-type none
```

生产前向要求 CUDA；支持 FP32 参数与 CUDA autocast，Haar、频域分支、门控和 IWT 集中在 FP32
执行。真实 batch128 训练曾在低频 SSM 的 FP16 计算中产生非有限输出；低频 Mamba2（含适配层）
固定使用 FP32，并在分支内部关闭 autocast；外层开启 AMP 时也保持 FP32。
该策略避免 SSM 内部 FP16 溢出；外部 Stem、Embedding 和分类头仍遵循调用方的 AMP 设置。全模型 `.half()` 部署不在本次支持范围。Mamba2 及 causal-conv1d 延迟到新 Block 构造时
导入，导入错误直接上抛。Mamba2 的 `headdim=8`，内部宽度仍按 32 对齐，使头数及融合路径
投影步长均为 8 的倍数；相比原 `headdim=64` 配置，相关权重形状发生变化。
高频 RepConv 的 3×3、1×1 分支均使用 DW 卷积（实虚打包后
`groups=2C`），融合后仍为 DW 3×3 卷积，支持现有 `utils.replace_batchnorm` 推理融合协议。
此处按后续要求由蓝图中的普通卷积改为 DW；高频卷积不再进行跨通道混合。
修改前普通卷积版本的新模型检查点不能直接严格加载到 DW 版本。
新模型只能严格恢复同架构的未融合检查点，旧 TinyViM Stage 权重和优化器状态不兼容；
未提供新模型预训练权重，`pretrained=True` 会明确报错。

2026-11-24 在 `Experiment`（PyTorch 2.2.2+cu121、RTX 4060 Laptop）实测：原
`headdim=64` 配置在 causal-conv1d 步长校验处失败。改为 `headdim=8` 后，3 项已有 CUDA
前向测试及 2 项临时补充测试通过：Block 的 FP32/FP16 AMP 与边界尺寸、S/B/L 的
224×224 FP32/FP16 AMP 前向、S 的 225×193 输入、单/双头训练模式前向，以及严格加载
检查点后的 DW 融合前向。模型输出仅验证形状，未验证数值等价、精度或训练效果。
上述前向测试文件已按要求删除。随机输入完整训练调试入口为根目录
`debug_train_wavelet.py`，无需命令行参数：

```shell
conda activate Experiment
python debug_train_wavelet.py
```

文件顶部集中配置模型、epoch、epoch size（每个 epoch 的 batch 数）、batch size、图像尺寸
和类别数。默认 S、1 epoch、1 个训练 batch、1 个验证 batch、batch size 2、224×224、
1000 类。参考 `train.sh` 调用的 `main.py`，复用真实训练/验证引擎、AdamW、AMP、
mixup/cutmix、梯度裁剪、EMA 和学习率调度；使用单 GPU，关闭需要外部教师权重的蒸馏。
随机图像直接使用 Tensor，不读取数据集或下载权重。

训练模型的 RepConv 为 `deploy=False`；`net.to_deploy()` 返回独立副本，将 RepConv
融合并设置为 `deploy=True`，用于参数统计与验证。原训练模型及优化器参数引用保留，
checkpoint 保存未融合权重，用于继续训练。部署副本不要用于恢复训练。

2026-11-24 实际完成该默认配置的一次完整运行：训练 loss 6.907751、验证 loss 6.939453，
AdamW 更新 1 次，参数统计（RepConv 已融合）为 6,581,200，PyTorch 峰值分配显存约 422 MiB。
模型、优化器、scheduler、scaler、EMA 的 checkpoint 保存和恢复均通过；另外 4 项局部
单元测试通过并已清理。结果写入每次新建的 `debug-output/wavelet_training/run_*/`，包含
`checkpoint.pth` 与 `summary.json`。首次反向内核编译耗时约 193 秒；这不是稳定训练耗时。
随机数据测试不代表真实数据上的精度或长期训练稳定性。

## 🌟 Citation

If you are interested in our work, please consider giving a 🌟 and citing our work below. 

```
@misc{tinyvim,
      title={TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba}, 
      author={Xiaowen Ma and Zhenliang Ni and Xinghao Chen},
      year={2024},
      eprint={2411.17473},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.17473}, 
}
```



## 💡Acknowledgment

Thanks to previous open-sourced repo: [Efficientformer](https://github.com/snap-research/EfficientFormer), [Swiftformer](https://github.com/Amshaker/SwiftFormer), [RepViT](https://github.com/THU-MIG/RepViT), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0), [mmdetection](https://github.com/open-mmlab/mmdetection/tree/v2.28.2)



## 统一训练入口（WaveletTinyViM / TinyViM）

先执行 `conda activate Experiment`，再运行 `bash cmd_scripts/train_new.sh --help` 查看全部参数。
原 `cmd_scripts/train.sh` 保留原样。新脚本使用当前环境的 Python 启动单机 torchrun，
每张 GPU 一个进程；`CUDA_VISIBLE_DEVICES=0,1` 配合 `--gpus 2` 指定两张可见卡。
脚本可从仓库外调用，数据路径支持空格及 `~`。

| 参数 | 默认值 | 合法值与约束 |
| --- | --- | --- |
| `--dataset` | `Imagenet` | 区分大小写：`Imagenet`、`cifar100`、`tinyimagenet` |
| `--data-dir` | 必填 | 对应数据集的可读根目录；与 `--data-path` 不可混用 |
| `--gpus` | 1 | 正整数，不超过当前可见 CUDA GPU 数 |
| `--model` | `WaveletTinyViM_S` | `WaveletTinyViM_S/B/L`、`TinyViM_S/B/L` 六个完整注册名 |
| `--batch-size` | ImageNet 256，其他 200 | 正整数，每 GPU 进程的 batch；全局 batch=batch×GPU 数 |
| `--epochs` | 300 | 正整数；恢复后必须大于 start_epoch |
| `--warmup-epochs` | ImageNet 5，其他 30 | 0≤值<epochs |
| `--cooldown-epochs` | 10 | 非负整数 |
| `--input-size` | 224 | 整数≥64；小尺寸需自行核对 BatchNorm 有效样本数 |
| `--lr` | 0.001 | 有限正数；有效 lr=基础 lr×batch×GPU 数/512 |
| `--output-dir` | `checkpoints` | 非空、可写；兼容 `--output_dir` |
| `--resume` | 空 | 同架构、同类别、同蒸馏头的未融合检查点，本地文件或 HTTPS |
| `--eval` | 关闭 | 必须同时给出 `--resume`；不创建教师或训练 loader |
| `--distillation-type` | `none` | `none`、`hard`、`soft` |
| `--teacher-model` | `regnety_160` | 支持 timm 分类工厂协议且输出单个 logits 张量的教师 |
| `--teacher-path` | 空 | hard/soft 训练必填，本地文件或 HTTPS；none/eval 不读取 |
| `--distillation-alpha` | 0.5 | 有限数，0≤值≤1 |
| `--distillation-tau` | 1.0 | 有限正数，soft 蒸馏温度 |
| `--num-workers` | 10 | 非负整数；兼容 `--num_workers` |
| `--seed` | 0 | 整数，0≤值≤2³²−1 |

其他参数沿用 `main.py`，命令行覆盖数据集专属默认值。`--data-set`、`--data-path`、
`--world_size`、`--dist_url` 和非 cuda 的 `--device` 由新脚本拒绝，防止覆盖其路由或进程配置。
直接运行 `main.py` 或专用入口时保留它们原有默认模型与蒸馏默认值。

CIFAR-100 读取 `train.txt/val.txt` 及图片，保留完整 100 类标签映射；Tiny ImageNet
读取原始 `train/`、`val/images/` 与 `val/val_annotations.txt`，要求 200 类。
ImageNet 读取原有 `train/val` 类别目录或 tar，分类头为 1000 类。
`python supportcode/main_cifar100.py --check-data --data-dir ROOT` 和
`python supportcode/main_tinyimagenet.py --check-data --data-dir ROOT` 可单独检查数据。
仓库内直接训练既支持这两个文件路径，也支持 `python -m supportcode.main_cifar100` 等模块形式。

下面两条命令用于完整真实数据训练验收：每卡 batch=128、输入224、1 epoch，保留正常增强、
mixup/cutmix、BatchNorm、RASampler 与 drop_last；验证 batch=192。不会裁减数据或限制 batch 数。

```bash
bash cmd_scripts/train_new.sh --dataset cifar100 --data-dir /home/kaiser/dl_project/data/cifar100 --gpus 1 --batch-size 128 --epochs 1 --warmup-epochs 0 --cooldown-epochs 0 --output-dir debug-output/wavelet_training/cifar100
bash cmd_scripts/train_new.sh --dataset tinyimagenet --data-dir /home/kaiser/dl_project/data/tiny-imagenet-200 --gpus 1 --batch-size 128 --epochs 1 --warmup-epochs 0 --cooldown-epochs 0 --output-dir debug-output/wavelet_training/tinyimagenet
# ImageNet 数据就绪后，将 /data/imagenet 换成真实目录：
bash cmd_scripts/train_new.sh --dataset Imagenet --data-dir /data/imagenet --gpus 1 --batch-size 128 --epochs 1 --warmup-epochs 0 --cooldown-epochs 0
```

输出位于 `输出根目录/模型名/时间戳_唯一后缀/`，包括生效参数 `args.txt`、模型结构 `model.txt`、
每 epoch 的 JSON 指标 `log.txt` 和未融合权重 `checkpoint_0.pth`。日志和参数记录完整数据集长度、
loader 批次数及基础/有效学习率；本地恢复训练沿用检查点父目录，多卡共享同一个运行目录。

蒸馏教师必须使用对应数据集类别数，权重顶层为 `model`，严格加载并冻结参数。
教师工厂需接受 `pretrained=False, num_classes=K, global_pool='avg'`，返回 `[B,K]` logits。
以下教师路径是待替换的示例，需要提供真实匹配权重：

```bash
bash cmd_scripts/train_new.sh --dataset cifar100 --data-dir /home/kaiser/dl_project/data/cifar100 --batch-size 128 --epochs 1 --warmup-epochs 0 --cooldown-epochs 0 --distillation-type hard --teacher-model regnety_160 --teacher-path /weights/cifar100-teacher.pth
bash cmd_scripts/train_new.sh --dataset tinyimagenet --data-dir /home/kaiser/dl_project/data/tiny-imagenet-200 --batch-size 128 --epochs 1 --warmup-epochs 0 --cooldown-epochs 0 --distillation-type soft --teacher-model regnety_160 --teacher-path /weights/tinyimagenet-teacher.pth --distillation-alpha 0.5 --distillation-tau 2
bash cmd_scripts/train_new.sh --dataset Imagenet --data-dir /data/imagenet --batch-size 128 --epochs 1 --warmup-epochs 0 --cooldown-epochs 0 --distillation-type hard --teacher-model regnety_160 --teacher-path https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth
```

仅评估严格加载学生，`--distillation-type` 必须与检查点保存时的头配置一致；hard/soft 学生
评估无需教师路径。以下检查点路径也需替换成实际训练输出：

```bash
bash cmd_scripts/train_new.sh --dataset cifar100 --data-dir /home/kaiser/dl_project/data/cifar100 --batch-size 128 --eval --resume /checkpoints/cifar100/checkpoint_0.pth --distillation-type none
bash cmd_scripts/train_new.sh --dataset tinyimagenet --data-dir /home/kaiser/dl_project/data/tiny-imagenet-200 --batch-size 128 --eval --resume /checkpoints/tinyimagenet/checkpoint_0.pth --distillation-type none
bash cmd_scripts/train_new.sh --dataset Imagenet --data-dir /data/imagenet --batch-size 128 --eval --resume /checkpoints/imagenet/checkpoint_0.pth --distillation-type none
```

早前 `debug_train_wavelet.py` 的 batch=2 随机数据结果不代表这里的真实数据验收。
本次实际验证结果见 [训练验证记录](debug-output/wavelet_training/validation.md)。

普通训练支持 `--max-train-batches 5`，每个 epoch 只训练前 5 个 batch，随后仍执行完整验证与
checkpoint 保存；默认不限制训练 batch。两个真实数据集的短训练命令如下：

```bash
bash cmd_scripts/train_new.sh --dataset cifar100 --data-dir /home/kaiser/dl_project/data/cifar100 --batch-size 64 --epochs 1 --max-train-batches 5 --warmup-epochs 0 --cooldown-epochs 0 --output-dir debug-output/training_safety/cifar100
bash cmd_scripts/train_new.sh --dataset tinyimagenet --data-dir /home/kaiser/dl_project/data/tiny-imagenet-200 --batch-size 64 --epochs 1 --max-train-batches 5 --warmup-epochs 0 --cooldown-epochs 0 --output-dir debug-output/training_safety/tinyimagenet
```

普通训练遇到 NaN 或 Inf loss 时，在该 batch 反向传播与参数更新前保存 `erro_{epoch}.pth` 和
`erro_{epoch}.log`，随后抛出 `FloatingPointError` 以失败状态退出。epoch 与日志中的 batch 索引
均从 0 开始。日志包含各进程 loss、学习率、训练配置、调用栈与保存状态；权重包含模型、
optimizer、scheduler、AMP scaler、EMA 和异常位置。多卡由主进程保存，全部进程同步中断。
若写盘失败，异常直接向上传播，日志中的 `checkpoint_save_started` 表示保存尚未完成。
异常快照用于诊断，不作为已完成 epoch 直接续训，也不随普通 checkpoint 轮换删除。

`checkpoint_best.pth` 按验证集 Top-1 提升保存，首次有效验证即使为 0 也保存。
每个普通 checkpoint 都记录 `best_accuracy`，续训恢复历史最佳指标；旧 checkpoint 缺少该字段时，
从同目录 `log.txt` 恢复截至该 epoch 的验证历史，缺少历史则明确报错。
直接调用普通 `engine.train_one_epoch()` 时须提供 `checkpoint_writer`，避免异常诊断遗漏。
五折训练使用独立的 `five_fold/engine.py::train_fold_one_epoch()`，保留原有训练行为，
不使用普通训练的 batch 限制，本次短训练测试不包含五折流程。
