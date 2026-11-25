import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import math

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, ModelEma

from data.samplers import RASampler
from data.datasets import build_dataset
from data.threeaugment import new_data_aug_generator
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from training_checkpoint import TrainingCheckpoint

import model
import utils


def get_args_parser():
    parser = argparse.ArgumentParser(
        'TinyViM training and evaluation script', add_help=False, allow_abbrev=False)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--max-train-batches', default=None, type=int,
                        help='optional training batch limit per epoch for smoke runs')

    # Model parameters
    parser.add_argument('--model', default='TinyViM_S', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224,
                        type=int, help='images input size')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument(
        '--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float,
                        default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu',
                        action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=0.02, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='agc',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.025,
                        help='weight decay (default: 0.025)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--ThreeAugment', action='store_true')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug',
                        action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str,
                        default='https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth')
    parser.add_argument('--distillation-type', default='hard',
                        choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha',
                        default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--set_bn_eval', action='store_true', default=False,
                        help='set BN layers to eval mode during finetuning.')

    # Dataset parameters
    paths = parser.add_mutually_exclusive_group()
    paths.add_argument('--data-path', dest='data_path', default='/root/FastBaseline/data/imagenet',
                       help='dataset root (exclusive with --data-dir)')
    paths.add_argument('--data-dir', dest='data_path', help='alias for --data-path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order',
                                 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--output_dir', '--output-dir', default='checkpoints',
                        help='nonempty output root; creates a model/run subdirectory')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', '--num-workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequency of model saving')
    
    parser.add_argument('--deploy', action='store_true', default=False)
    parser.add_argument('--project', default='tinyvim', type=str)
    return parser


def validate_training_args(args):
    """Validate shared configuration before loading data or initializing CUDA."""
    if args.max_train_batches is not None and (
            type(args.max_train_batches) is not int or args.max_train_batches <= 0):
        raise ValueError('max_train_batches must be a positive integer or None')
    for name, minimum in (("batch_size", 1), ("epochs", 1), ("input_size", 64),
                          ("num_workers", 0), ("cooldown_epochs", 0), ("start_epoch", 0)):
        value = getattr(args, name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup_epochs must satisfy 0 <= warmup_epochs < epochs")
    if not args.eval and args.start_epoch >= args.epochs:
        raise ValueError("start_epoch must be smaller than epochs")
    if not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("seed must be in [0, 2**32-1]")
    for name in ("lr", "distillation_tau"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.distillation_alpha) or not 0 <= args.distillation_alpha <= 1:
        raise ValueError("distillation_alpha must be finite and in [0, 1]")
    if not args.output_dir.strip():
        raise ValueError("output_dir must not be empty")
    if not args.data_path.strip():
        raise ValueError("data_path must not be empty")
    if args.distillation_type not in ("none", "hard", "soft"):
        raise ValueError("distillation_type must be none, hard or soft")
    if args.eval and not args.resume:
        raise ValueError("eval requires resume")
    if not args.eval and args.distillation_type != "none" and not args.teacher_path:
        raise ValueError("teacher_path is required for distillation training")
    if args.model not in {"TinyViM_S", "TinyViM_B", "TinyViM_L",
                          "WaveletTinyViM_S", "WaveletTinyViM_B", "WaveletTinyViM_L"}:
        raise ValueError(f"unknown model: {args.model}")


def prepare_output_dir(args):
    """Choose one shared run directory, preserving local resume locations."""
    chosen = [None]
    if utils.is_main_process():
        if args.resume and not args.eval and not args.resume.startswith('https://'):
            chosen[0] = str(Path(args.resume).expanduser().resolve().parent)
        else:
            # The unique suffix also prevents collisions between simultaneous runs.
            parent = Path(args.output_dir).expanduser().resolve() / args.model
            chosen[0] = str(parent / (datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                                     + '_' + os.urandom(4).hex()))
    if utils.is_dist_avail_and_initialized():
        torch.distributed.broadcast_object_list(chosen, src=0)
    output = Path(chosen[0])
    # All ranks create the same directory; failures propagate to the launcher.
    output.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output)
    return output


def build_teacher(args, num_classes, device):
    """Load a frozen classification teacher only for distillation training."""
    if args.eval or args.distillation_type == 'none':
        return None
    if not args.teacher_path:
        raise ValueError("teacher_path is required for distillation training")
    print(f"Creating teacher: {args.teacher_model}, classes={num_classes}, weights={args.teacher_path}")
    teacher = create_model(args.teacher_model, pretrained=False,
                           num_classes=num_classes, global_pool='avg')
    if args.teacher_path.startswith('https://'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.teacher_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(Path(args.teacher_path).expanduser(), map_location='cpu')
    teacher.load_state_dict(checkpoint['model'], strict=True)
    teacher.requires_grad_(False)
    return teacher.to(device).eval()


def main(args, dataset_builder=None):
    validate_training_args(args)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requires an available CUDA GPU")
    if args.model.startswith('Wavelet') and not args.device.startswith('cuda'):
        raise ValueError("WaveletTinyViM requires device=cuda")

    utils.init_distributed_mode(args)
    output_dir = prepare_output_dir(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError(
            "Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    # random.seed(seed)

    cudnn.benchmark = True

    factory = build_dataset if dataset_builder is None else dataset_builder
    args.data_path = str(Path(args.data_path).expanduser())
    dataset_train, args.nb_classes = factory(is_train=True, args=args)
    dataset_val, val_classes = factory(is_train=False, args=args)
    if args.nb_classes <= 0 or args.nb_classes != val_classes:
        raise ValueError("training and validation class counts must match and be positive")

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = None
    if not args.eval:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
        if len(data_loader_train) == 0:
            raise ValueError("training loader is empty; check dataset and batch_size")
        if args.ThreeAugment:
            dataset_train.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    if len(data_loader_val) == 0:
        raise ValueError("validation loader is empty")
    args.train_samples = len(dataset_train)
    args.val_samples = len(dataset_val)
    args.train_batches = len(data_loader_train) if data_loader_train is not None else 0
    if args.max_train_batches is not None:
        args.train_batches = min(args.train_batches, args.max_train_batches)
    args.val_batches = len(data_loader_val)
    print(f"Data: train={args.train_samples}, val={args.val_samples}, classes={args.nb_classes}, "
          f"train_batches={args.train_batches}, val_batches={args.val_batches}, "
          f"batch_per_gpu={args.batch_size}, val_batch={int(1.5 * args.batch_size)}", flush=True)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        distillation=(args.distillation_type != 'none'),
        pretrained=False,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            print("Loading local checkpoint at {}".format(args.finetune))
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.l.weight', 'head.l.bias',
                  'head_dist.l.weight', 'head_dist.l.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but
        # before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.base_lr = args.lr
    args.lr = linear_scaled_lr
    print(f"Learning rate: base={args.base_lr}, effective={args.lr}")
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = build_teacher(args, args.nb_classes, device)

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is
    # 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    if args.output_dir and utils.is_main_process():
        with (output_dir / "model.txt").open("a") as f:
            f.write(str(model))
            print(str(model))
    if args.output_dir and utils.is_main_process():
        with (output_dir / "args.txt").open("a") as f:
            f.write(json.dumps(args.__dict__, indent=2) + "\n")
            print(json.dumps(args.__dict__, indent=2) + "\n")
    checkpoint_writer = TrainingCheckpoint(
        output_dir, model_without_ddp, optimizer, lr_scheduler, loss_scaler, model_ema, args)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print("Loading local checkpoint at {}".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
        if not args.eval and 'failure' in checkpoint:
            raise ValueError('Interrupted checkpoints are diagnostic snapshots, not completed epochs')
        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        print(msg)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(
                    model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            checkpoint_writer.restore_best(checkpoint)
    if args.eval:
        utils.replace_batchnorm(model)
        print(f"Evaluating model: {args.model}")
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    if args.start_epoch >= args.epochs:
        raise ValueError("resume start_epoch must be smaller than epochs")
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, args.clip_mode, model_ema, mixup_fn,
            # set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
            set_training_mode=True,
            set_bn_eval=args.set_bn_eval, # set bn to eval if finetune
            checkpoint_writer=checkpoint_writer,
            max_batches=args.max_train_batches,
        )

        lr_scheduler.step(epoch)

        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        checkpoint_writer.save_epoch(epoch, test_stats['acc1'])
        print(f'Max accuracy: {checkpoint_writer.best_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'TinyViM training and evaluation script', parents=[get_args_parser()], allow_abbrev=False)
    args = parser.parse_args()
    main(args)
