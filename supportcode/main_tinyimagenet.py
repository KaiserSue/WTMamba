"""Standalone Tiny ImageNet entry; upload together with data/tinyimagenet.py.

Replace main.py with main_tinyimagenet.py in existing python/torchrun commands.
All original training/evaluation options are accepted. For a CPU data check:

    python main_tinyimagenet.py --check-data --data-path /path/to/tiny-imagenet-200

This entry changes the dataset factory in memory only; main.py, the original
loader, and the on-disk dataset remain unchanged.
"""

import argparse
import datetime
from collections import Counter
from pathlib import Path

from data.tinyimagenet import build_dataset


def main():
    # A separate check path avoids importing model/CUDA dependencies for inspection.
    check_parser = argparse.ArgumentParser(add_help=False)
    check_parser.add_argument('--check-data', action='store_true')
    check_parser.add_argument('--data-path')
    check_args, _ = check_parser.parse_known_args()
    if check_args.check_data:
        if not check_args.data_path:
            check_parser.error('--check-data requires --data-path')
        from torchvision.datasets import ImageFolder
        from data.tinyimagenet import TinyImageNetValidation

        root = Path(check_args.data_path).expanduser()
        train = ImageFolder(str(root / 'train'))
        val = TinyImageNetValidation(root, train.class_to_idx)
        counts = Counter(val.targets)
        train_image, _ = train[0]
        val_image, _ = val[0]
        print(f'Train: {len(train)} images, {len(train.classes)} classes')
        print(f'Validation: {len(val)} images, {len(counts)} labeled classes')
        print(f'Validation images per class: {dict(sorted(Counter(counts.values()).items()))}')
        print(f'Class mappings match: {train.class_to_idx == val.class_to_idx}')
        print(f'Decoded train/validation images: {train_image.size}, {val_image.size}')
        return

    import main as training

    parser = argparse.ArgumentParser(
        'TinyViM Tiny ImageNet training and evaluation',
        parents=[training.get_args_parser()],
    )
    parser.set_defaults(distillation_type='none', warmup_epochs=30, batch_size=200)
    args = parser.parse_args()
    if args.data_set != 'IMNET':
        parser.error('this entry requires --data-set IMNET')
    # main imports its factory directly, so replace that binding, not data.datasets.
    training.build_dataset = build_dataset
    if args.resume and not args.eval:
        args.output_dir = '/'.join(args.resume.split('/')[:-1])
    elif args.output_dir:
        args.output_dir = (
            args.output_dir + f'/{args.model}/'
            + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    else:
        parser.error('--output_dir must not be empty')
    training.main(args)


if __name__ == '__main__':
    main()
