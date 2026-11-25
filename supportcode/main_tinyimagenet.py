"""Standalone Tiny ImageNet entry; upload together with data/tinyimagenet.py.

Replace main.py with main_tinyimagenet.py in existing python/torchrun commands.
All original training/evaluation options are accepted. For a CPU data check:

    python main_tinyimagenet.py --check-data --data-path /path/to/tiny-imagenet-200

This entry explicitly supplies the original-layout dataset factory to main.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

# Support both repository execution and the original copied-to-root layout.
_entry_dir = Path(__file__).resolve().parent
_repo_root = _entry_dir if (_entry_dir / 'model').is_dir() else _entry_dir.parent
if not (_repo_root / 'main.py').is_file() or not (_repo_root / 'model').is_dir():
    raise RuntimeError(f"Cannot locate training repository from {_entry_dir}")
sys.path.insert(0, str(_repo_root))

from data.tinyimagenet import build_dataset as _build_dataset


def build_dataset(is_train, args):
    """Keep the original-layout adapter's class contract at the dataset entry."""
    dataset, classes = _build_dataset(is_train, args)
    if classes != 200:
        raise ValueError(f"Tiny ImageNet requires 200 classes, found {classes} at {args.data_path}")
    return dataset, classes


def main():
    # A separate check path avoids importing model/CUDA dependencies for inspection.
    check_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    check_parser.add_argument('--check-data', action='store_true')
    check_paths = check_parser.add_mutually_exclusive_group()
    check_paths.add_argument('--data-path', dest='data_path')
    check_paths.add_argument('--data-dir', dest='data_path')
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
        parents=[training.get_args_parser()], allow_abbrev=False,
    )
    parser.set_defaults(distillation_type='none', warmup_epochs=30, batch_size=200)
    args = parser.parse_args()
    if args.data_set != 'IMNET':
        parser.error('this entry requires --data-set IMNET')
    training.main(args, dataset_builder=build_dataset)


if __name__ == '__main__':
    main()
