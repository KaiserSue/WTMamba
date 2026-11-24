"""Entry for exported CIFAR-100 images with train.txt/val.txt annotations.

Upload this file and data/cifar100_manifest.py, preserving their relative paths.
Replace main.py with main_cifar100.py in existing python/torchrun commands.
The dataset defaults to CIFAR, the output head has 100 classes, and the original
input-size default (224) and preprocessing remain in effect. For a CPU check:

    python main_cifar100.py --check-data --data-path /path/to/cifar100
"""

import argparse
import datetime
from collections import Counter
from pathlib import Path

from data.cifar100_manifest import CIFAR100Manifest, build_dataset


def main():
    check_parser = argparse.ArgumentParser(add_help=False)
    check_parser.add_argument('--check-data', action='store_true')
    check_parser.add_argument('--data-path')
    check_args, _ = check_parser.parse_known_args()
    if check_args.check_data:
        if not check_args.data_path:
            check_parser.error('--check-data requires --data-path')
        train = CIFAR100Manifest(check_args.data_path, 'train')
        val = CIFAR100Manifest(check_args.data_path, 'val')
        for name, dataset in (('Train', train), ('Validation', val)):
            counts = Counter(dataset.targets)
            image, _ = dataset[0]
            print(f'{name}: {len(dataset)} images, {len(counts)} labeled classes')
            print(f'{name} images per class (images: classes): {dict(sorted(Counter(counts.values()).items()))}')
            print(f'{name} decoded image size: {image.size}')
        print(f'Class mappings match: {train.class_to_idx == val.class_to_idx}')
        return

    import main as training

    parser = argparse.ArgumentParser(
        'TinyViM CIFAR-100 training and evaluation',
        parents=[training.get_args_parser()],
    )
    parser.set_defaults(
        data_set='CIFAR', distillation_type='none', warmup_epochs=30, batch_size=200,
    )
    args = parser.parse_args()
    if args.data_set != 'CIFAR':
        parser.error('this entry requires --data-set CIFAR')
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
