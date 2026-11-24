"""Tiny ImageNet loader for the original train/ and val/images/ layout.

Validation labels come from val/val_annotations.txt and use the same sorted
class indices as ImageFolder(train). No dataset files are moved or rewritten.
"""

from pathlib import Path

from torchvision.datasets import ImageFolder, VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader, find_classes

from data.datasets import build_transform


class TinyImageNetValidation(VisionDataset):
    def __init__(self, root, class_to_idx, transform=None):
        root = Path(root).expanduser()
        super().__init__(str(root / 'val'), transform=transform)
        self.class_to_idx = dict(class_to_idx)
        self.classes = sorted(self.class_to_idx, key=self.class_to_idx.get)
        self.loader = default_loader
        image_dir = root / 'val' / 'images'
        annotation_path = root / 'val' / 'val_annotations.txt'
        samples = []
        seen = set()
        for line_number, line in enumerate(annotation_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f'{annotation_path}:{line_number}: expected 6 fields')
            filename, class_name = fields[:2]
            if Path(filename).name != filename or '\\' in filename:
                raise ValueError(f'Invalid image filename: {filename}')
            if filename in seen:
                raise ValueError(f'Duplicate validation image: {filename}')
            if class_name not in self.class_to_idx:
                raise ValueError(f'Unknown validation class: {class_name}')
            path = image_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            seen.add(filename)
            samples.append((str(path), self.class_to_idx[class_name]))
        if not samples:
            raise ValueError(f'No validation annotations in {annotation_path}')
        image_names = {
            path.name for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
        }
        if image_names != seen:
            raise ValueError(
                'Validation images and annotations differ: '
                f'{len(image_names - seen)} unannotated images, '
                f'{len(seen - image_names)} invalid image entries'
            )
        self.samples = sorted(samples)
        self.imgs = self.samples
        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_dataset(is_train, args):
    """Match main.build_dataset's interface, deriving class count from train/."""
    if args.data_set != 'IMNET':
        raise ValueError('The Tiny ImageNet entry requires --data-set IMNET')
    root = Path(args.data_path).expanduser()
    transform = build_transform(is_train, args)
    if is_train:
        dataset = ImageFolder(str(root / 'train'), transform=transform)
    else:
        _, class_to_idx = find_classes(str(root / 'train'))
        dataset = TinyImageNetValidation(root, class_to_idx, transform=transform)
    return dataset, len(dataset.classes)
