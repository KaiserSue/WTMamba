"""Load exported CIFAR-100 images using root/train.txt and root/val.txt.

Each manifest row is '<split>/<class>/<image> <numeric label>'. Labels are
preserved exactly; torchvision's original CIFAR-100 pickle format is not used.
"""

from pathlib import Path, PurePosixPath

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

from data.datasets import build_transform


def _read_manifest(root, split):
    manifest = root / f'{split}.txt'
    samples = []
    mapping = {}
    label_names = {}
    seen = set()
    for number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f'{manifest}:{number}: expected image path and label')
        relative = PurePosixPath(fields[0])
        if (relative.is_absolute() or len(relative.parts) != 3
                or '..' in relative.parts or '\\' in fields[0]
                or relative.parts[0] != split
                or relative.suffix.lower() not in IMG_EXTENSIONS):
            raise ValueError(f'{manifest}:{number}: invalid image path {fields[0]}')
        label = int(fields[1])
        if not 0 <= label < 100:
            raise ValueError(f'{manifest}:{number}: label must be in [0, 99]')
        name = relative.parts[1]
        if name in mapping and mapping[name] != label:
            raise ValueError(f'{manifest}:{number}: conflicting labels for {name}')
        if label in label_names and label_names[label] != name:
            raise ValueError(f'{manifest}:{number}: label {label} has multiple class names')
        if relative in seen:
            raise ValueError(f'{manifest}:{number}: duplicate image {relative}')
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(relative)
        mapping[name] = label
        label_names[label] = name
        samples.append((str(path), label))
    if not samples:
        raise ValueError(f'No samples in {manifest}')
    return samples, mapping


class CIFAR100Manifest(VisionDataset):
    def __init__(self, root, split, transform=None):
        if split not in ('train', 'val'):
            raise ValueError('CIFAR-100 split must be train or val')
        root = Path(root).expanduser()
        super().__init__(str(root), transform=transform)
        train_samples, mapping = _read_manifest(root, 'train')
        if set(mapping.values()) != set(range(100)):
            raise ValueError('train.txt must define all 100 CIFAR-100 classes')
        samples = train_samples
        if split == 'val':
            samples, val_mapping = _read_manifest(root, 'val')
            for name, label in val_mapping.items():
                if name not in mapping or mapping[name] != label:
                    raise ValueError(f'Validation class mapping differs from train.txt: {name}={label}')
        self.split = split
        self.class_to_idx = mapping
        self.classes = sorted(mapping, key=mapping.get)
        self.samples = samples
        self.imgs = samples
        self.targets = [label for _, label in samples]
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_dataset(is_train, args):
    """Provide main.py's dataset factory interface for manifest-based CIFAR-100."""
    if args.data_set != 'CIFAR':
        raise ValueError('The CIFAR-100 entry requires --data-set CIFAR')
    dataset = CIFAR100Manifest(
        args.data_path, 'train' if is_train else 'val',
        transform=build_transform(is_train, args),
    )
    return dataset, 100
