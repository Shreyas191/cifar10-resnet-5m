"""
CIFAR-10 data pipeline.

Train-time augmentation (strong, all standard for CIFAR SOTA recipes):
  RandomCrop(32, padding=4, reflect) -> RandomHorizontalFlip
  -> TrivialAugmentWide (or AutoAugment CIFAR10 policy) -> ToTensor -> Normalize
  -> RandomErasing (Cutout-style)
plus batch-level CutMix / MixUp applied inside the training loop (see train.py).

Also supports loading the Kaggle competition test set, which is distributed as a
pickle file (same layout as the original CIFAR-10 python batches).
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def build_train_transform(autoaugment: str = "trivial", random_erasing: float = 0.25):
    ops = [T.RandomCrop(32, padding=4, padding_mode="reflect"), T.RandomHorizontalFlip()]
    if autoaugment == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif autoaugment == "cifar":
        ops.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
    elif autoaugment == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=10))
    elif autoaugment not in (None, "none"):
        raise ValueError(f"Unknown autoaugment policy: {autoaugment}")
    ops += [T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]
    if random_erasing > 0:
        ops.append(T.RandomErasing(p=random_erasing, scale=(0.02, 0.2), value="random"))
    return T.Compose(ops)


def build_test_transform():
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])


def get_cifar10_loaders(
    data_dir="./data",
    batch_size=128,
    num_workers=4,
    autoaugment="trivial",
    random_erasing=0.25,
    val_split=0,
    seed=0,
    train_csv=None,
    train_dir=None,
):
    """Returns (train_loader, val_loader_or_None, test_loader).

    train_csv/train_dir : the competition's own training set (train_labels.csv + train_images/).
                          If given, we train on THAT (recommended -- the Kaggle test images come
                          from the same source). Otherwise torchvision's CIFAR-10 train split.
    val_split > 0       : carves a held-out validation set off the training images, so that
                          hyper-parameters can be tuned without touching the test set.
    The 10k torchvision CIFAR-10 test split is always used as the reported test set; it is
    never trained on.
    """
    train_tf, test_tf = build_train_transform(autoaugment, random_erasing), build_test_transform()
    if train_csv:
        train_dir = train_dir or os.path.join(os.path.dirname(train_csv), "train_images")
        full_train = CSVImageDataset(train_csv, train_dir, transform=train_tf)
        val_base = CSVImageDataset(train_csv, train_dir, transform=test_tf)
        print(f"Training on competition images: {len(full_train)} from {train_dir}")
    else:
        full_train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
        val_base = torchvision.datasets.CIFAR10(data_dir, train=True, download=False, transform=test_tf)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    val_loader = None
    if val_split > 0:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(full_train), generator=g).tolist()
        val_idx, train_idx = perm[:val_split], perm[val_split:]
        val_set = torch.utils.data.Subset(val_base, val_idx)
        full_train = torch.utils.data.Subset(full_train, train_idx)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(
        full_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Kaggle competition test set
# ---------------------------------------------------------------------------
class PickleImageDataset(Dataset):
    """Loads the competition's unlabeled test pickle.

    Handles the common layouts:
      {b'data': uint8 (N, 3072)}             -- original CIFAR batch format (CHW flattened)
      {b'data': uint8 (N, 32, 32, 3)}        -- HWC images
      {b'data': ..., b'ids': [...]}          -- optional explicit ids
    """

    def __init__(self, path, transform=None):
        with open(path, "rb") as f:
            raw = pickle.load(f, encoding="bytes")
        if isinstance(raw, dict):
            key = b"data" if b"data" in raw else "data"
            data = raw[key]
            ids = raw.get(b"ids", raw.get("ids", None))
        else:
            data, ids = raw, None
        data = np.asarray(data)
        if data.ndim == 2:  # (N, 3072) -> (N, 32, 32, 3)
            data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        elif data.ndim == 4 and data.shape[1] == 3:  # (N, 3, 32, 32)
            data = data.transpose(0, 2, 3, 1)
        self.data = data.astype(np.uint8)
        self.ids = list(ids) if ids is not None else list(range(len(self.data)))
        self.transform = transform or build_test_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.fromarray(self.data[i])
        return self.transform(img), self.ids[i]


class CSVImageDataset(Dataset):
    """Images listed in a CSV (`id,filename[,label]`) living in `img_dir`.

    This is the format of the Fall-2026 competition: `test.csv` + a folder of
    `test_XXXX.png` files.  If a `label` column exists it is returned too, so the
    same class can serve a PNG-based *training* set.
    """

    def __init__(self, csv_path, img_dir, transform=None, id_col="id", file_col="filename", label_col="label"):
        import csv as _csv
        with open(csv_path, newline="") as f:
            rows = list(_csv.DictReader(f))
        cols = {c.lower(): c for c in rows[0]} if rows else {}
        id_col, file_col, label_col = cols.get(id_col, id_col), cols.get(file_col, file_col), cols.get(label_col, label_col)
        self.ids = [int(r[id_col]) for r in rows] if id_col in cols else list(range(len(rows)))
        self.files = [os.path.join(img_dir, r[file_col]) for r in rows]
        self.labels = None
        if rows and label_col in cols:
            raw = [r[label_col] for r in rows]
            self.labels = [CLASSES.index(v.lower()) if not v.strip().lstrip("-").isdigit() else int(v) for v in raw]
        self.transform = transform or build_test_transform()
        missing = [p for p in self.files[:5] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Image not found, check --test_dir: {missing[0]}")
        # Decode every PNG once and keep the uint8 pixels in RAM (50k x 32x32x3 = 150 MB).
        # Re-reading 50k small files per epoch from a network disk (Kaggle /kaggle/input,
        # Google Drive) starves the GPU; this makes epochs I/O-free.
        from PIL import Image
        cache = os.path.join(os.path.dirname(os.path.abspath(csv_path)) if os.access(os.path.dirname(os.path.abspath(csv_path)), os.W_OK) else "/tmp",
                             f".cache_{os.path.basename(csv_path)}_{len(self.files)}.npy")
        if os.path.exists(cache):
            self.data = np.load(cache)
        else:
            self.data = np.stack([np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8) for f in self.files])
            try:
                np.save(cache, self.data)
            except OSError:
                pass

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        x = self.transform(Image.fromarray(self.data[i]))
        return (x, self.labels[i]) if self.labels is not None else (x, self.ids[i])


def get_kaggle_test_loader(path, img_dir=None, batch_size=256, num_workers=4):
    """`path` is either the competition pickle or a CSV (then `img_dir` holds the PNGs)."""
    if path.lower().endswith(".csv"):
        img_dir = img_dir or os.path.join(os.path.dirname(path), "test_images")
        ds = CSVImageDataset(path, img_dir)
    else:
        ds = PickleImageDataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
