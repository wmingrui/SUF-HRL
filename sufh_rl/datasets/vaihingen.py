from pathlib import Path
import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VaihingenMulticlassDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        split_file: str,
        crop_size: int = 512,
        mode: str = "train",   # train / val / test
        normalize: bool = True,
        ignore_index: int = 255,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rot90_prob: float = 0.5,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.split_file = Path(split_file)

        self.crop_size = crop_size
        self.mode = mode
        self.normalize = normalize
        self.ignore_index = ignore_index

        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob

        with open(self.split_file, "r", encoding="utf-8") as f:
            self.sample_names = [line.strip() for line in f if line.strip()]

        if len(self.sample_names) == 0:
            raise ValueError(f"No samples found in split file: {self.split_file}")

    def __len__(self):
        return len(self.sample_names)

    def _load_image(self, path: Path):
        img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        return img

    def _load_label(self, path: Path):
        lbl = np.array(Image.open(path), dtype=np.uint8)
        return lbl

    def _random_flip_rotate(self, image: np.ndarray, label: np.ndarray):
        if random.random() < self.hflip_prob:
            image = np.ascontiguousarray(image[:, ::-1, :])
            label = np.ascontiguousarray(label[:, ::-1])

        if random.random() < self.vflip_prob:
            image = np.ascontiguousarray(image[::-1, :, :])
            label = np.ascontiguousarray(label[::-1, :])

        if random.random() < self.rot90_prob:
            k = random.randint(0, 3)
            if k > 0:
                image = np.ascontiguousarray(np.rot90(image, k=k, axes=(0, 1)))
                label = np.ascontiguousarray(np.rot90(label, k=k, axes=(0, 1)))

        return image, label

    def _to_tensor(self, image: np.ndarray, label: np.ndarray):
        image = image.astype(np.float32) / 255.0
        if self.normalize:
            image = (image - IMAGENET_MEAN) / IMAGENET_STD

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()   # [3,H,W]
        label = torch.from_numpy(label.astype(np.int64)).long()      # [H,W]
        return image, label

    def __getitem__(self, idx):
        name = self.sample_names[idx]
        image_path = self.image_dir / f"{name}.png"
        label_path = self.label_dir / f"{name}.png"

        image = self._load_image(image_path)
        label = self._load_label(label_path)

        if self.mode == "train":
            image, label = self._random_flip_rotate(image, label)

        image, label = self._to_tensor(image, label)

        return {
            "image": image,
            "label": label,
            "name": name,
            "index": idx,
        }


if __name__ == "__main__":
    # 简单自检
    DATA_ROOT = Path("/root/autodl-tmp/data/vaihingen")
    ds = VaihingenMulticlassDataset(
        image_dir=str(DATA_ROOT / "processed_multiclass/images"),
        label_dir=str(DATA_ROOT / "processed_multiclass/labels"),
        split_file=str(DATA_ROOT / "splits/train.txt"),
        crop_size=512,
        mode="train",
        normalize=True,
        ignore_index=255,
    )

    print("dataset len:", len(ds))
    sample = ds[0]
    print("name :", sample["name"])
    print("image:", sample["image"].shape, sample["image"].dtype)
    print("label:", sample["label"].shape, sample["label"].dtype)
    print("label unique:", torch.unique(sample["label"]))