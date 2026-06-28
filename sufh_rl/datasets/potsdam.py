from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import random


class PotsdamMulticlassDataset(Dataset):
   

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        split_file: str,
        crop_size: Optional[int] = 256,
        mode: str = "train",
        normalize: bool = True,
        ignore_index: int = 255,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.split_file = Path(split_file)

        self.crop_size = crop_size
        self.mode = mode
        self.normalize = normalize
        self.ignore_index = ignore_index

        assert self.image_dir.exists(), f"image_dir not found: {self.image_dir}"
        assert self.label_dir.exists(), f"label_dir not found: {self.label_dir}"
        assert self.split_file.exists(), f"split_file not found: {self.split_file}"
        assert self.mode in ["train", "val", "test"]

        self.ids = self._load_ids()
        assert len(self.ids) > 0, f"No ids found in {self.split_file}"

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.num_classes = 6
        self.class_names = [
            "impervious_surfaces",
            "building",
            "low_vegetation",
            "tree",
            "car",
            "clutter_background",
        ]

    def _load_ids(self) -> List[str]:
        ids = self.split_file.read_text().strip().splitlines()
        ids = [x.strip() for x in ids if x.strip()]
        return ids

    def __len__(self):
        return len(self.ids)

    def _load_image_label(self, sample_id: str):
        image_path = self.image_dir / f"{sample_id}.png"
        label_path = self.label_dir / f"{sample_id}.png"

        assert image_path.exists(), f"Missing image: {image_path}"
        assert label_path.exists(), f"Missing label: {label_path}"

        image = Image.open(image_path).convert("RGB")


        label = Image.open(label_path)
        if label.mode != "L":
            label = label.convert("L")

        return image, label

    def _random_crop_params(self, image: Image.Image, crop_size: int):
        w, h = image.size
        th, tw = crop_size, crop_size

        if w == tw and h == th:
            return 0, 0, h, w

        if w < tw or h < th:

            return None

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def _apply_transforms(self, image: Image.Image, label: Image.Image):
      

        if self.mode == "train":
            if random.random() < 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)

            if random.random() < 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)

            if self.crop_size is not None:
                params = self._random_crop_params(image, self.crop_size)
                if params is not None:
                    i, j, h, w = params
                    image = TF.crop(image, i, j, h, w)
                    label = TF.crop(label, i, j, h, w)
                else:
                    image = TF.resize(
                        image,
                        [self.crop_size, self.crop_size],
                        interpolation=InterpolationMode.BILINEAR,
                    )
                    label = TF.resize(
                        label,
                        [self.crop_size, self.crop_size],
                        interpolation=InterpolationMode.NEAREST,
                    )

        else:
            if self.crop_size is not None:
                w, h = image.size
                th, tw = self.crop_size, self.crop_size

                if w >= tw and h >= th:
                    i = max((h - th) // 2, 0)
                    j = max((w - tw) // 2, 0)
                    image = TF.crop(image, i, j, th, tw)
                    label = TF.crop(label, i, j, th, tw)
                else:
                    image = TF.resize(
                        image,
                        [self.crop_size, self.crop_size],
                        interpolation=InterpolationMode.BILINEAR,
                    )
                    label = TF.resize(
                        label,
                        [self.crop_size, self.crop_size],
                        interpolation=InterpolationMode.NEAREST,
                    )

        return image, label

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_id = self.ids[idx]

        image, label = self._load_image_label(sample_id)
        image, label = self._apply_transforms(image, label)

        # image: [H,W,3] uint8 -> [3,H,W] float32 in [0,1]
        image = TF.to_tensor(image)

        if self.normalize:
            image = TF.normalize(image, mean=self.mean, std=self.std)

     
        label = np.array(label, dtype=np.uint8)


        valid_mask = ((label >= 0) & (label < self.num_classes)) | (label == self.ignore_index)
        if not np.all(valid_mask):
            bad_values = np.unique(label[~valid_mask]).tolist()
            raise ValueError(
                f"Sample {sample_id} has invalid label values: {bad_values}. "
                f"Expected 0~{self.num_classes - 1} or ignore_index={self.ignore_index}"
            )

        label = torch.from_numpy(label.astype(np.int64))  # [H,W], long

        return {
            "image": image,   # [3,H,W], float32
            "label": label,   # [H,W], int64
            "id": sample_id,
        }


if __name__ == "__main__":

    data_root = Path("/root/autodl-tmp/data/potsdam")

    dataset = PotsdamMulticlassDataset(
        image_dir=str(data_root / "processed_multiclass/images"),
        label_dir=str(data_root / "processed_multiclass/labels"),
        split_file=str(data_root / "splits/train.txt"),
        crop_size=256,
        mode="train",
        normalize=True,
        ignore_index=255,
    )

    print(f"dataset size: {len(dataset)}")

    sample = dataset[0]
    print("image shape:", sample["image"].shape, sample["image"].dtype)
    print("label shape:", sample["label"].shape, sample["label"].dtype)
    print("sample id:", sample["id"])
    print("unique labels:", torch.unique(sample["label"]).tolist())
