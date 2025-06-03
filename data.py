# --- Dataset and DataLoader ---
import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class CelebADatasetWrapper(Dataset):
    def __init__(
        self,
        config: dict[str, Any],
        split: str = "train",
        target_attribute: str | None = None,
    ):
        self.config = config
        self.image_size = config["dataset"]["image_size"]
        self.center_crop_size = config["dataset"]["center_crop_size"]
        self.root_dir = Path(config["dataset"]["root_dir"])
        self.target_attribute = target_attribute
        self.attributes_names = []  # Will be populated if target_attribute is used

        transform_list = [
            transforms.CenterCrop(self.center_crop_size),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize to [-1, 1]
        ]
        self.transform = transforms.Compose(transform_list)

        target_type = (
            ["attr"] if self.target_attribute else []
        )  # Request attributes if needed

        # Check if CelebA dataset exists, if not, prompt to download
        # A bit tricky as torchvision downloads automatically if download=True
        # For simplicity, assume it will download or is present.

        self.celeba_dataset = datasets.CelebA(
            root=str(self.root_dir),
            split=split if config["dataset"]["celeba_use_predefined_splits"] else "all",
            target_type=target_type,
            transform=self.transform,
            download=True,  # Set to True to attempt download if not found
        )

        if self.target_attribute:
            self.attributes_names = self.celeba_dataset.attr_names
            try:
                self.target_attribute_idx = self.attributes_names.index(
                    self.target_attribute
                )
            except ValueError:
                logging.error(
                    f"Attribute '{self.target_attribute}' not found in CelebA attributes. Available: {self.attributes_names}"
                )
                raise

        if not config["dataset"]["celeba_use_predefined_splits"] and split != "all":
            # Perform custom split
            total_size = len(self.celeba_dataset)
            train_ratio = config["dataset"]["split_ratios"]["train"]
            val_ratio = config["dataset"]["split_ratios"]["val"]

            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            test_size = total_size - train_size - val_size

            if (
                train_size + val_size + test_size != total_size
            ):  # Check for rounding issues
                # Adjust train_size to make sum exact, usually it's the largest
                train_size = total_size - val_size - test_size

            logging.info(
                f"Custom splitting 'all' data: Train={train_size}, Val={val_size}, Test={test_size}"
            )

            # Ensure reproducible splits
            generator = torch.Generator().manual_seed(config["seed"])
            train_dataset, val_dataset, test_dataset = random_split(
                self.celeba_dataset,
                [train_size, val_size, test_size],
                generator=generator,
            )

            if split == "train":
                self.dataset_subset = train_dataset
            elif split == "valid":  # torchvision uses 'valid' for validation split name
                self.dataset_subset = val_dataset
            elif split == "test":
                self.dataset_subset = test_dataset
            else:
                raise ValueError(f"Invalid split name for custom split: {split}")
        else:
            self.dataset_subset = self.celeba_dataset

    def __len__(self) -> int:
        return len(self.dataset_subset)

    def __getitem__(self, idx: int) -> Tensor | tuple[Tensor, Tensor]:
        if self.target_attribute:
            img, attr = self.dataset_subset[idx]
            target_attr_val = attr[
                self.target_attribute_idx
            ].float()  # Ensure it's a float tensor
            return img, target_attr_val
        else:
            # If dataset_subset is from random_split, it directly returns the item.
            # If dataset_subset is CelebA itself, it might return (img, attr_tensor) or just img
            # depending on target_type. We only care about img here.
            item = self.dataset_subset[idx]
            if isinstance(item, tuple):  # (img, attributes)
                return item[0]
            return item  # just img


def get_dataloaders(
    config: dict[str, Any], target_attribute_for_vis: str | None = None
) -> dict[str, DataLoader]:
    dataloaders = {}
    splits_to_load = ["train", "valid"]  # Always load train and val
    if (
        config["dataset"]["celeba_use_predefined_splits"]
        or (
            1.0
            - config["dataset"]["split_ratios"]["train"]
            - config["dataset"]["split_ratios"]["val"]
        )
        > 1e-5
    ):  # if test split is non-zero
        splits_to_load.append("test")

    for split in splits_to_load:
        # For val/test loader used for embeddings, pass the target_attribute
        current_target_attribute = (
            target_attribute_for_vis if split in ["valid", "test"] else None
        )
        dataset = CelebADatasetWrapper(
            config, split=split, target_attribute=current_target_attribute
        )
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=config["dataset"]["batch_size"],
            shuffle=(split == "train"),
            num_workers=config["dataset"]["num_workers"],
            pin_memory=config["dataset"]["pin_memory"],
            drop_last=(split == "train"),  # Drop last incomplete batch for training
        )
        logging.info(f"Loaded {split} dataset with {len(dataset)} images.")
    return dataloaders
