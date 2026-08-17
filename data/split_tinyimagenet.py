"""
Split TinyImageNet dataset for class-incremental continual learning.

TinyImageNet: 200 classes, 500 train / 50 val images per class, 64x64 images.
Split into T tasks of C classes each.

Usage is identical to SplitCIFAR100 — same interface for seamless cross-dataset transfer.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from typing import List, Tuple, Optional, Dict
from data.split_cifar100 import RemappedSubset
from PIL import Image


class TinyImageNetDataset(Dataset):
    """
    TinyImageNet dataset loader.
    
    Expects the standard TinyImageNet directory structure:
        tiny-imagenet-200/
        ├── train/
        │   ├── n01443537/
        │   │   └── images/
        │   └── ...
        ├── val/
        │   ├── images/
        │   └── val_annotations.txt
        └── wnids.txt
    """
    
    def __init__(self, root: str, split: str = "train", transform=None):
        self.root = os.path.join(root, "tiny-imagenet-200")
        self.split = split
        self.transform = transform
        self.samples = []
        self.targets = []
        
        # Read class list
        wnids_path = os.path.join(self.root, "wnids.txt")
        with open(wnids_path, "r", encoding="utf-8") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        if split == "train":
            self._load_train()
        else:
            self._load_val()
    
    def _load_train(self):
        train_dir = os.path.join(self.root, "train")
        for class_name in self.classes:
            class_dir = os.path.join(train_dir, class_name, "images")
            if not os.path.isdir(class_dir):
                continue
            label = self.class_to_idx[class_name]
            for fname in os.listdir(class_dir):
                if fname.endswith(".JPEG"):
                    self.samples.append(os.path.join(class_dir, fname))
                    self.targets.append(label)
    
    def _load_val(self):
        val_dir = os.path.join(self.root, "val")
        annotations_path = os.path.join(val_dir, "val_annotations.txt")
        
        img_to_class = {}
        with open(annotations_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                parts = line.strip().split("\t")
                img_to_class[parts[0]] = parts[1]
        
        images_dir = os.path.join(val_dir, "images")
        for fname, class_name in img_to_class.items():
            if class_name in self.class_to_idx:
                self.samples.append(os.path.join(images_dir, fname))
                self.targets.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        label = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class SplitTinyImageNet:
    """
    Manages Split TinyImageNet: 200 classes → T tasks × C classes.
    
    Interface is identical to SplitCIFAR100 for seamless cross-dataset transfer.
    """
    
    def __init__(
        self,
        root: str = "./data",
        num_tasks: int = 10,
        classes_per_task: int = 20,
        img_size: int = 224,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.val_ratio = val_ratio
        self.rng = np.random.RandomState(seed)
        
        assert num_tasks * classes_per_task == 200, \
            f"num_tasks({num_tasks}) * classes_per_task({classes_per_task}) must equal 200"
        
        self.train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=img_size // 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.train_dataset = TinyImageNetDataset(root, "train", self.train_transform)
        self.test_dataset = TinyImageNetDataset(root, "val", self.test_transform)
        
        self.train_targets = np.array(self.train_dataset.targets)
        self.test_targets = np.array(self.test_dataset.targets)
        
        self.train_class_indices: Dict[int, np.ndarray] = {}
        self.test_class_indices: Dict[int, np.ndarray] = {}
        for c in range(200):
            self.train_class_indices[c] = np.where(self.train_targets == c)[0]
            self.test_class_indices[c] = np.where(self.test_targets == c)[0]
        
        self.class_order = list(range(200))
        self.label_map = {orig: pos for pos, orig in enumerate(self.class_order)}
    
    def set_ordering(self, ordering=None, seed=None):
        if ordering is not None:
            self.class_order = list(ordering)
        elif seed is not None:
            rng = np.random.RandomState(seed)
            self.class_order = rng.permutation(200).tolist()
        else:
            self.class_order = list(range(200))
        self.label_map = {orig: pos for pos, orig in enumerate(self.class_order)}
    
    def get_task_classes(self, task_id: int) -> List[int]:
        start = task_id * self.classes_per_task
        end = start + self.classes_per_task
        return self.class_order[start:end]
    
    def get_task(self, task_id, batch_size=64, num_workers=4):
        classes = self.get_task_classes(task_id)
        train_indices, val_indices, test_indices = [], [], []
        
        for c in classes:
            c_train = self.train_class_indices[c]
            n = len(c_train)
            n_val = max(1, int(n * self.val_ratio))
            perm = self.rng.permutation(n)
            val_indices.extend(c_train[perm[:n_val]])
            train_indices.extend(c_train[perm[n_val:]])
            test_indices.extend(self.test_class_indices[c])
        
        train_loader = DataLoader(
            RemappedSubset(self.train_dataset, train_indices, self.label_map),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            RemappedSubset(self.train_dataset, val_indices, self.label_map),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            RemappedSubset(self.test_dataset, test_indices, self.label_map),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader
    
    def get_all_test_loaders(self, up_to_task, batch_size=64, num_workers=4):
        loaders = []
        for t in range(up_to_task + 1):
            _, _, test_loader = self.get_task(t, batch_size, num_workers)
            loaders.append(test_loader)
        return loaders
    
    def get_probe_set(self, task_id, probe_size=128):
        classes = self.get_task_classes(task_id)
        indices = []
        per_class = max(1, probe_size // len(classes))
        for c in classes:
            c_idx = self.train_class_indices[c]
            chosen = self.rng.choice(c_idx, size=min(per_class, len(c_idx)), replace=False)
            indices.extend(chosen)
        indices = indices[:probe_size]
        images, labels = [], []
        for idx in indices:
            img, label = self.train_dataset[idx]
            images.append(img)
            labels.append(self.label_map[int(label)])
        return torch.stack(images), torch.tensor(labels)


def generate_orderings_tinyimagenet(num_orderings: int, base_seed: int = 0) -> List[List[int]]:
    orderings = []
    for i in range(num_orderings):
        rng = np.random.RandomState(base_seed + i)
        orderings.append(rng.permutation(200).tolist())
    return orderings
