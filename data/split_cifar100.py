"""
Split CIFAR-100 dataset for class-incremental continual learning.

Splits 100 classes into T tasks of C classes each.
Supports random class orderings for meta-training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional, Dict


class RemappedSubset(Dataset):
    """
    Wraps a dataset subset and remaps original class labels to global
    contiguous indices based on class ordering.

    In class-incremental learning with a shared head of size
    (num_tasks * classes_per_task), labels must be in [0, total_classes).
    Raw CIFAR-100 labels are 0-99 in arbitrary order, so we map each
    original class ID to its position in class_order.

    Example: if class_order = [47, 3, 88, ...], then:
        original label 47 → 0
        original label 3  → 1
        original label 88 → 2
    """

    def __init__(self, base_dataset, indices, label_map: Dict[int, int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, original_label = self.base_dataset[self.indices[i]]
        remapped_label = self.label_map[int(original_label)]
        return img, remapped_label


class SplitCIFAR100:
    """
    Manages Split CIFAR-100: 100 classes → T tasks × C classes.
    
    Each call to get_task(t) returns train/val loaders for task t's classes,
    with the class-to-task mapping determined by the current ordering.
    """
    
    def __init__(
        self,
        root: str = "./data",
        num_tasks: int = 10,
        classes_per_task: int = 10,
        img_size: int = 224,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.val_ratio = val_ratio
        self.rng = np.random.RandomState(seed)
        
        # Allow partial splits for smoke testing
        self.total_classes = min(num_tasks * classes_per_task, 100)
        
        # Transforms: standard ImageNet-style for ViT
        self.train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=img_size // 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            ),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            ),
        ])
        
        # Load full datasets
        self.train_dataset = datasets.CIFAR100(
            root=root, train=True, download=True, transform=self.train_transform,
        )
        self.test_dataset = datasets.CIFAR100(
            root=root, train=False, download=True, transform=self.test_transform,
        )
        
        # Build class-to-indices mapping
        self.train_targets = np.array(self.train_dataset.targets)
        self.test_targets = np.array(self.test_dataset.targets)
        
        self.train_class_indices: Dict[int, np.ndarray] = {}
        self.test_class_indices: Dict[int, np.ndarray] = {}
        for c in range(100):
            self.train_class_indices[c] = np.where(self.train_targets == c)[0]
            self.test_class_indices[c] = np.where(self.test_targets == c)[0]
        
        # Default ordering
        self.class_order = list(range(100))
        self.label_map = {orig: pos for pos, orig in enumerate(self.class_order)}
    
    def set_ordering(self, ordering: Optional[List[int]] = None, seed: Optional[int] = None):
        """Set the class ordering for this task sequence."""
        if ordering is not None:
            assert len(ordering) == 100 and set(ordering) == set(range(100))
            self.class_order = list(ordering)
        elif seed is not None:
            rng = np.random.RandomState(seed)
            self.class_order = rng.permutation(100).tolist()
        else:
            self.class_order = list(range(100))
        # Build label map: original class ID → global contiguous index
        self.label_map = {orig: pos for pos, orig in enumerate(self.class_order)}
    
    def get_task_classes(self, task_id: int) -> List[int]:
        """Get the original class IDs for a given task."""
        start = task_id * self.classes_per_task
        end = start + self.classes_per_task
        return self.class_order[start:end]
    
    def get_task(
        self, 
        task_id: int, 
        batch_size: int = 64, 
        num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns (train_loader, val_loader, test_loader) for the given task.
        
        Train/val split is done per-class to maintain class balance.
        """
        classes = self.get_task_classes(task_id)
        
        # Collect indices
        train_indices, val_indices, test_indices = [], [], []
        
        for c in classes:
            c_train = self.train_class_indices[c]
            # Split into train/val
            n = len(c_train)
            n_val = max(1, int(n * self.val_ratio))
            perm = self.rng.permutation(n)
            val_indices.extend(c_train[perm[:n_val]])
            train_indices.extend(c_train[perm[n_val:]])
            # Test
            test_indices.extend(self.test_class_indices[c])
        
        train_loader = DataLoader(
            RemappedSubset(self.train_dataset, train_indices, self.label_map),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=False,
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
    
    def get_all_test_loaders(
        self, 
        up_to_task: int, 
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> List[DataLoader]:
        """Get test loaders for tasks 0..up_to_task (inclusive)."""
        loaders = []
        for t in range(up_to_task + 1):
            _, _, test_loader = self.get_task(t, batch_size, num_workers)
            loaders.append(test_loader)
        return loaders
    
    def get_probe_set(
        self, 
        task_id: int, 
        probe_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a small probe subset for state construction.
        Returns (images, labels) tensors.
        """
        classes = self.get_task_classes(task_id)
        indices = []
        per_class = max(1, probe_size // len(classes))
        
        for c in classes:
            c_idx = self.train_class_indices[c]
            chosen = self.rng.choice(c_idx, size=min(per_class, len(c_idx)), replace=False)
            indices.extend(chosen)
        
        # Truncate to exact probe_size
        indices = indices[:probe_size]
        
        images, labels = [], []
        for idx in indices:
            img, label = self.train_dataset[idx]
            images.append(img)
            labels.append(self.label_map[int(label)])
        
        return torch.stack(images), torch.tensor(labels)


def generate_orderings(num_orderings: int, base_seed: int = 0) -> List[List[int]]:
    """Generate random class orderings for meta-training/eval."""
    orderings = []
    for i in range(num_orderings):
        rng = np.random.RandomState(base_seed + i)
        orderings.append(rng.permutation(100).tolist())
    return orderings
