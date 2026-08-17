"""
Reservoir Memory Buffer for Experience Replay.

This implements the replay component referenced in Algorithm 1
("Train on D_t with replay from M"). Without it, the shared
classification head catastrophically forgets previous tasks.

Reservoir sampling guarantees each seen sample has equal probability
of being in the buffer, regardless of arrival order — standard for
class-incremental continual learning (Chaudhry et al., 2019).
"""

import torch
import numpy as np
from typing import Tuple, Optional


class ReservoirBuffer:
    """
    Fixed-size memory buffer with reservoir sampling.

    Stores (image, label) pairs on CPU to save GPU memory.
    Samples are moved to device only when retrieved for replay.
    """

    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self.images = None      # (capacity, C, H, W) on CPU
        self.labels = None      # (capacity,) on CPU
        self.n_seen = 0         # total samples seen (for reservoir prob)
        self.n_stored = 0       # current count in buffer

    def __len__(self):
        return self.n_stored

    def is_empty(self) -> bool:
        return self.n_stored == 0

    def _init_storage(self, sample_img: torch.Tensor):
        """Lazy-init storage tensors once we know image shape."""
        c, h, w = sample_img.shape
        self.images = torch.zeros(self.capacity, c, h, w, dtype=torch.float32)
        self.labels = torch.zeros(self.capacity, dtype=torch.long)

    def add_batch(self, images: torch.Tensor, labels: torch.Tensor):
        """
        Add a batch of samples via reservoir sampling.

        Args:
            images: (B, C, H, W) tensor (any device)
            labels: (B,) tensor
        """
        images = images.detach().cpu()
        labels = labels.detach().cpu()

        if self.images is None:
            self._init_storage(images[0])

        for i in range(images.shape[0]):
            self.n_seen += 1
            if self.n_stored < self.capacity:
                # Buffer not full — just append
                self.images[self.n_stored] = images[i]
                self.labels[self.n_stored] = labels[i]
                self.n_stored += 1
            else:
                # Reservoir: replace with probability capacity/n_seen
                j = np.random.randint(0, self.n_seen)
                if j < self.capacity:
                    self.images[j] = images[i]
                    self.labels[j] = labels[i]

    def add_from_loader(self, loader, max_batches: Optional[int] = None):
        """Add samples from a DataLoader via reservoir sampling."""
        for b, (images, labels) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            self.add_batch(images, labels)

    def sample(self, batch_size: int, device: str = "cpu") -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample a random batch from the buffer.

        Returns None if buffer is empty.
        """
        if self.n_stored == 0:
            return None
        n = min(batch_size, self.n_stored)
        idx = np.random.choice(self.n_stored, size=n, replace=False)
        imgs = self.images[idx].to(device)
        lbls = self.labels[idx].to(device)
        return imgs, lbls

    def reset(self):
        """Clear the buffer (between task sequences)."""
        self.images = None
        self.labels = None
        self.n_seen = 0
        self.n_stored = 0
