"""
State Encoder: Constructs the compact state vector for the bandit policy.

State components:
  (a) Task embedding: mean penultimate features on probe set
  (b) Similarity summary: [max, mean, min, most_recent] cosine sim
  (c) Gradient profile: per-layer gradient norms
  (d) Budget state: [params_used/budget_max, t/(t+T_ref)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


class StateEncoder:
    """
    Constructs the bandit state vector from model + task data.
    
    All computation is lightweight:
    - 1 forward pass for task embedding
    - 1 backward pass for gradient profile
    - Cosine similarity computation (trivial)
    - Budget bookkeeping (trivial)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_layers: int,
        budget_max: int,
        t_ref: int = 20,
        device: str = "cuda",
    ):
        self.backbone = backbone
        self.num_layers = num_layers
        self.budget_max = budget_max
        self.t_ref = t_ref
        self.device = device
        
        # History
        self.task_embeddings: List[torch.Tensor] = []
        self.params_used = 0
    
    @property
    def state_dim(self) -> int:
        """
        State dimension: d_e + 4 + L + 2
        d_e is determined at first call (depends on backbone).
        """
        if len(self.task_embeddings) > 0:
            d_e = self.task_embeddings[0].shape[0]
        else:
            # Estimate from backbone — will be set properly on first call
            d_e = 192  # ViT-Tiny default
        return d_e + 4 + self.num_layers + 2
    
    def reset(self):
        """Reset for a new task sequence."""
        self.task_embeddings = []
        self.params_used = 0
    
    @torch.no_grad()
    def compute_task_embedding(
        self,
        probe_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute task embedding: mean of penultimate-layer features.
        
        Args:
            probe_images: (N, C, H, W) tensor
        
        Returns:
            (d_e,) tensor — task embedding
        """
        self.backbone.eval()
        images = probe_images.to(self.device)
        
        # Get features before classification head
        # timm ViTs: forward_features returns (B, num_patches+1, dim)
        # We take the CLS token
        features = self.backbone.forward_features(images)
        if features.dim() == 3:
            features = features[:, 0]  # CLS token
        
        embedding = features.mean(dim=0)  # (d_e,)
        return embedding.cpu()
    
    def compute_similarity_summary(
        self,
        current_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity summary: [max, mean, min, most_recent].
        
        Returns zeros for the first task (no history).
        """
        if len(self.task_embeddings) == 0:
            return torch.zeros(4)
        
        sims = []
        for emb in self.task_embeddings:
            sim = F.cosine_similarity(
                current_embedding.unsqueeze(0),
                emb.unsqueeze(0),
            ).item()
            sims.append(sim)
        
        sims = np.array(sims)
        return torch.tensor([
            sims.max(),
            sims.mean(),
            sims.min(),
            sims[-1],  # most recent task
        ], dtype=torch.float32)
    
    def compute_gradient_profile(
        self,
        probe_images: torch.Tensor,
        probe_labels: torch.Tensor,
        head: nn.Module,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
    ) -> torch.Tensor:
        """
        Compute per-layer gradient norms from one backward pass on probe set.
        
        Returns:
            (L,) tensor — gradient norm per layer
        """
        self.backbone.eval()
        head.train()
        
        images = probe_images.to(self.device)
        labels = probe_labels.to(self.device)
        
        # Enable gradients temporarily for backbone
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        
        # Forward
        features = self.backbone.forward_features(images)
        if features.dim() == 3:
            features = features[:, 0]
        logits = head(features)
        loss = loss_fn(logits, labels)
        
        # Backward
        loss.backward()
        
        # Collect per-layer gradient norms
        grad_norms = []
        for layer_idx in range(self.num_layers):
            block = self.backbone.blocks[layer_idx]
            layer_norm = 0.0
            count = 0
            for p in block.parameters():
                if p.grad is not None:
                    layer_norm += p.grad.norm().item() ** 2
                    count += 1
            grad_norms.append(np.sqrt(layer_norm) if count > 0 else 0.0)
        
        # Clean up
        self.backbone.zero_grad()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        
        # Normalize to [0, 1] range for stability
        grad_norms = np.array(grad_norms)
        if grad_norms.max() > 0:
            grad_norms = grad_norms / grad_norms.max()
        
        return torch.tensor(grad_norms, dtype=torch.float32)
    
    def compute_budget_state(self, task_idx: int) -> torch.Tensor:
        """
        Budget state: [params_used/budget_max, t/(t+T_ref)].
        
        Uses saturating counter for task progress (no assumption on T).
        """
        budget_frac = self.params_used / max(self.budget_max, 1)
        task_frac = task_idx / (task_idx + self.t_ref)
        return torch.tensor([budget_frac, task_frac], dtype=torch.float32)
    
    def construct_state(
        self,
        task_idx: int,
        probe_images: torch.Tensor,
        probe_labels: torch.Tensor,
        head: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full state construction pipeline.
        
        Returns:
            state: (state_dim,) tensor — full state vector
            task_embedding: (d_e,) tensor — stored for future similarity
            gradient_profile: (L,) tensor — used by selective profiles
        """
        # (a) Task embedding — 1 forward pass
        task_emb = self.compute_task_embedding(probe_images)
        
        # (b) Similarity summary — trivial
        sim_summary = self.compute_similarity_summary(task_emb)
        
        # (c) Gradient profile — 1 backward pass
        grad_profile = self.compute_gradient_profile(
            probe_images, probe_labels, head,
        )
        
        # (d) Budget state — trivial
        budget = self.compute_budget_state(task_idx)
        
        # Concatenate
        state = torch.cat([task_emb, sim_summary, grad_profile, budget])
        
        return state, task_emb, grad_profile
    
    def register_task(self, task_embedding: torch.Tensor, param_cost: int):
        """Record a completed task's embedding and parameter usage."""
        self.task_embeddings.append(task_embedding)
        self.params_used += param_cost
    
    def get_most_similar_task(self, current_embedding: torch.Tensor) -> Optional[int]:
        """Return the task ID most similar to the current task, or None."""
        if len(self.task_embeddings) == 0:
            return None
        
        best_sim = -1.0
        best_task = 0
        for i, emb in enumerate(self.task_embeddings):
            sim = F.cosine_similarity(
                current_embedding.unsqueeze(0),
                emb.unsqueeze(0),
            ).item()
            if sim > best_sim:
                best_sim = sim
                best_task = i
        
        return best_task
