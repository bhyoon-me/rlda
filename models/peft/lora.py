"""
Dynamic-rank LoRA implementation.

Supports:
- Creating adapters with specified rank per layer
- Copy-init from a previous adapter (for reuse profiles)
- Freezing/unfreezing individual adapters
- Computing total adapter parameter count
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class LoRALinear(nn.Module):
    """
    LoRA adapter for a single linear layer.
    
    Wraps a frozen linear layer with a low-rank update:
        output = frozen_linear(x) + (x @ A^T @ B^T) * scaling
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha_ratio: float = 2.0,
        init_std: float = 0.01,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha_ratio * rank
        self.scaling = self.alpha / self.rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with small random, B with zeros (so ΔW = 0 at init)
        nn.init.normal_(self.lora_A, std=init_std)
        # B is already zeros
        
        self.enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the LoRA update: x @ A^T @ B^T * scaling."""
        if not self.enabled or self.rank == 0:
            return torch.zeros(
                *x.shape[:-1], self.out_features, 
                device=x.device, dtype=x.dtype,
            )
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
    
    def num_params(self) -> int:
        return self.rank * (self.in_features + self.out_features)
    
    @torch.no_grad()
    def copy_from(self, other: "LoRALinear"):
        """Copy-init from another LoRA adapter (for reuse profiles)."""
        if self.rank == other.rank:
            self.lora_A.copy_(other.lora_A)
            self.lora_B.copy_(other.lora_B)
        elif self.rank < other.rank:
            # Take top-rank components (by B column norm)
            norms = other.lora_B.norm(dim=0)
            _, top_idx = norms.topk(self.rank)
            self.lora_A.copy_(other.lora_A[top_idx])
            self.lora_B.copy_(other.lora_B[:, top_idx])
        else:
            # Copy what exists, zero-init the rest
            r_copy = other.rank
            self.lora_A[:r_copy].copy_(other.lora_A)
            self.lora_B[:, :r_copy].copy_(other.lora_B)
            nn.init.normal_(self.lora_A[r_copy:], std=0.01)
            # B remainder already zeros from init


class LoRAManager:
    """
    Manages LoRA adapters across all layers of a backbone.
    
    Handles:
    - Creating adapters per profile specification
    - Tracking adapters across tasks (for copy-init reuse)
    - Computing total parameter count
    - Enabling/disabling adapters
    """
    
    def __init__(self, backbone: nn.Module, config: dict):
        self.backbone = backbone
        self.config = config
        self.num_layers = config["backbone"]["num_layers"]
        self.alpha_ratio = config["lora"]["alpha_ratio"]
        self.init_std = config["lora"]["init_std"]
        
        # Storage: task_id -> {layer_idx -> LoRALinear}
        self.task_adapters: Dict[int, Dict[int, Dict[str, LoRALinear]]] = {}
        
        # Current active adapters
        self.active_adapters: Dict[int, Dict[str, LoRALinear]] = {}
        
        # Discover target modules in backbone
        self.target_layers = self._discover_target_layers()
    
    def _discover_target_layers(self) -> Dict[int, Dict[str, Tuple[int, int]]]:
        """
        Find target linear layers in each transformer block.
        Returns: {layer_idx: {module_name: (in_features, out_features)}}
        """
        layers = {}
        target_names = self.config["lora"]["target_modules"]
        
        for layer_idx in range(self.num_layers):
            layers[layer_idx] = {}
            # Assumes timm-style ViT: backbone.blocks[layer_idx].attn.qkv, etc.
            block = self.backbone.blocks[layer_idx]
            
            for name in target_names:
                module = _get_submodule(block, name)
                if module is not None and isinstance(module, nn.Linear):
                    layers[layer_idx][name] = (module.in_features, module.out_features)
        
        return layers
    
    def create_adapters(
        self,
        task_id: int,
        layer_mask: Dict[int, bool],  # which layers get adapters
        rank: int,
        copy_from_task: Optional[int] = None,
    ) -> Dict[int, Dict[str, LoRALinear]]:
        """
        Create LoRA adapters for a new task.
        
        Args:
            task_id: current task ID
            layer_mask: {layer_idx: True/False} — which layers get adapters
            rank: LoRA rank for this task
            copy_from_task: if set, copy-init from this task's adapters
        
        Returns:
            {layer_idx: {module_name: LoRALinear}}
        """
        adapters = {}
        
        for layer_idx, modules in self.target_layers.items():
            if not layer_mask.get(layer_idx, False):
                continue
            
            adapters[layer_idx] = {}
            for name, (in_f, out_f) in modules.items():
                adapter = LoRALinear(
                    in_features=in_f,
                    out_features=out_f,
                    rank=rank,
                    alpha_ratio=self.alpha_ratio,
                    init_std=self.init_std,
                )
                
                # Copy-init if reusing
                if (copy_from_task is not None 
                    and copy_from_task in self.task_adapters
                    and layer_idx in self.task_adapters[copy_from_task]
                    and name in self.task_adapters[copy_from_task][layer_idx]):
                    adapter.copy_from(
                        self.task_adapters[copy_from_task][layer_idx][name]
                    )
                
                adapters[layer_idx][name] = adapter
        
        # Store and activate
        self.task_adapters[task_id] = adapters
        self.active_adapters = adapters
        
        return adapters
    
    def get_adapter_params(self) -> list:
        """Get all trainable parameters from active adapters."""
        params = []
        for layer_adapters in self.active_adapters.values():
            for adapter in layer_adapters.values():
                params.extend(adapter.parameters())
        return params
    
    def total_params(self) -> int:
        """Total adapter parameters across all tasks."""
        total = 0
        for task_adapters in self.task_adapters.values():
            for layer_adapters in task_adapters.values():
                for adapter in layer_adapters.values():
                    total += adapter.num_params()
        return total
    
    def freeze_previous_adapters(self, current_task: int):
        """Freeze all adapters except the current task's."""
        for task_id, task_adapters in self.task_adapters.items():
            requires_grad = (task_id == current_task)
            for layer_adapters in task_adapters.values():
                for adapter in layer_adapters.values():
                    for p in adapter.parameters():
                        p.requires_grad = requires_grad


def _get_submodule(module: nn.Module, name: str) -> Optional[nn.Module]:
    """Safely get a submodule by dot-separated name."""
    parts = name.split(".")
    current = module
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current
