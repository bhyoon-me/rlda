"""
Allocation Profiles: Maps the 9 discrete profiles to concrete adapter configurations.

This is the bridge between the bandit's action (profile index) and the
LoRA manager's create_adapters() call.
"""

import numpy as np
import torch
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class AllocationConfig:
    """Concrete adapter configuration derived from a profile."""
    layer_mask: Dict[int, bool]      # which layers get adapters
    rank: int                         # LoRA rank
    protection: str                   # "low", "med", "high"
    copy_from_task: Optional[int]     # task to copy-init from (None = fresh)


PROFILE_NAMES = [
    "minimal",          # 0
    "conservative",     # 1
    "balanced",         # 2
    "aggressive",       # 3
    "plastic",          # 4
    "selective_low",    # 5
    "selective_high",   # 6
    "reuse_low",        # 7
    "reuse_high",       # 8
]


def resolve_profile(
    profile_idx: int,
    num_layers: int,
    gradient_profile: Optional[np.ndarray] = None,
    most_similar_task: Optional[int] = None,
    attention_layers: Optional[List[int]] = None,
) -> AllocationConfig:
    """
    Convert a profile index (0-8) into a concrete AllocationConfig.
    
    Args:
        profile_idx: index into PROFILE_NAMES
        num_layers: total number of transformer blocks
        gradient_profile: per-layer gradient norms (for selective profiles)
        most_similar_task: task ID of most similar previous task (for reuse)
        attention_layers: indices of attention-containing layers
    
    Returns:
        AllocationConfig with all fields populated
    """
    if attention_layers is None:
        attention_layers = list(range(num_layers))  # assume all layers have attention
    
    name = PROFILE_NAMES[profile_idx]
    
    # --- Layer selection ---
    if name == "minimal":
        # Top 2 layers only
        active = set(range(num_layers - 2, num_layers))
        
    elif name == "conservative":
        # Attention layers only (in ViT, all layers have attention, 
        # so this means qkv/proj but not fc1/fc2 — handled at LoRA level)
        active = set(attention_layers)
        
    elif name in ("balanced", "aggressive", "plastic"):
        # All layers
        active = set(range(num_layers))
        
    elif name in ("selective_low", "selective_high"):
        # Gradient-top-50% layers
        if gradient_profile is not None:
            top_k = max(1, num_layers // 2)
            top_indices = np.argsort(gradient_profile)[-top_k:]
            active = set(top_indices.tolist())
        else:
            # Fallback: top half of layers
            active = set(range(num_layers // 2, num_layers))
            
    elif name == "reuse_low":
        # Copy-init from most similar + top 2 layers new
        active = set(range(num_layers - 2, num_layers))
        
    elif name == "reuse_high":
        # Copy-init from most similar + all layers new
        active = set(range(num_layers))
    else:
        raise ValueError(f"Unknown profile: {name}")
    
    layer_mask = {i: (i in active) for i in range(num_layers)}
    
    # --- Rank ---
    rank_map = {
        "minimal": 2, "conservative": 4, "balanced": 4,
        "aggressive": 8, "plastic": 16,
        "selective_low": 4, "selective_high": 8,
        "reuse_low": 2, "reuse_high": 4,
    }
    rank = rank_map[name]
    
    # --- Protection ---
    protection_map = {
        "minimal": "high", "conservative": "high", "balanced": "med",
        "aggressive": "med", "plastic": "low",
        "selective_low": "med", "selective_high": "med",
        "reuse_low": "high", "reuse_high": "med",
    }
    protection = protection_map[name]
    
    # --- Copy-init source ---
    copy_from = None
    if name in ("reuse_low", "reuse_high") and most_similar_task is not None:
        copy_from = most_similar_task
    
    return AllocationConfig(
        layer_mask=layer_mask,
        rank=rank,
        protection=protection,
        copy_from_task=copy_from,
    )


def profile_param_cost(
    profile_idx: int,
    num_layers: int, 
    layer_dims: Dict[int, Dict[str, tuple]],
    gradient_profile: Optional[np.ndarray] = None,
) -> int:
    """
    Compute the parameter cost of a profile without actually creating adapters.
    Used for the Cost term in reward.
    """
    config = resolve_profile(
        profile_idx, num_layers, gradient_profile=gradient_profile,
    )
    
    total = 0
    for layer_idx, active in config.layer_mask.items():
        if not active:
            continue
        if layer_idx in layer_dims:
            for name, (in_f, out_f) in layer_dims[layer_idx].items():
                total += config.rank * (in_f + out_f)
    
    return total
