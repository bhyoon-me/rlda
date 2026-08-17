"""
LoRA Injection: Wires LoRA adapters into a timm ViT backbone's forward pass.

Strategy: Forward hooks on target Linear layers.
- No module replacement (preserves backbone structure for timm compatibility)
- Hook adds LoRA output to the original Linear output
- Hooks are registered/removed per task to control which adapters are active
- All previous task adapters contribute during inference (additive)

This is the critical bridge between:
  - LoRALinear modules (models/peft/lora.py) that compute ΔW·x
  - The backbone's actual forward pass

Architecture note (timm ViT-Tiny):
  backbone.blocks[i].attn.qkv     → Linear(192, 576)   [Q,K,V concatenated]
  backbone.blocks[i].attn.proj    → Linear(192, 192)   [attention output projection]
  backbone.blocks[i].mlp.fc1      → Linear(192, 768)   [FFN up-projection]
  backbone.blocks[i].mlp.fc2      → Linear(768, 192)   [FFN down-projection]
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Any


# Module path templates for timm ViT
# Maps our config names to actual attribute paths within a block
TIMM_VIT_MODULE_PATHS = {
    "qkv": "attn.qkv",
    "proj": "attn.proj",
    "fc1": "mlp.fc1",
    "fc2": "mlp.fc2",
}


class LoRAHookHandle:
    """Tracks a single registered hook so it can be cleanly removed."""
    
    def __init__(
        self,
        layer_idx: int,
        module_name: str,
        task_id: int,
        handle: torch.utils.hooks.RemovableHandle,
    ):
        self.layer_idx = layer_idx
        self.module_name = module_name
        self.task_id = task_id
        self.handle = handle
    
    def remove(self):
        self.handle.remove()


class LoRAInjector:
    """
    Manages LoRA injection into a timm ViT backbone via forward hooks.
    
    Lifecycle per task:
        1. create_adapters() — builds LoRALinear modules for selected layers
        2. inject()          — registers hooks that add LoRA output to backbone
        3. [train on task]
        4. freeze_task()     — freezes adapter params, keeps hooks active
    
    During inference, ALL injected adapters contribute additively:
        output = original_linear(x) + Σ_task LoRA_task(x)
    
    This matches the paper: after learning T tasks, the accumulated
    modification is ΔW^{1:T} = Σ_k α_k · B_k · A_k
    """
    
    def __init__(self, backbone: nn.Module, config: dict):
        self.backbone = backbone
        self.config = config
        self.num_layers = config["backbone"]["num_layers"]
        self.alpha_ratio = config["lora"]["alpha_ratio"]
        self.init_std = config["lora"]["init_std"]
        self.device = next(backbone.parameters()).device
        
        # Resolve module paths
        target_module_names = config["lora"]["target_modules"]
        self.module_paths = {}
        for name in target_module_names:
            if name in TIMM_VIT_MODULE_PATHS:
                self.module_paths[name] = TIMM_VIT_MODULE_PATHS[name]
            else:
                self.module_paths[name] = name
        
        # Storage
        # task_adapters[task_id][layer_idx][module_name] = LoRALinear
        self.task_adapters: Dict[int, Dict[int, Dict[str, nn.Module]]] = {}
        
        # Active hooks
        self.hooks: List[LoRAHookHandle] = []
        
        # Module reference cache: (layer_idx, module_name) -> nn.Linear
        self._module_cache: Dict[Tuple[int, str], nn.Linear] = {}
        self._build_module_cache()
        
        # Track which task's adapters are currently trainable
        self.current_task: Optional[int] = None
    
    def _build_module_cache(self):
        """Cache references to all target Linear modules in the backbone."""
        for layer_idx in range(self.num_layers):
            block = self.backbone.blocks[layer_idx]
            for name, path in self.module_paths.items():
                module = _get_nested_attr(block, path)
                if module is not None and isinstance(module, nn.Linear):
                    self._module_cache[(layer_idx, name)] = module
                else:
                    print(f"  [WARNING] Could not find {path} in block {layer_idx}")
    
    def get_target_dims(self) -> Dict[int, Dict[str, Tuple[int, int]]]:
        """
        Returns {layer_idx: {module_name: (in_features, out_features)}}.
        Used by state encoder and profile cost estimation.
        """
        dims = {}
        for (layer_idx, name), module in self._module_cache.items():
            if layer_idx not in dims:
                dims[layer_idx] = {}
            dims[layer_idx][name] = (module.in_features, module.out_features)
        return dims
    
    def create_adapters(
        self,
        task_id: int,
        layer_mask: Dict[int, bool],
        rank: int,
        copy_from_task: Optional[int] = None,
    ) -> Dict[int, Dict[str, nn.Module]]:
        """
        Create LoRA adapters for a new task.
        
        Args:
            task_id: current task identifier
            layer_mask: {layer_idx: True/False} for which layers get adapters
            rank: LoRA rank
            copy_from_task: if set, copy-init from this previous task's adapters
        
        Returns:
            Created adapter dict (also stored internally)
        """
        from models.peft.lora import LoRALinear
        
        adapters: Dict[int, Dict[str, nn.Module]] = {}
        
        for (layer_idx, name), linear in self._module_cache.items():
            if not layer_mask.get(layer_idx, False):
                continue
            
            if layer_idx not in adapters:
                adapters[layer_idx] = {}
            
            adapter = LoRALinear(
                in_features=linear.in_features,
                out_features=linear.out_features,
                rank=rank,
                alpha_ratio=self.alpha_ratio,
                init_std=self.init_std,
            ).to(self.device)
            
            # Copy-init from previous task if reuse profile
            if (copy_from_task is not None
                    and copy_from_task in self.task_adapters
                    and layer_idx in self.task_adapters[copy_from_task]
                    and name in self.task_adapters[copy_from_task][layer_idx]):
                adapter.copy_from(
                    self.task_adapters[copy_from_task][layer_idx][name]
                )
            
            adapters[layer_idx][name] = adapter
        
        self.task_adapters[task_id] = adapters
        self.current_task = task_id
        return adapters
    
    def inject(self, task_ids: Optional[List[int]] = None):
        """
        Register forward hooks for the specified tasks' adapters.
        
        If task_ids is None, inject ALL stored tasks (normal inference mode).
        Call remove_hooks() first to start clean.
        
        Hook mechanism:
            For each target Linear module, the hook intercepts its output
            and adds the LoRA contribution:
            
                hook_output = original_output + lora_adapter(input)
        """
        self.remove_hooks()
        
        if task_ids is None:
            task_ids = list(self.task_adapters.keys())
        
        for task_id in task_ids:
            if task_id not in self.task_adapters:
                continue
            
            for layer_idx, layer_adapters in self.task_adapters[task_id].items():
                for module_name, adapter in layer_adapters.items():
                    key = (layer_idx, module_name)
                    if key not in self._module_cache:
                        continue
                    
                    target_module = self._module_cache[key]
                    
                    # Create hook closure — captures adapter by reference
                    hook_fn = self._make_hook(adapter)
                    handle = target_module.register_forward_hook(hook_fn)
                    
                    self.hooks.append(LoRAHookHandle(
                        layer_idx=layer_idx,
                        module_name=module_name,
                        task_id=task_id,
                        handle=handle,
                    ))
    
    def _make_hook(self, adapter: nn.Module) -> Callable:
        """
        Create a forward hook that adds LoRA output to the original output.
        
        Hook signature: hook(module, input, output) -> modified_output
        
        Important: input is a tuple, input[0] is the actual tensor.
        The LoRA adapter takes the same input as the original Linear
        and adds its output.
        """
        def hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
            x = input[0]  # (batch, seq_len, in_features) or (batch, in_features)
            lora_out = adapter(x)  # (batch, ..., out_features)
            return output + lora_out
        
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def freeze_task(self, task_id: int):
        """Freeze all adapter parameters for a completed task."""
        if task_id in self.task_adapters:
            for layer_adapters in self.task_adapters[task_id].values():
                for adapter in layer_adapters.values():
                    for p in adapter.parameters():
                        p.requires_grad = False
    
    def unfreeze_task(self, task_id: int):
        """Unfreeze adapter parameters (for current task training)."""
        if task_id in self.task_adapters:
            for layer_adapters in self.task_adapters[task_id].values():
                for adapter in layer_adapters.values():
                    for p in adapter.parameters():
                        p.requires_grad = True
    
    def set_trainable(self, task_id: int):
        """Freeze all tasks except the specified one."""
        for tid in self.task_adapters:
            if tid == task_id:
                self.unfreeze_task(tid)
            else:
                self.freeze_task(tid)
        self.current_task = task_id
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters from the current task's adapters."""
        params = []
        if self.current_task is not None and self.current_task in self.task_adapters:
            for layer_adapters in self.task_adapters[self.current_task].values():
                for adapter in layer_adapters.values():
                    for p in adapter.parameters():
                        if p.requires_grad:
                            params.append(p)
        return params
    
    def get_all_adapter_params(self) -> List[nn.Parameter]:
        """Get ALL adapter parameters across all tasks (for saving/loading)."""
        params = []
        for task_adapters in self.task_adapters.values():
            for layer_adapters in task_adapters.values():
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
    
    def task_params(self, task_id: int) -> int:
        """Parameter count for a specific task's adapters."""
        if task_id not in self.task_adapters:
            return 0
        total = 0
        for layer_adapters in self.task_adapters[task_id].values():
            for adapter in layer_adapters.values():
                total += adapter.num_params()
        return total
    
    def summary(self) -> str:
        """Human-readable summary of injected adapters."""
        lines = ["LoRA Injection Summary:"]
        lines.append(f"  Total tasks: {len(self.task_adapters)}")
        lines.append(f"  Total params: {self.total_params():,}")
        lines.append(f"  Active hooks: {len(self.hooks)}")
        for tid in sorted(self.task_adapters.keys()):
            n_layers = len(self.task_adapters[tid])
            n_params = self.task_params(tid)
            trainable = tid == self.current_task
            lines.append(f"  Task {tid}: {n_layers} layers, {n_params:,} params"
                        f" {'[TRAINABLE]' if trainable else '[FROZEN]'}")
        return "\n".join(lines)
    
    def reset(self):
        """Full reset — remove all hooks and adapters."""
        self.remove_hooks()
        self.task_adapters.clear()
        self.current_task = None


def _get_nested_attr(module: nn.Module, path: str) -> Optional[nn.Module]:
    """Get a nested attribute by dot-separated path. Returns None if not found."""
    parts = path.split(".")
    current = module
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current
