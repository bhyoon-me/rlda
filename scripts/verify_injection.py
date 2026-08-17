"""
Verification: Test that LoRA injection actually modifies backbone outputs.

Run this FIRST to confirm the hook-based LoRA works before any experiments.

Tests:
1. Backbone output changes after injection (ΔW ≠ 0 once trained)
2. Backbone output is identical before injection (ΔW = 0 at init)
3. Multiple task adapters accumulate additively
4. Freezing works correctly
5. Copy-init produces non-zero initialization
6. Gradient flows through adapters but not backbone
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np


def test_lora_injection():
    """Full verification of LoRA injection system."""
    
    print("=" * 60)
    print("LoRA Injection Verification")
    print("=" * 60)
    
    # ─── Setup ───
    try:
        import timm
    except ImportError:
        print("SKIP: timm not installed. Run: pip install timm")
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load ViT-Tiny
    backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    
    config = {
        "backbone": {"num_layers": 12},
        "lora": {
            "target_modules": ["qkv", "proj", "fc1", "fc2"],
            "alpha_ratio": 2.0,
            "init_std": 0.01,
        },
    }
    
    # Dummy input
    x = torch.randn(2, 3, 224, 224, device=device)
    
    from models.peft.injection import LoRAInjector
    
    # ─── Test 1: Output identical at init (B=0 → ΔW=0) ───
    print("\n[Test 1] Output unchanged at initialization (B=0)...")
    injector = LoRAInjector(backbone, config)
    
    with torch.no_grad():
        out_before = backbone.forward_features(x).clone()
    
    # Create adapters for "task 0" on all layers
    layer_mask = {i: True for i in range(12)}
    injector.create_adapters(task_id=0, layer_mask=layer_mask, rank=4)
    injector.inject()
    
    with torch.no_grad():
        out_after = backbone.forward_features(x).clone()
    
    diff = (out_before - out_after).abs().max().item()
    assert diff < 1e-5, f"FAIL: output changed by {diff} at init (should be ~0)"
    print(f"  PASS: max diff = {diff:.2e} (expected ~0 because B=0)")
    
    injector.remove_hooks()
    injector.reset()
    
    # ─── Test 2: Output changes after training adapter ───
    print("\n[Test 2] Output changes after adapter training...")
    injector = LoRAInjector(backbone, config)
    
    with torch.no_grad():
        out_clean = backbone.forward_features(x)[:, 0].clone()
    
    injector.create_adapters(task_id=0, layer_mask=layer_mask, rank=4)
    injector.inject()
    
    # Simulate training: update B so ΔW ≠ 0
    head = nn.Linear(backbone.embed_dim, 10).to(device)
    adapter_params = injector.get_trainable_params()
    optimizer = torch.optim.Adam(list(head.parameters()) + adapter_params, lr=1e-3)
    labels = torch.randint(0, 10, (2,), device=device)
    
    for step in range(5):
        features = backbone.forward_features(x)
        if features.dim() == 3:
            features = features[:, 0]
        logits = head(features)
        loss = nn.CrossEntropyLoss()(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        out_trained = backbone.forward_features(x)[:, 0].clone()
    
    diff = (out_clean - out_trained).abs().max().item()
    assert diff > 1e-4, f"FAIL: output didn't change after training (diff={diff})"
    print(f"  PASS: max diff = {diff:.4f} (output changed after training)")
    
    # ─── Test 3: Gradient flows through adapters, not backbone ───
    print("\n[Test 3] Gradient flows through adapters only...")
    
    backbone_grads = []
    for p in backbone.parameters():
        if p.grad is not None:
            backbone_grads.append(p.grad.abs().sum().item())
    backbone_has_grad = sum(backbone_grads) > 0
    
    adapter_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in adapter_params
    )
    
    assert not backbone_has_grad, "FAIL: backbone received gradients"
    assert adapter_has_grad, "FAIL: adapters didn't receive gradients"
    print(f"  PASS: backbone grad = 0, adapter grad > 0")
    
    # ─── Test 4: Multiple tasks accumulate ───
    print("\n[Test 4] Multiple task adapters accumulate additively...")
    injector.freeze_task(0)
    
    # Add task 1 adapters
    injector.create_adapters(task_id=1, layer_mask=layer_mask, rank=8)
    injector.set_trainable(1)
    
    # Manually set B to non-zero for task 1 to force visible change
    for layer_adapters in injector.task_adapters[1].values():
        for adapter in layer_adapters.values():
            nn.init.normal_(adapter.lora_B, std=0.1)
    
    # Inject both tasks
    injector.inject()
    
    with torch.no_grad():
        out_two_tasks = backbone.forward_features(x)[:, 0].clone()
    
    # Should differ from single-task output
    diff = (out_trained - out_two_tasks).abs().max().item()
    assert diff > 1e-4, f"FAIL: adding task 1 didn't change output (diff={diff})"
    print(f"  PASS: two-task output differs from one-task by {diff:.4f}")
    
    # Verify hook count
    assert len(injector.hooks) == 2 * 12 * 4, \
        f"FAIL: expected {2*12*4} hooks, got {len(injector.hooks)}"
    print(f"  PASS: {len(injector.hooks)} hooks active (2 tasks × 12 layers × 4 modules)")
    
    # ─── Test 5: Freeze works ───
    print("\n[Test 5] Frozen adapters don't update...")
    task0_params_before = [
        p.clone() for task_adapters in [injector.task_adapters[0]]
        for la in task_adapters.values() for a in la.values()
        for p in a.parameters()
    ]
    
    # Train only task 1
    optimizer2 = torch.optim.Adam(injector.get_trainable_params(), lr=1e-2)
    for step in range(3):
        features = backbone.forward_features(x)[:, 0]
        logits = head(features)
        loss = nn.CrossEntropyLoss()(logits, labels)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
    
    task0_params_after = [
        p.clone() for task_adapters in [injector.task_adapters[0]]
        for la in task_adapters.values() for a in la.values()
        for p in a.parameters()
    ]
    
    task0_changed = any(
        (b - a).abs().max().item() > 1e-7
        for b, a in zip(task0_params_before, task0_params_after)
    )
    assert not task0_changed, "FAIL: frozen task 0 params changed"
    print(f"  PASS: task 0 params unchanged after training task 1")
    
    # ─── Test 6: Copy-init works ───
    print("\n[Test 6] Copy-init produces non-zero initialization...")
    injector2 = LoRAInjector(backbone, config)
    
    # Create task 0 with random trained weights
    injector2.create_adapters(task_id=0, layer_mask=layer_mask, rank=4)
    for la in injector2.task_adapters[0].values():
        for a in la.values():
            nn.init.normal_(a.lora_A, std=0.5)
            nn.init.normal_(a.lora_B, std=0.5)
    
    # Create task 1 with copy-init from task 0
    injector2.create_adapters(
        task_id=1, layer_mask=layer_mask, rank=4, copy_from_task=0,
    )
    
    # Check that task 1 adapters are initialized from task 0
    for layer_idx in injector2.task_adapters[1]:
        for name in injector2.task_adapters[1][layer_idx]:
            a0 = injector2.task_adapters[0][layer_idx][name]
            a1 = injector2.task_adapters[1][layer_idx][name]
            diff_A = (a0.lora_A - a1.lora_A).abs().max().item()
            diff_B = (a0.lora_B - a1.lora_B).abs().max().item()
            assert diff_A < 1e-6, f"FAIL: A not copied at layer {layer_idx}.{name}"
            assert diff_B < 1e-6, f"FAIL: B not copied at layer {layer_idx}.{name}"
    
    print(f"  PASS: all adapter weights correctly copied")
    
    # ─── Summary ───
    print(f"\n{'=' * 60}")
    print(f"  Injector summary:")
    print(injector.summary())
    
    print(f"\n{'=' * 60}")
    print("  ALL TESTS PASSED")
    print(f"{'=' * 60}")
    
    return True


if __name__ == "__main__":
    success = test_lora_injection()
    sys.exit(0 if success else 1)
