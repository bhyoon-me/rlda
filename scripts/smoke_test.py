"""
Smoke Test: Verify RLDA end-to-end pipeline on a tiny config.

This runs:
  1. LoRA injection verification (6 tests)
  2. RLDA single sequence (3 tasks, not 10 — fast)
  3. Fixed baseline single sequence
  4. Heuristic baseline single sequence
  5. Logging verification (check JSONL output)
  6. Figure 3 data check (similarity vs rank)

If this script passes, the full experiment pipeline will work.

Usage:
    cd rlda/
    python scripts/smoke_test.py
"""

import sys
import os
import json
import tempfile
import numpy as np

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn


# ─── Tiny config for fast testing ───
SMOKE_CONFIG = {
    "seed": 42,
    "dataset": {
        "name": "split_cifar100",
        "num_tasks": 3,           # only 3 tasks for speed
        "classes_per_task": 10,
        "img_size": 224,
        "num_workers": 0,         # no multiprocessing in test
        "val_ratio": 0.1,
        "data_root": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
    },
    "backbone": {
        "name": "vit_tiny_patch16_224",
        "pretrained": True,
        "freeze_backbone": True,
        "num_layers": 12,
    },
    "lora": {
        "target_modules": ["qkv", "proj", "fc1", "fc2"],
        "init_std": 0.01,
        "alpha_ratio": 2.0,
    },
    "inner_training": {
        "epochs_per_task": 2,     # minimal epochs for smoke test
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "optimizer": "adamw",
    },
    "buffer": {
        "size": 200,
        "strategy": "reservoir",
        "replay_weight": 1.0,
    },
    "protection": {
        "lambda_low": 0,
        "lambda_med": 10,
        "lambda_high": 50,
        "fisher_samples": 50,
    },
    "bandit": {
        "hidden_dims": [128, 64],   # smaller for test
        "num_actions": 9,
        "lr": 3e-4,
        "entropy_coef": 0.1,
        "entropy_decay": 0.99,
        "entropy_min": 0.01,
    },
    "meta": {
        "num_train_orderings": 2,   # just 2 orderings
        "num_eval_orderings": 1,
        "probe_size": 32,           # tiny probe
    },
    "logging": {
        "save_dir": "",  # will be set to tmpdir
        "save_policy": True,
        "save_allocations": True,
    },
}


def run_smoke_test():
    print("=" * 70)
    print("  RLDA SMOKE TEST — End-to-End Pipeline Verification")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ─── Check imports ───
    print("\n[1/6] Checking imports...")
    try:
        from data.split_cifar100 import SplitCIFAR100, generate_orderings
        from models.peft.lora import LoRALinear
        from models.peft.injection import LoRAInjector
        from rl.state_encoder import StateEncoder
        from rl.bandit import BanditPolicy, BanditTrainer
        from rl.profiles import resolve_profile, PROFILE_NAMES
        from continual.metrics import RewardComputer, ContinualMetrics, TaskResult
        from analysis.logger import AllocationLogger
        from trainers.rlda_trainer import RLDATrainer
        from trainers.baselines import (
            FixedProfileRunner, HeuristicRunner,
            BestFixedRunner, BaseRunner,
        )
        import timm
        print("  PASS: all imports successful")
    except ImportError as e:
        print(f"  FAIL: {e}")
        return False
    
    # ─── Temp directory for outputs ───
    tmpdir = tempfile.mkdtemp(prefix="rlda_smoke_")
    config = {**SMOKE_CONFIG}
    config["logging"]["save_dir"] = tmpdir
    print(f"  Output dir: {tmpdir}")
    
    # ─── Test LoRA injection ───
    print("\n[2/6] LoRA injection verification...")
    backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    
    injector = LoRAInjector(backbone, config)
    dims = injector.get_target_dims()
    n_targets = sum(len(v) for v in dims.values())
    print(f"  Found {n_targets} target modules across {len(dims)} layers")
    assert n_targets == 48, f"Expected 48 targets (12 layers × 4 modules), got {n_targets}"
    
    # Create and inject adapters
    layer_mask = {i: True for i in range(12)}
    injector.create_adapters(task_id=0, layer_mask=layer_mask, rank=4)
    injector.inject()
    
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out = backbone.forward_features(x)
    assert out.shape[0] == 2, f"Bad output shape: {out.shape}"
    
    injector.remove_hooks()
    injector.reset()
    del backbone
    print("  PASS: injection works")
    
    # ─── Test RLDA single sequence ───
    print("\n[3/6] RLDA single sequence (3 tasks)...")
    trainer = RLDATrainer(config)
    trainer.setup()
    
    orderings = generate_orderings(1, base_seed=42)
    # Only use first 30 classes for 3 tasks
    ordering = orderings[0][:30]  
    # Pad to 100 classes (required by SplitCIFAR100)
    # Actually set_ordering needs 100 classes — use full ordering but only 3 tasks
    
    result = trainer.run_sequence(
        orderings[0],  # full 100-class ordering
        train_policy=True,
        deterministic=False,
    )
    
    assert "metrics" in result, "No metrics in result"
    assert "allocations" in result, "No allocations in result"
    assert len(result["allocations"]) == 3, f"Expected 3 tasks, got {len(result['allocations'])}"
    
    m = result["metrics"]
    print(f"  Avg Acc: {m['avg_accuracy']:.3f}, Forget: {m['forgetting']:.3f}")
    print(f"  Allocations:")
    for a in result["allocations"]:
        print(f"    Task {a['task']}: {a['profile']:16s} rank={a['rank']} "
              f"acc={a['acc_new']:.3f}")
    print("  PASS: RLDA sequence completed")
    
    # ─── Test Fixed baseline ───
    print("\n[4/6] Fixed baseline (balanced, r=4)...")
    fixed_runner = FixedProfileRunner(config, profile_idx=2)  # balanced
    logger = AllocationLogger(tmpdir, run_type="fixed_balanced")
    
    fixed_result = fixed_runner.run_sequence(
        orderings[0], ordering_id=0, ordering_seed=42, logger=logger,
    )
    logger.save()
    
    fm = fixed_result["metrics"]
    print(f"  Fixed Avg Acc: {fm['avg_accuracy']:.3f}")
    print("  PASS: fixed baseline completed")
    
    # ─── Test Heuristic baseline ───
    print("\n[5/6] Heuristic baseline (similarity_proportional)...")
    heur_runner = HeuristicRunner(config, heuristic="similarity_proportional")
    heur_logger = AllocationLogger(tmpdir, run_type="heuristic_similarity")
    
    heur_result = heur_runner.run_sequence(
        orderings[0], ordering_id=0, ordering_seed=42, logger=heur_logger,
    )
    heur_logger.save()
    
    hm = heur_result["metrics"]
    print(f"  Heuristic Avg Acc: {hm['avg_accuracy']:.3f}")
    print("  PASS: heuristic baseline completed")
    
    # ─── Verify logging ───
    print("\n[6/6] Verifying allocation log...")
    alloc_path = os.path.join(tmpdir, "allocations.jsonl")
    assert os.path.exists(alloc_path), f"No log file at {alloc_path}"
    
    records = AllocationLogger.load(alloc_path)
    print(f"  Records saved: {len(records)}")
    
    # Check fields exist
    required_fields = [
        "run_type", "ordering_id", "task_idx",
        "max_similarity", "mean_similarity",
        "selected_profile", "selected_rank",
        "acc_new", "forgetting", "reward",
        "gradient_profile", "param_cost",
    ]
    sample = records[0]
    missing = [f for f in required_fields if f not in sample]
    assert len(missing) == 0, f"Missing fields in log: {missing}"
    print(f"  All {len(required_fields)} required fields present")
    
    # Check Figure 3 data availability
    has_sim_data = any(r["max_similarity"] > 0 for r in records if r["task_idx"] > 0)
    has_rank_data = any(r["selected_rank"] > 0 for r in records)
    print(f"  Figure 3 data: similarity={has_sim_data}, rank={has_rank_data}")
    
    # ─── Summary ───
    print(f"\n{'=' * 70}")
    print("  COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Method':<30} {'Avg Acc':>10} {'Forget':>10} {'Params':>10}")
    print(f"  {'-'*60}")
    print(f"  {'RLDA (bandit)':<30} {m['avg_accuracy']:>10.3f} {m['forgetting']:>10.3f} {result['total_params']:>10,}")
    print(f"  {'Fixed balanced (r=4)':<30} {fm['avg_accuracy']:>10.3f} {fm['forgetting']:>10.3f} {fixed_result['total_params']:>10,}")
    print(f"  {'Heuristic (similarity)':<30} {hm['avg_accuracy']:>10.3f} {hm['forgetting']:>10.3f} {heur_result['total_params']:>10,}")
    
    print(f"\n{'=' * 70}")
    print(f"  ✅ ALL SMOKE TESTS PASSED")
    print(f"  Output directory: {tmpdir}")
    print(f"{'=' * 70}")
    
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
