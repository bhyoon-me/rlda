"""
RLDA Trainer v2: Full implementation with LoRA injection via hooks.

Implements Algorithm 1 from the paper.
Key change from v1: Uses LoRAInjector (hook-based) instead of LoRAManager.
LoRA adapters are now actually wired into the backbone's forward pass.

Outer loop: iterate over task orderings (meta-training)
Inner loop: for each task in sequence:
    1. Construct state (1 fwd + 1 bwd on probe set)
    2. Select profile (bandit)
    3. Create adapters + inject hooks
    4. Train on task (standard SGD — backbone frozen, adapters + head trained)
    5. Evaluate all seen tasks & compute reward
    6. Update bandit policy
    7. Freeze current task's adapters, keep hooks active for future inference
"""

import os
import json
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple

from data.split_cifar100 import SplitCIFAR100, generate_orderings
from data.split_tinyimagenet import SplitTinyImageNet, generate_orderings_tinyimagenet
from models.peft.lora import LoRALinear
from models.peft.injection import LoRAInjector
from rl.state_encoder import StateEncoder
from rl.bandit import BanditPolicy, BanditTrainer
from rl.profiles import resolve_profile, profile_param_cost, PROFILE_NAMES
from continual.metrics import RewardComputer, ContinualMetrics, TaskResult
from continual.buffer import ReservoirBuffer


class RLDATrainer:
    """
    Main RLDA training pipeline.
    
    Usage:
        trainer = RLDATrainer(config)
        trainer.setup()
        train_results = trainer.meta_train()
        eval_results = trainer.meta_eval()
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.backbone = None
        self.head = None
        self.dataset = None
        self.injector = None        # replaces old LoRAManager
        self.state_encoder = None
        self.policy = None
        self.policy_trainer = None
        self.reward_computer = None
    
    # ─────────────────────────────────────────────────────────
    #  Setup
    # ─────────────────────────────────────────────────────────
    
    def setup(self):
        """Initialize all components. Call once before meta_train/meta_eval."""
        self._setup_backbone()
        self._setup_dataset()
        
        # --- Injector (replaces LoRAManager) ---
        self.injector = LoRAInjector(self.backbone, self.config)
        target_dims = self.injector.get_target_dims()
        print(f"[RLDA] LoRA targets discovered:")
        for li, modules in sorted(target_dims.items()):
            for name, (inf, outf) in modules.items():
                print(f"    block[{li}].{name}: Linear({inf}, {outf})")
        
        # --- State Encoder ---
        self.state_encoder = StateEncoder(
            backbone=self.backbone,
            num_layers=self.config["backbone"]["num_layers"],
            budget_max=self._compute_budget_max(),
            t_ref=20,
            device=str(self.device),
        )
        
        # Determine state dim from a dummy forward
        probe_images, probe_labels = self.dataset.get_probe_set(
            0, self.config["meta"]["probe_size"],
        )
        dummy_state, _, _ = self.state_encoder.construct_state(
            0, probe_images, probe_labels, self.head,
        )
        state_dim = dummy_state.shape[0]
        self.state_encoder.reset()
        
        # --- Bandit Policy ---
        bc = self.config["bandit"]
        self.policy = BanditPolicy(
            state_dim=state_dim,
            num_actions=bc["num_actions"],
            hidden_dims=bc["hidden_dims"],
        ).to(self.device)
        
        self.policy_trainer = BanditTrainer(
            policy=self.policy,
            lr=float(bc["lr"]),
            entropy_coef=float(bc["entropy_coef"]),
            entropy_decay=float(bc["entropy_decay"]),
            entropy_min=float(bc["entropy_min"]),
        )
        
        # --- Reward ---
        self.reward_computer = RewardComputer(
            budget_max=self._compute_budget_max(),
        )
        
        print(f"\n[RLDA] Setup complete.")
        print(f"  Backbone: {self.config['backbone']['name']}")
        print(f"  State dim: {state_dim}")
        print(f"  Policy params: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"  Num profiles: {bc['num_actions']}")
        print(f"  Device: {self.device}")
    
    def _setup_backbone(self):
        """Load pretrained ViT and create shared classification head."""
        import timm
        
        name = self.config["backbone"]["name"]
        self.backbone = timm.create_model(name, pretrained=True, num_classes=0)
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        
        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Shared incremental head — num classes depends on dataset
        dc = self.config["dataset"]
        num_classes = int(dc["num_tasks"]) * int(dc["classes_per_task"])
        embed_dim = self.backbone.embed_dim
        self.head = nn.Linear(embed_dim, num_classes).to(self.device)
    
    def _setup_dataset(self):
        dc = self.config["dataset"]
        dataset_name = dc.get("name", "split_cifar100")
        data_root = dc.get("data_root", "./data")
        
        if dataset_name == "split_tinyimagenet":
            self.dataset = SplitTinyImageNet(
                root=data_root,
                num_tasks=int(dc["num_tasks"]),
                classes_per_task=int(dc["classes_per_task"]),
                img_size=int(dc["img_size"]),
                val_ratio=float(dc["val_ratio"]),
            )
        else:
            self.dataset = SplitCIFAR100(
                root=data_root,
                num_tasks=int(dc["num_tasks"]),
                classes_per_task=int(dc["classes_per_task"]),
                img_size=int(dc["img_size"]),
                val_ratio=float(dc["val_ratio"]),
            )
    
    def _compute_budget_max(self) -> int:
        """Estimate maximum parameter budget for normalization."""
        L = self.config["backbone"]["num_layers"]
        max_rank = 16
        # ViT-Tiny: 4 modules per block, each ~192+192 or 192+768
        # Conservative estimate
        return L * max_rank * 4 * 1000 * self.config["dataset"]["num_tasks"]
    
    # ─────────────────────────────────────────────────────────
    #  Sequence reset
    # ─────────────────────────────────────────────────────────
    
    def _reset_for_sequence(self):
        """Reset model state for a fresh task sequence."""
        # Reload backbone to pristine pretrained weights
        self._setup_backbone()
        
        # Fresh injector on the new backbone
        self.injector = LoRAInjector(self.backbone, self.config)
        
        # Fresh replay buffer
        self.buffer = ReservoirBuffer(capacity=int(self.config["buffer"]["size"]))
        
        # Fresh state encoder
        self.state_encoder = StateEncoder(
            backbone=self.backbone,
            num_layers=self.config["backbone"]["num_layers"],
            budget_max=self._compute_budget_max(),
            t_ref=20,
            device=str(self.device),
        )
        
        self.reward_computer.reset()
    
    # ─────────────────────────────────────────────────────────
    #  Core: Run one task sequence (Algorithm 1, lines 2-31)
    # ─────────────────────────────────────────────────────────
    
    def run_sequence(
        self,
        ordering: List[int],
        train_policy: bool = True,
        deterministic: bool = False,
        ordering_id: int = 0,
        ordering_seed: int = 0,
        logger = None,
    ) -> Dict:
        """Run one complete task sequence."""
        self._reset_for_sequence()
        self.dataset.set_ordering(ordering)
        
        num_tasks = self.config["dataset"]["num_tasks"]
        metrics = ContinualMetrics(num_tasks)
        allocation_history = []
        reward_history = []
        
        for t in range(num_tasks):
            # ════════════════════════════════════════
            #  Phase 1: State Construction
            # ════════════════════════════════════════
            probe_images, probe_labels = self.dataset.get_probe_set(
                t, self.config["meta"]["probe_size"],
            )
            
            # Inject all previous task hooks for accurate state computation
            self.injector.inject()  # all stored tasks
            
            state, task_emb, grad_profile = self.state_encoder.construct_state(
                t, probe_images, probe_labels, self.head,
            )
            
            # ════════════════════════════════════════
            #  Phase 2: Action Selection (Bandit)
            # ════════════════════════════════════════
            state_device = state.to(self.device)
            action, log_prob = self.policy.get_action(
                state_device, deterministic=deterministic,
            )
            
            # ════════════════════════════════════════
            #  Phase 3: Adapter Configuration
            # ════════════════════════════════════════
            most_similar = self.state_encoder.get_most_similar_task(task_emb)
            alloc_config = resolve_profile(
                profile_idx=action,
                num_layers=self.config["backbone"]["num_layers"],
                gradient_profile=grad_profile.numpy(),
                most_similar_task=most_similar,
            )
            
            # Create new adapters for this task
            adapters = self.injector.create_adapters(
                task_id=t,
                layer_mask=alloc_config.layer_mask,
                rank=alloc_config.rank,
                copy_from_task=alloc_config.copy_from_task,
            )
            
            # Set only current task trainable
            self.injector.set_trainable(t)
            
            # Re-inject ALL hooks (previous frozen + current trainable)
            self.injector.inject()
            
            # ════════════════════════════════════════
            #  Phase 4: Inner Training (standard SGD)
            # ════════════════════════════════════════
            train_loader, val_loader, _ = self.dataset.get_task(
                t,
                batch_size=self.config["inner_training"]["batch_size"],
                num_workers=self.config["dataset"]["num_workers"],
            )
            
            self._inner_train(t, train_loader, alloc_config.protection)
            
            # ════════════════════════════════════════
            #  Phase 5: Evaluation & Reward
            # ════════════════════════════════════════
            # Freeze current task after training
            self.injector.freeze_task(t)
            
            # Evaluate with all hooks active (all tasks contribute)
            self.injector.inject()
            all_accs = self._evaluate_all_tasks(t)
            
            param_cost = self.injector.task_params(t)
            
            result = TaskResult(
                task_id=t,
                accuracy=all_accs[t],
                all_accuracies=all_accs,
                param_cost=param_cost,
            )
            
            reward_info = self.reward_computer.compute(result)
            metrics.update(t, all_accs)
            
            # ════════════════════════════════════════
            #  Phase 6: Policy Update
            # ════════════════════════════════════════
            if train_policy:
                self.policy_trainer.store_transition(
                    state=state,
                    action=action,
                    reward=reward_info["reward"],
                    log_prob=log_prob,
                )
            
            # ════════════════════════════════════════
            #  Phase 7: Bookkeeping
            # ════════════════════════════════════════
            self.state_encoder.register_task(task_emb, param_cost)
            
            # Extract similarity info from state vector
            sim_summary = self.state_encoder.compute_similarity_summary(task_emb)
            sim_info = {
                "max_similarity": sim_summary[0].item() if t > 0 else 0.0,
                "mean_similarity": sim_summary[1].item() if t > 0 else 0.0,
                "min_similarity": sim_summary[2].item() if t > 0 else 0.0,
                "recent_similarity": sim_summary[3].item() if t > 0 else 0.0,
            }
            
            allocation_history.append({
                "task": t,
                "profile": PROFILE_NAMES[action],
                "profile_idx": action,
                "rank": alloc_config.rank,
                "protection": alloc_config.protection,
                "copy_from": alloc_config.copy_from_task,
                "num_active_layers": sum(alloc_config.layer_mask.values()),
                "param_cost": param_cost,
                "reward": reward_info["reward"],
                "acc_new": reward_info["acc_new"],
                "forgetting": reward_info["forgetting"],
                **sim_info,
            })
            
            reward_history.append(reward_info["reward"])
            
            # Log to JSONL if logger provided
            if logger:
                logger.log_task(
                    ordering_id=ordering_id,
                    ordering_seed=ordering_seed,
                    task_idx=t,
                    task_classes=self.dataset.get_task_classes(t),
                    state_info={
                        **sim_info,
                        "gradient_profile": grad_profile.tolist(),
                        "budget_fraction": self.state_encoder.params_used / max(self._compute_budget_max(), 1),
                    },
                    action_info={
                        "profile": PROFILE_NAMES[action],
                        "profile_idx": action,
                        "rank": alloc_config.rank,
                        "protection": alloc_config.protection,
                        "copy_from_task": alloc_config.copy_from_task,
                        "num_active_layers": sum(alloc_config.layer_mask.values()),
                    },
                    outcome_info={
                        "param_cost": param_cost,
                        "cumulative_params": self.injector.total_params(),
                        "acc_new": reward_info["acc_new"],
                        "forgetting": reward_info["forgetting"],
                        "avg_accuracy": float(np.mean(list(all_accs.values()))),
                        "bwt": metrics.backward_transfer if t > 0 else 0.0,
                        "reward": reward_info["reward"],
                        "reward_normalized": reward_info["reward"],
                    },
                )
            
            print(f"  Task {t}: profile={PROFILE_NAMES[action]:16s} "
                  f"rank={alloc_config.rank:2d}  "
                  f"acc={all_accs[t]:.3f}  "
                  f"forget={reward_info['forgetting']:.3f}  "
                  f"R={reward_info['reward']:.3f}  "
                  f"params={param_cost:,}")
        
        # Update policy after full sequence
        policy_metrics = {}
        if train_policy:
            policy_metrics = self.policy_trainer.update()
        
        # Clean up hooks
        self.injector.remove_hooks()
        
        # Diagnostic: per-task final accuracy (reveals forgetting pattern)
        metrics.print_final_row()
        
        return {
            "metrics": metrics.summary(),
            "allocations": allocation_history,
            "rewards": reward_history,
            "total_params": self.injector.total_params(),
            "policy_metrics": policy_metrics,
            "final_accuracies": metrics.final_accuracies(),
        }
    
    # ─────────────────────────────────────────────────────────
    #  Inner training loop
    # ─────────────────────────────────────────────────────────
    
    def _inner_train(self, task_id: int, train_loader, protection_level: str):
        """
        Standard training loop for one task.
        
        The backbone is frozen. LoRA hooks add trainable ΔW to the forward pass.
        We optimize: current task's LoRA params + classification head.
        
        Loss = L_CE(current) + μ * L_CE(replay) + λ * L_protect
        
        Experience replay (μ * L_replay) is critical: without it, the shared
        head catastrophically forgets previous tasks.
        """
        tc = self.config["inner_training"]
        pc = self.config["protection"]
        bc = self.config["buffer"]
        replay_weight = float(bc.get("replay_weight", 0.5))
        replay_batch = int(tc["batch_size"]) * 2  # 2x replay for stronger retention
        
        # Trainable params: current task adapters + head
        adapter_params = self.injector.get_trainable_params()
        head_params = list(self.head.parameters())
        
        param_groups = [
            {"params": adapter_params, "lr": float(tc["lr"])},
            {"params": head_params, "lr": float(tc["lr"])},
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(tc["weight_decay"]),
        )
        
        lambda_map = {
            "low": float(pc["lambda_low"]),
            "med": float(pc["lambda_med"]),
            "high": float(pc["lambda_high"]),
        }
        lambda_protect = lambda_map[protection_level]
        
        param_snapshot = {
            id(p): p.detach().clone()
            for p in adapter_params if p.requires_grad
        }
        
        loss_fn = nn.CrossEntropyLoss()
        self.head.train()
        self.backbone.eval()
        
        total_loss_sum = 0.0
        total_steps = 0
        
        def _features(x):
            f = self.backbone.forward_features(x)
            if f.dim() == 3:
                f = f[:, 0]  # CLS token
            return f
        
        for epoch in range(int(tc["epochs_per_task"])):
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # ─── Current task loss ───
                logits = self.head(_features(images))
                loss_ce = loss_fn(logits, labels)
                
                # ─── Experience replay loss (prevents head forgetting) ───
                loss_replay = torch.tensor(0.0, device=self.device)
                replay = self.buffer.sample(replay_batch, device=str(self.device))
                if replay is not None:
                    rx, ry = replay
                    replay_logits = self.head(_features(rx))
                    loss_replay = loss_fn(replay_logits, ry)
                
                # ─── Protection (L2 on adapter drift) ───
                loss_protect = torch.tensor(0.0, device=self.device)
                for p in adapter_params:
                    if p.requires_grad and id(p) in param_snapshot:
                        loss_protect = loss_protect + (
                            (p - param_snapshot[id(p)]) ** 2
                        ).sum()
                
                loss = loss_ce + replay_weight * loss_replay + lambda_protect * loss_protect
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter_params + head_params, max_norm=1.0)
                optimizer.step()
                
                total_loss_sum += loss_ce.item()
                total_steps += 1
        
        # ─── Add current task samples to buffer (reservoir) ───
        self.buffer.add_from_loader(train_loader)
        
        avg_loss = total_loss_sum / max(total_steps, 1)
        return avg_loss
    
    # ─────────────────────────────────────────────────────────
    #  Evaluation
    # ─────────────────────────────────────────────────────────
    
    @torch.no_grad()
    def _evaluate_all_tasks(self, up_to_task: int) -> Dict[int, float]:
        """
        Evaluate accuracy on all tasks seen so far.
        
        All LoRA hooks are active → the backbone produces features
        modified by ALL tasks' adapters (additive).
        """
        self.backbone.eval()
        self.head.eval()
        
        all_accs = {}
        test_loaders = self.dataset.get_all_test_loaders(
            up_to_task,
            batch_size=self.config["inner_training"]["batch_size"],
            num_workers=self.config["dataset"]["num_workers"],
        )
        
        for t, loader in enumerate(test_loaders):
            correct, total = 0, 0
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                features = self.backbone.forward_features(images)
                if features.dim() == 3:
                    features = features[:, 0]
                
                logits = self.head(features)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            all_accs[t] = correct / max(total, 1)
        
        return all_accs
    
    # ─────────────────────────────────────────────────────────
    #  Meta-training & Meta-evaluation
    # ─────────────────────────────────────────────────────────
    
    def _generate_orderings(self, num_orderings: int, base_seed: int) -> List[List[int]]:
        """Generate orderings appropriate for the current dataset."""
        dataset_name = self.config["dataset"].get("name", "split_cifar100")
        if dataset_name == "split_tinyimagenet":
            return generate_orderings_tinyimagenet(num_orderings, base_seed)
        else:
            return generate_orderings(num_orderings, base_seed)
    
    def meta_train(self, logger=None) -> List[Dict]:
        """Meta-train: run policy over M orderings (Algorithm 1, lines 1-32)."""
        mc = self.config["meta"]
        train_orderings = self._generate_orderings(int(mc["num_train_orderings"]), base_seed=0)
        
        all_results = []
        for i, ordering in enumerate(train_orderings):
            print(f"\n{'═'*70}")
            print(f" Meta-train ordering {i+1}/{mc['num_train_orderings']}")
            print(f"{'═'*70}")
            
            result = self.run_sequence(
                ordering, train_policy=True, deterministic=False,
                ordering_id=i, ordering_seed=i,
                logger=logger,
            )
            all_results.append(result)
            
            m = result["metrics"]
            pm = result.get("policy_metrics", {})
            print(f"  ──── Sequence summary ────")
            print(f"  Avg Acc: {m['avg_accuracy']:.3f}  "
                  f"Forget: {m['forgetting']:.3f}  "
                  f"BWT: {m['bwt']:.3f}  "
                  f"Params: {result['total_params']:,}")
            if pm:
                print(f"  Policy loss: {pm.get('policy_loss', 0):.4f}  "
                      f"Entropy: {pm.get('entropy', 0):.3f}  "
                      f"Coef: {pm.get('entropy_coef', 0):.4f}")
        
        if logger:
            logger.save()
        
        return all_results
    
    def meta_eval(self, logger=None) -> List[Dict]:
        """Meta-eval: deploy trained policy on held-out orderings (line 35)."""
        mc = self.config["meta"]
        eval_orderings = self._generate_orderings(int(mc["num_eval_orderings"]), base_seed=10000)
        
        all_results = []
        for i, ordering in enumerate(eval_orderings):
            print(f"\n{'═'*70}")
            print(f" Meta-eval ordering {i+1}/{mc['num_eval_orderings']}")
            print(f"{'═'*70}")
            
            result = self.run_sequence(
                ordering, train_policy=False, deterministic=True,
                ordering_id=i, ordering_seed=10000+i,
                logger=logger,
            )
            all_results.append(result)
        
        if logger:
            logger.save()
        
        # Summary statistics
        accs = [r["metrics"]["avg_accuracy"] for r in all_results]
        forgets = [r["metrics"]["forgetting"] for r in all_results]
        print(f"\n{'═'*70}")
        print(f" Meta-eval Summary (zero-shot transfer)")
        print(f"{'═'*70}")
        print(f"  Avg Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"  Forgetting:   {np.mean(forgets):.3f} ± {np.std(forgets):.3f}")
        
        return all_results


# ═══════════════════════════════════════════════════════════════
#  Baseline runners
# ═══════════════════════════════════════════════════════════════

class FixedProfileRunner:
    """Run a fixed allocation profile for all tasks (baseline)."""
    
    def __init__(self, config: dict, profile_idx: int):
        self.config = config
        self.profile_idx = profile_idx
        self.profile_name = PROFILE_NAMES[profile_idx]
    
    def run_sequence(self, ordering: List[int]) -> Dict:
        """Same as RLDA but with fixed action = self.profile_idx."""
        # Create a trainer that always selects the same profile
        trainer = RLDATrainer(self.config)
        trainer._setup_backbone()
        trainer._setup_dataset()
        trainer.injector = LoRAInjector(trainer.backbone, trainer.config)
        trainer.state_encoder = StateEncoder(
            backbone=trainer.backbone,
            num_layers=trainer.config["backbone"]["num_layers"],
            budget_max=trainer._compute_budget_max(),
            t_ref=20,
            device=str(trainer.device),
        )
        trainer.reward_computer = RewardComputer(
            budget_max=trainer._compute_budget_max(),
        )
        
        # Override policy with a dummy that always returns fixed profile
        class FixedPolicy:
            def __init__(self, action):
                self._action = action
            def get_action(self, state, deterministic=False):
                return self._action, 0.0
            def parameters(self):
                return iter([])
        
        trainer.policy = FixedPolicy(self.profile_idx)
        trainer.policy_trainer = None
        trainer.dataset.set_ordering(ordering)
        
        # Run sequence without policy training
        # (simplified — reuses run_sequence logic but skips policy update)
        return trainer.run_sequence(
            ordering, train_policy=False, deterministic=True,
        )


class HeuristicRunner:
    """
    Runs a heuristic allocation strategy.
    
    Heuristics:
    - uniform: always use "balanced" profile (idx=2)
    - similarity_proportional: rank inversely proportional to max similarity
    - gradient_proportional: rank proportional to max gradient norm
    """
    
    def __init__(self, config: dict, heuristic: str):
        self.config = config
        self.heuristic = heuristic
    
    def select_profile(
        self,
        max_similarity: float,
        mean_gradient: float,
    ) -> int:
        """Select profile index based on heuristic rule."""
        if self.heuristic == "uniform":
            return 2  # balanced
        
        elif self.heuristic == "similarity_proportional":
            # High similarity → low rank (minimal/conservative)
            # Low similarity → high rank (aggressive/plastic)
            if max_similarity > 0.85:
                return 0  # minimal
            elif max_similarity > 0.7:
                return 1  # conservative
            elif max_similarity > 0.5:
                return 2  # balanced
            elif max_similarity > 0.3:
                return 3  # aggressive
            else:
                return 4  # plastic
        
        elif self.heuristic == "gradient_proportional":
            # High gradient → high rank (needs more adaptation)
            # Low gradient → low rank
            if mean_gradient > 0.8:
                return 4  # plastic
            elif mean_gradient > 0.6:
                return 3  # aggressive
            elif mean_gradient > 0.4:
                return 2  # balanced
            elif mean_gradient > 0.2:
                return 1  # conservative
            else:
                return 0  # minimal
        
        raise ValueError(f"Unknown heuristic: {self.heuristic}")
