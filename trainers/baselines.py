"""
Baseline Runners for RLDA paper.

All baselines share the same inner training loop and evaluation as RLDA.
The ONLY difference is how the allocation profile is selected.

Baselines:
  1. FixedProfileRunner   — same profile for all tasks
  2. HeuristicRunner      — rule-based profile selection from state signals
  3. BestFixedRunner      — post-hoc best single profile per ordering
  4. OracleRunner         — post-hoc best profile per task (9× compute)

Each runner outputs the same result format as RLDATrainer.run_sequence().
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

from data.split_cifar100 import SplitCIFAR100, generate_orderings
from data.split_tinyimagenet import SplitTinyImageNet, generate_orderings_tinyimagenet
from models.peft.injection import LoRAInjector
from rl.state_encoder import StateEncoder
from rl.profiles import resolve_profile, PROFILE_NAMES
from continual.metrics import RewardComputer, ContinualMetrics, TaskResult
from continual.buffer import ReservoirBuffer
from analysis.logger import AllocationLogger


class BaseRunner:
    """
    Shared infrastructure for all baselines.
    
    Subclasses only override `select_profile()`.
    Everything else — backbone, dataset, inner training, evaluation,
    logging — is identical to RLDA.
    """
    
    def __init__(self, config: dict, run_type: str = "baseline"):
        self.config = config
        self.run_type = run_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _setup_backbone(self):
        import timm
        name = self.config["backbone"]["name"]
        backbone = timm.create_model(name, pretrained=True, num_classes=0)
        backbone = backbone.to(self.device).eval()
        for p in backbone.parameters():
            p.requires_grad = False
        dc = self.config["dataset"]
        num_classes = int(dc["num_tasks"]) * int(dc["classes_per_task"])
        head = nn.Linear(backbone.embed_dim, num_classes).to(self.device)
        return backbone, head
    
    def _setup_dataset(self):
        """Create the appropriate dataset based on config."""
        dc = self.config["dataset"]
        dataset_name = dc.get("name", "split_cifar100")
        data_root = dc.get("data_root", "./data")
        if dataset_name == "split_tinyimagenet":
            return SplitTinyImageNet(
                root=data_root,
                num_tasks=int(dc["num_tasks"]),
                classes_per_task=int(dc["classes_per_task"]),
                img_size=int(dc["img_size"]),
                val_ratio=float(dc["val_ratio"]),
            )
        else:
            return SplitCIFAR100(
                root=data_root,
                num_tasks=int(dc["num_tasks"]),
                classes_per_task=int(dc["classes_per_task"]),
                img_size=int(dc["img_size"]),
                val_ratio=float(dc["val_ratio"]),
            )
    
    def _compute_budget_max(self) -> int:
        L = self.config["backbone"]["num_layers"]
        return L * 16 * 4 * 1000 * self.config["dataset"]["num_tasks"]
    
    def select_profile(
        self,
        task_idx: int,
        state_info: Dict,
        grad_profile: np.ndarray,
    ) -> int:
        """Override in subclasses. Returns profile index (0-8)."""
        raise NotImplementedError
    
    def run_sequence(
        self,
        ordering: List[int],
        ordering_id: int = 0,
        ordering_seed: int = 0,
        logger: Optional[AllocationLogger] = None,
    ) -> Dict:
        """Run one task sequence with this baseline's profile selection."""
        backbone, head = self._setup_backbone()
        
        dc = self.config["dataset"]
        dataset = self._setup_dataset()
        dataset.set_ordering(ordering)
        
        injector = LoRAInjector(backbone, self.config)
        buffer = ReservoirBuffer(capacity=int(self.config["buffer"]["size"]))
        state_encoder = StateEncoder(
            backbone=backbone,
            num_layers=self.config["backbone"]["num_layers"],
            budget_max=self._compute_budget_max(),
            t_ref=20,
            device=str(self.device),
        )
        reward_computer = RewardComputer(budget_max=self._compute_budget_max())
        
        num_tasks = dc["num_tasks"]
        metrics = ContinualMetrics(num_tasks)
        allocation_history = []
        
        for t in range(num_tasks):
            # State construction
            probe_images, probe_labels = dataset.get_probe_set(
                t, self.config["meta"]["probe_size"],
            )
            injector.inject()
            state, task_emb, grad_profile = state_encoder.construct_state(
                t, probe_images, probe_labels, head,
            )
            
            # Similarity info
            sim_summary = state_encoder.compute_similarity_summary(task_emb)
            state_info = {
                "max_similarity": sim_summary[0].item(),
                "mean_similarity": sim_summary[1].item(),
                "min_similarity": sim_summary[2].item(),
                "recent_similarity": sim_summary[3].item(),
                "gradient_profile": grad_profile.tolist(),
                "budget_fraction": state_encoder.params_used / max(self._compute_budget_max(), 1),
            }
            
            # Profile selection (baseline-specific)
            action = self.select_profile(t, state_info, grad_profile.numpy())
            
            # Configure adapters
            most_similar = state_encoder.get_most_similar_task(task_emb)
            alloc_config = resolve_profile(
                profile_idx=action,
                num_layers=self.config["backbone"]["num_layers"],
                gradient_profile=grad_profile.numpy(),
                most_similar_task=most_similar,
            )
            
            adapters = injector.create_adapters(
                task_id=t,
                layer_mask=alloc_config.layer_mask,
                rank=alloc_config.rank,
                copy_from_task=alloc_config.copy_from_task,
            )
            injector.set_trainable(t)
            injector.inject()
            
            # Inner training
            train_loader, _, _ = dataset.get_task(
                t,
                batch_size=self.config["inner_training"]["batch_size"],
                num_workers=self.config["dataset"]["num_workers"],
            )
            self._inner_train(
                backbone, head, injector, t, train_loader, alloc_config.protection,
                buffer=buffer,
            )
            
            # Evaluate
            injector.freeze_task(t)
            injector.inject()
            all_accs = self._evaluate_all_tasks(
                backbone, head, dataset, t,
            )
            
            param_cost = injector.task_params(t)
            result = TaskResult(
                task_id=t,
                accuracy=all_accs[t],
                all_accuracies=all_accs,
                param_cost=param_cost,
            )
            reward_info = reward_computer.compute(result)
            metrics.update(t, all_accs)
            state_encoder.register_task(task_emb, param_cost)
            
            # Log
            action_info = {
                "profile": PROFILE_NAMES[action],
                "profile_idx": action,
                "rank": alloc_config.rank,
                "protection": alloc_config.protection,
                "copy_from_task": alloc_config.copy_from_task,
                "num_active_layers": sum(alloc_config.layer_mask.values()),
            }
            outcome_info = {
                "param_cost": param_cost,
                "cumulative_params": injector.total_params(),
                "acc_new": reward_info["acc_new"],
                "forgetting": reward_info["forgetting"],
                "avg_accuracy": np.mean(list(all_accs.values())),
                "bwt": metrics.backward_transfer if t > 0 else 0.0,
                "reward": reward_info["reward"],
                "reward_normalized": reward_info["reward"],
            }
            
            if logger:
                logger.log_task(
                    ordering_id=ordering_id,
                    ordering_seed=ordering_seed,
                    task_idx=t,
                    task_classes=dataset.get_task_classes(t),
                    state_info=state_info,
                    action_info=action_info,
                    outcome_info=outcome_info,
                )
            
            allocation_history.append({**action_info, **outcome_info, "task": t})
            
            # Progress print (so runs don't look frozen)
            print(f"    Task {t}: profile={PROFILE_NAMES[action]:16s} "
                  f"rank={alloc_config.rank:2d}  acc={all_accs[t]:.3f}  "
                  f"forget={reward_info['forgetting']:.3f}", flush=True)
        
        injector.remove_hooks()
        
        metrics.print_final_row()
        
        return {
            "metrics": metrics.summary(),
            "allocations": allocation_history,
            "total_params": injector.total_params(),
            "final_accuracies": metrics.final_accuracies(),
        }
    
    def _inner_train(self, backbone, head, injector, task_id, train_loader, protection_level, buffer=None):
        """Same inner training as RLDA trainer (with experience replay)."""
        tc = self.config["inner_training"]
        pc = self.config["protection"]
        bc = self.config["buffer"]
        replay_weight = float(bc.get("replay_weight", 0.5))
        replay_batch = int(tc["batch_size"]) * 2
        
        adapter_params = injector.get_trainable_params()
        head_params = list(head.parameters())
        
        optimizer = torch.optim.AdamW(
            [{"params": adapter_params}, {"params": head_params}],
            lr=float(tc["lr"]),
            weight_decay=float(tc["weight_decay"]),
        )
        
        lambda_map = {"low": float(pc["lambda_low"]), "med": float(pc["lambda_med"]), "high": float(pc["lambda_high"])}
        lambda_protect = lambda_map[protection_level]
        
        param_snapshot = {id(p): p.detach().clone() for p in adapter_params if p.requires_grad}
        loss_fn = nn.CrossEntropyLoss()
        head.train()
        backbone.eval()
        
        def _feat(x):
            f = backbone.forward_features(x)
            if f.dim() == 3:
                f = f[:, 0]
            return f
        
        for epoch in range(int(tc["epochs_per_task"])):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                logits = head(_feat(images))
                loss_ce = loss_fn(logits, labels)
                
                # Experience replay
                loss_replay = torch.tensor(0.0, device=self.device)
                if buffer is not None:
                    replay = buffer.sample(replay_batch, device=str(self.device))
                    if replay is not None:
                        rx, ry = replay
                        loss_replay = loss_fn(head(_feat(rx)), ry)
                
                loss_protect = torch.tensor(0.0, device=self.device)
                for p in adapter_params:
                    if p.requires_grad and id(p) in param_snapshot:
                        loss_protect = loss_protect + ((p - param_snapshot[id(p)]) ** 2).sum()
                
                loss = loss_ce + replay_weight * loss_replay + lambda_protect * loss_protect
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter_params + head_params, max_norm=1.0)
                optimizer.step()
        
        # Add current task to buffer
        if buffer is not None:
            buffer.add_from_loader(train_loader)
    
    @torch.no_grad()
    def _evaluate_all_tasks(self, backbone, head, dataset, up_to_task) -> Dict[int, float]:
        backbone.eval()
        head.eval()
        all_accs = {}
        test_loaders = dataset.get_all_test_loaders(
            up_to_task,
            batch_size=self.config["inner_training"]["batch_size"],
            num_workers=self.config["dataset"]["num_workers"],
        )
        for t, loader in enumerate(test_loaders):
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                features = backbone.forward_features(images)
                if features.dim() == 3:
                    features = features[:, 0]
                logits = head(features)
                correct += (logits.argmax(-1) == labels).sum().item()
                total += labels.size(0)
            all_accs[t] = correct / max(total, 1)
        return all_accs


# ═══════════════════════════════════════════════════════════════
#  1. Fixed Profile Baseline
# ═══════════════════════════════════════════════════════════════

class FixedProfileRunner(BaseRunner):
    """Always selects the same profile."""
    
    def __init__(self, config: dict, profile_idx: int):
        super().__init__(config, run_type=f"baseline_fixed_{PROFILE_NAMES[profile_idx]}")
        self.profile_idx = profile_idx
    
    def select_profile(self, task_idx, state_info, grad_profile) -> int:
        return self.profile_idx


# ═══════════════════════════════════════════════════════════════
#  2. Heuristic Baselines
# ═══════════════════════════════════════════════════════════════

class HeuristicRunner(BaseRunner):
    """
    Rule-based profile selection.
    
    Three heuristics:
    - uniform: always "balanced" (profile 2)
    - similarity_proportional: rank ∝ (1 - max_similarity)
    - gradient_proportional: rank ∝ mean gradient norm
    """
    
    def __init__(self, config: dict, heuristic: str):
        super().__init__(config, run_type=f"heuristic_{heuristic}")
        self.heuristic = heuristic
    
    def select_profile(self, task_idx, state_info, grad_profile) -> int:
        if self.heuristic == "uniform":
            return 2  # balanced
        
        elif self.heuristic == "similarity_proportional":
            max_sim = state_info["max_similarity"]
            if task_idx == 0:
                return 2  # first task — no similarity info
            if max_sim > 0.85:
                return 0  # minimal
            elif max_sim > 0.7:
                return 1  # conservative
            elif max_sim > 0.5:
                return 2  # balanced
            elif max_sim > 0.3:
                return 3  # aggressive
            else:
                return 4  # plastic
        
        elif self.heuristic == "gradient_proportional":
            mean_grad = float(np.mean(grad_profile)) if len(grad_profile) > 0 else 0.5
            if mean_grad > 0.8:
                return 4  # plastic
            elif mean_grad > 0.6:
                return 3  # aggressive
            elif mean_grad > 0.4:
                return 2  # balanced
            elif mean_grad > 0.2:
                return 1  # conservative
            else:
                return 0  # minimal
        
        raise ValueError(f"Unknown heuristic: {self.heuristic}")


# ═══════════════════════════════════════════════════════════════
#  3. Best Fixed Profile (per-ordering post-hoc)
# ═══════════════════════════════════════════════════════════════

class BestFixedRunner:
    """
    Retrospective best fixed profile: for each ordering, try all 9 fixed
    profiles and report the one with highest average accuracy.
    
    This is ordering-specific post-hoc — a strong reference baseline.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.runners = [
            FixedProfileRunner(config, i) for i in range(len(PROFILE_NAMES))
        ]
    
    def run_sequence(
        self,
        ordering: List[int],
        ordering_id: int = 0,
        ordering_seed: int = 0,
        logger: Optional[AllocationLogger] = None,
    ) -> Dict:
        """Try all 9 profiles, return the best one's results."""
        best_result = None
        best_acc = -1.0
        best_profile = -1
        all_profile_results = {}
        
        for i, runner in enumerate(self.runners):
            print(f"    BestFixed: trying profile {i} ({PROFILE_NAMES[i]})...")
            result = runner.run_sequence(
                ordering, ordering_id=ordering_id, ordering_seed=ordering_seed,
            )
            avg_acc = result["metrics"]["avg_accuracy"]
            all_profile_results[PROFILE_NAMES[i]] = avg_acc
            
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_result = result
                best_profile = i
        
        # Log the winning profile's allocations
        if logger:
            winner = FixedProfileRunner(self.config, best_profile)
            winner_result = winner.run_sequence(
                ordering, ordering_id=ordering_id,
                ordering_seed=ordering_seed, logger=logger,
            )
            # Override logger's run_type
            for rec in logger.records[-self.config["dataset"]["num_tasks"]:]:
                rec.run_type = "best_fixed"
        
        best_result["best_profile"] = PROFILE_NAMES[best_profile]
        best_result["all_profile_accs"] = all_profile_results
        
        print(f"    BestFixed winner: {PROFILE_NAMES[best_profile]} "
              f"(acc={best_acc:.3f})")
        
        return best_result


# ═══════════════════════════════════════════════════════════════
#  4. Retrospective Oracle (per-task post-hoc)
# ═══════════════════════════════════════════════════════════════

class OracleRunner:
    """
    Retrospective allocation oracle: for each task in the sequence,
    try all 9 profiles and pick the one with the best reward.
    
    IMPORTANT: This is NOT a realistic method — it requires 9× compute
    and uses hindsight. It serves as an upper bound reference.
    
    Implementation: For each task, we checkpoint the model state,
    try each profile, evaluate, and keep the best one before proceeding.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def run_sequence(
        self,
        ordering: List[int],
        ordering_id: int = 0,
        ordering_seed: int = 0,
        logger: Optional[AllocationLogger] = None,
    ) -> Dict:
        """Per-task oracle selection."""
        import timm
        from copy import deepcopy
        
        dc = self.config["dataset"]
        dataset_name = dc.get("name", "split_cifar100")
        data_root = dc.get("data_root", "./data")
        if dataset_name == "split_tinyimagenet":
            dataset = SplitTinyImageNet(
                root=data_root,
                num_tasks=int(dc["num_tasks"]),
                classes_per_task=int(dc["classes_per_task"]),
                img_size=int(dc["img_size"]),
                val_ratio=float(dc["val_ratio"]),
            )
        else:
            dataset = SplitCIFAR100(
                root=data_root,
                num_tasks=int(dc["num_tasks"]),
                classes_per_task=int(dc["classes_per_task"]),
                img_size=int(dc["img_size"]),
                val_ratio=float(dc["val_ratio"]),
            )
        dataset.set_ordering(ordering)
        
        # Initial setup
        num_classes = int(dc["num_tasks"]) * int(dc["classes_per_task"])
        backbone = timm.create_model(
            self.config["backbone"]["name"], pretrained=True, num_classes=0,
        ).to(self.device).eval()
        for p in backbone.parameters():
            p.requires_grad = False
        head = nn.Linear(backbone.embed_dim, num_classes).to(self.device)
        
        injector = LoRAInjector(backbone, self.config)
        buffer = ReservoirBuffer(capacity=int(self.config["buffer"]["size"]))
        state_encoder = StateEncoder(
            backbone=backbone,
            num_layers=self.config["backbone"]["num_layers"],
            budget_max=self._compute_budget_max(),
            t_ref=20,
            device=str(self.device),
        )
        reward_computer = RewardComputer(budget_max=self._compute_budget_max())
        
        num_tasks = dc["num_tasks"]
        metrics = ContinualMetrics(num_tasks)
        allocation_history = []
        
        for t in range(num_tasks):
            probe_images, probe_labels = dataset.get_probe_set(
                t, self.config["meta"]["probe_size"],
            )
            injector.inject()
            state, task_emb, grad_profile = state_encoder.construct_state(
                t, probe_images, probe_labels, head,
            )
            
            most_similar = state_encoder.get_most_similar_task(task_emb)
            
            # Checkpoint current state
            backbone_state = deepcopy(backbone.state_dict())
            head_state = deepcopy(head.state_dict())
            injector_adapters_state = {
                tid: {
                    li: {
                        name: deepcopy(a.state_dict())
                        for name, a in la.items()
                    }
                    for li, la in ta.items()
                }
                for tid, ta in injector.task_adapters.items()
            }
            reward_state = deepcopy(reward_computer.best_accuracies)
            
            best_reward = -float("inf")
            best_profile = 0
            best_result_info = None
            
            # Try each profile
            for pi in range(len(PROFILE_NAMES)):
                print(f"    Oracle task {t}: trying profile {pi+1}/9 "
                      f"({PROFILE_NAMES[pi]})...", flush=True)
                # Restore checkpoint
                backbone.load_state_dict(backbone_state)
                head.load_state_dict(head_state)
                injector.remove_hooks()
                # Restore previous task adapters
                for tid, ta in injector_adapters_state.items():
                    if tid not in injector.task_adapters:
                        continue
                    for li, la in ta.items():
                        for name, sd in la.items():
                            injector.task_adapters[tid][li][name].load_state_dict(sd)
                
                alloc_config = resolve_profile(
                    profile_idx=pi,
                    num_layers=self.config["backbone"]["num_layers"],
                    gradient_profile=grad_profile.numpy(),
                    most_similar_task=most_similar,
                )
                
                # Remove current task adapters if they exist from previous trial
                if t in injector.task_adapters:
                    del injector.task_adapters[t]
                
                adapters = injector.create_adapters(
                    task_id=t,
                    layer_mask=alloc_config.layer_mask,
                    rank=alloc_config.rank,
                    copy_from_task=alloc_config.copy_from_task,
                )
                injector.set_trainable(t)
                injector.inject()
                
                # Train
                train_loader, _, _ = dataset.get_task(
                    t,
                    batch_size=self.config["inner_training"]["batch_size"],
                    num_workers=self.config["dataset"]["num_workers"],
                )
                self._inner_train(
                    backbone, head, injector, t, train_loader, alloc_config.protection,
                    buffer=buffer, add_to_buffer=False,
                )
                
                # Evaluate
                injector.freeze_task(t)
                injector.inject()
                all_accs = self._evaluate_all_tasks(backbone, head, dataset, t)
                
                param_cost = injector.task_params(t)
                temp_reward_computer = RewardComputer(budget_max=self._compute_budget_max())
                temp_reward_computer.best_accuracies = deepcopy(reward_state)
                result_info = temp_reward_computer.compute(TaskResult(
                    task_id=t, accuracy=all_accs[t],
                    all_accuracies=all_accs, param_cost=param_cost,
                ))
                
                if result_info["reward"] > best_reward:
                    best_reward = result_info["reward"]
                    best_profile = pi
                    best_result_info = {
                        "all_accs": all_accs,
                        "param_cost": param_cost,
                        "reward_info": result_info,
                        "alloc_config": alloc_config,
                        "backbone_state": deepcopy(backbone.state_dict()),
                        "head_state": deepcopy(head.state_dict()),
                        "task_adapter_state": deepcopy(injector.task_adapters[t]),
                    }
            
            # Commit best profile
            backbone.load_state_dict(best_result_info["backbone_state"])
            head.load_state_dict(best_result_info["head_state"])
            # Restore correct adapter for task t
            if t in injector.task_adapters:
                del injector.task_adapters[t]
            injector.task_adapters[t] = best_result_info["task_adapter_state"]
            injector.freeze_task(t)
            
            all_accs = best_result_info["all_accs"]
            reward_info = best_result_info["reward_info"]
            reward_computer.best_accuracies = deepcopy(reward_state)
            reward_computer.compute(TaskResult(
                task_id=t, accuracy=all_accs[t],
                all_accuracies=all_accs,
                param_cost=best_result_info["param_cost"],
            ))
            
            metrics.update(t, all_accs)
            state_encoder.register_task(task_emb, best_result_info["param_cost"])
            
            # Add committed task to replay buffer for future tasks
            commit_loader, _, _ = dataset.get_task(
                t,
                batch_size=self.config["inner_training"]["batch_size"],
                num_workers=self.config["dataset"]["num_workers"],
            )
            buffer.add_from_loader(commit_loader)
            
            alloc = best_result_info["alloc_config"]
            allocation_history.append({
                "task": t,
                "profile": PROFILE_NAMES[best_profile],
                "profile_idx": best_profile,
                "rank": alloc.rank,
                "acc_new": reward_info["acc_new"],
                "forgetting": reward_info["forgetting"],
                "reward": reward_info["reward"],
                "param_cost": best_result_info["param_cost"],
            })
            
            print(f"  Oracle task {t}: best={PROFILE_NAMES[best_profile]} "
                  f"acc={all_accs[t]:.3f} R={best_reward:.3f}")
        
        injector.remove_hooks()
        
        metrics.print_final_row()
        
        return {
            "metrics": metrics.summary(),
            "allocations": allocation_history,
            "total_params": injector.total_params(),
            "final_accuracies": metrics.final_accuracies(),
        }
    
    def _compute_budget_max(self):
        L = self.config["backbone"]["num_layers"]
        return L * 16 * 4 * 1000 * self.config["dataset"]["num_tasks"]
    
    def _inner_train(self, backbone, head, injector, task_id, train_loader, protection_level,
                     buffer=None, add_to_buffer=False):
        """Inner training with experience replay (oracle version)."""
        tc = self.config["inner_training"]
        pc = self.config["protection"]
        bc = self.config["buffer"]
        replay_weight = float(bc.get("replay_weight", 0.5))
        replay_batch = int(tc["batch_size"]) * 2
        adapter_params = injector.get_trainable_params()
        head_params = list(head.parameters())
        optimizer = torch.optim.AdamW(
            [{"params": adapter_params}, {"params": head_params}],
            lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]),
        )
        lambda_map = {"low": float(pc["lambda_low"]), "med": float(pc["lambda_med"]), "high": float(pc["lambda_high"])}
        lambda_protect = lambda_map[protection_level]
        param_snapshot = {id(p): p.detach().clone() for p in adapter_params if p.requires_grad}
        loss_fn = nn.CrossEntropyLoss()
        head.train()
        
        def _feat(x):
            f = backbone.forward_features(x)
            if f.dim() == 3:
                f = f[:, 0]
            return f
        
        for epoch in range(int(tc["epochs_per_task"])):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = head(_feat(images))
                loss_ce = loss_fn(logits, labels)
                
                loss_replay = torch.tensor(0.0, device=self.device)
                if buffer is not None:
                    replay = buffer.sample(replay_batch, device=str(self.device))
                    if replay is not None:
                        rx, ry = replay
                        loss_replay = loss_fn(head(_feat(rx)), ry)
                
                loss_protect = torch.tensor(0.0, device=self.device)
                for p in adapter_params:
                    if p.requires_grad and id(p) in param_snapshot:
                        loss_protect = loss_protect + ((p - param_snapshot[id(p)]) ** 2).sum()
                loss = loss_ce + replay_weight * loss_replay + lambda_protect * loss_protect
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter_params + head_params, 1.0)
                optimizer.step()
        
        if add_to_buffer and buffer is not None:
            buffer.add_from_loader(train_loader)
    
    @torch.no_grad()
    def _evaluate_all_tasks(self, backbone, head, dataset, up_to_task):
        backbone.eval()
        head.eval()
        all_accs = {}
        for t, loader in enumerate(dataset.get_all_test_loaders(
            up_to_task,
            batch_size=self.config["inner_training"]["batch_size"],
            num_workers=self.config["dataset"]["num_workers"],
        )):
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                features = backbone.forward_features(images)
                if features.dim() == 3:
                    features = features[:, 0]
                logits = head(features)
                correct += (logits.argmax(-1) == labels).sum().item()
                total += labels.size(0)
            all_accs[t] = correct / max(total, 1)
        return all_accs
