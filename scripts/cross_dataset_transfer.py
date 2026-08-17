"""
Cross-Dataset Transfer Experiment.

Train allocation policy on Split CIFAR-100 → Deploy zero-shot on Split TinyImageNet.

This is the key novelty-boosting experiment:
  - If it works: "policy learns dataset-agnostic allocation principles"
  - If it partially works: "policy captures transferable structure with some degradation"
  - If it fails: "allocation is dataset-specific" (still useful as negative result)

Usage:
    python scripts/cross_dataset_transfer.py --config configs/split_cifar100_mvp.yaml

The script:
  1. Trains RLDA policy on Split CIFAR-100 (or loads pretrained)
  2. Deploys the SAME policy on Split TinyImageNet (zero-shot, no policy update)
  3. Compares against fixed baselines on TinyImageNet
  4. Reports transfer gap
"""

import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from data.split_cifar100 import SplitCIFAR100, generate_orderings
from data.split_tinyimagenet import SplitTinyImageNet, generate_orderings_tinyimagenet
from models.peft.injection import LoRAInjector
from rl.state_encoder import StateEncoder
from rl.bandit import BanditPolicy, BanditTrainer
from rl.profiles import resolve_profile, PROFILE_NAMES
from continual.metrics import RewardComputer, ContinualMetrics, TaskResult
from analysis.logger import AllocationLogger


def load_config(path):
    with open(path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dc = config.get("dataset", {})
    data_root = dc.get("data_root", "./data")
    if not os.path.isabs(data_root):
        data_root = os.path.abspath(os.path.join(repo_dir, data_root))
    dc["data_root"] = data_root
    os.makedirs(data_root, exist_ok=True)
    config["dataset"] = dc
    return config


class CrossDatasetRunner:
    """
    Runs a pretrained policy on a different dataset.
    
    Key design: the state encoder produces normalized, relative features,
    so a policy trained on CIFAR-100 states can interpret TinyImageNet states.
    """
    
    def __init__(self, config: dict, policy: BanditPolicy):
        self.config = config
        self.policy = policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def run_on_tinyimagenet(
        self,
        num_orderings: int = 10,
        num_tasks: int = 10,
        classes_per_task: int = 20,
    ) -> dict:
        """Deploy CIFAR-100-trained policy on TinyImageNet zero-shot."""
        import timm
        
        results = []
        orderings = generate_orderings_tinyimagenet(num_orderings, base_seed=20000)
        
        # TinyImageNet config (override dataset settings)
        tin_config = {**self.config}
        tin_config["dataset"] = {
            "name": "split_tinyimagenet",
            "num_tasks": num_tasks,
            "classes_per_task": classes_per_task,
            "img_size": 224,
            "num_workers": 4,
            "val_ratio": 0.1,
        }
        
        for i, ordering in enumerate(orderings):
            print(f"\n{'═'*70}")
            print(f" Cross-dataset transfer: TinyImageNet ordering {i+1}/{num_orderings}")
            print(f"{'═'*70}")
            
            # Fresh backbone + dataset
            backbone = timm.create_model(
                self.config["backbone"]["name"], pretrained=True, num_classes=0,
            ).to(self.device).eval()
            for p in backbone.parameters():
                p.requires_grad = False
            
            # Head for 200 classes (TinyImageNet)
            head = torch.nn.Linear(backbone.embed_dim, 200).to(self.device)
            
            # Fresh injector + state encoder
            injector = LoRAInjector(backbone, self.config)
            from continual.buffer import ReservoirBuffer
            buffer = ReservoirBuffer(capacity=int(self.config["buffer"]["size"]))
            state_encoder = StateEncoder(
                backbone=backbone,
                num_layers=self.config["backbone"]["num_layers"],
                budget_max=self._budget_max(num_tasks),
                t_ref=20,
                device=str(self.device),
            )
            reward_computer = RewardComputer(budget_max=self._budget_max(num_tasks))
            
            # Dataset
            dataset = SplitTinyImageNet(
                root=self.config["dataset"].get("data_root", "./data"),
                num_tasks=num_tasks,
                classes_per_task=classes_per_task,
                img_size=224,
            )
            dataset.set_ordering(ordering)
            
            metrics = ContinualMetrics(num_tasks)
            allocation_history = []
            
            for t in range(num_tasks):
                # State construction (same as CIFAR-100 training)
                probe_images, probe_labels = dataset.get_probe_set(
                    t, self.config["meta"]["probe_size"],
                )
                injector.inject()
                state, task_emb, grad_profile = state_encoder.construct_state(
                    t, probe_images, probe_labels, head,
                )
                
                # Policy decision (ZERO-SHOT — no update)
                state_device = state.to(self.device)
                with torch.no_grad():
                    action, _ = self.policy.get_action(state_device, deterministic=True)
                
                # Configure adapters
                most_similar = state_encoder.get_most_similar_task(task_emb)
                alloc_config = resolve_profile(
                    profile_idx=action,
                    num_layers=self.config["backbone"]["num_layers"],
                    gradient_profile=grad_profile.numpy(),
                    most_similar_task=most_similar,
                )
                
                injector.create_adapters(
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
                self._inner_train(backbone, head, injector, t, train_loader, alloc_config.protection, buffer=buffer)
                
                # Evaluate
                injector.freeze_task(t)
                injector.inject()
                all_accs = self._evaluate_all(backbone, head, dataset, t)
                
                param_cost = injector.task_params(t)
                reward_info = reward_computer.compute(TaskResult(
                    task_id=t, accuracy=all_accs[t],
                    all_accuracies=all_accs, param_cost=param_cost,
                ))
                metrics.update(t, all_accs)
                state_encoder.register_task(task_emb, param_cost)
                
                print(f"  Task {t}: {PROFILE_NAMES[action]:16s} "
                      f"rank={alloc_config.rank:2d} acc={all_accs[t]:.3f}")
                
                allocation_history.append({
                    "task": t, "profile": PROFILE_NAMES[action],
                    "rank": alloc_config.rank, "acc_new": all_accs[t],
                })
            
            injector.remove_hooks()
            metrics.print_final_row()
            result = {"metrics": metrics.summary(), "allocations": allocation_history,
                      "total_params": injector.total_params(),
                      "final_accuracies": metrics.final_accuracies()}
            results.append(result)
            
            print(f"  → Avg Acc: {result['metrics']['avg_accuracy']:.3f}")
        
        return results
    
    def _budget_max(self, num_tasks):
        return self.config["backbone"]["num_layers"] * 16 * 4 * 1000 * num_tasks
    
    def _inner_train(self, backbone, head, injector, task_id, train_loader, protection, buffer=None):
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
        lam = lambda_map[protection]
        snapshot = {id(p): p.detach().clone() for p in adapter_params if p.requires_grad}
        loss_fn = torch.nn.CrossEntropyLoss()
        head.train()
        
        def _feat(x):
            f = backbone.forward_features(x)
            if f.dim() == 3:
                f = f[:, 0]
            return f
        
        for epoch in range(int(tc["epochs_per_task"])):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                loss = loss_fn(head(_feat(images)), labels)
                
                if buffer is not None:
                    replay = buffer.sample(replay_batch, device=str(self.device))
                    if replay is not None:
                        rx, ry = replay
                        loss = loss + replay_weight * loss_fn(head(_feat(rx)), ry)
                
                for p in adapter_params:
                    if p.requires_grad and id(p) in snapshot:
                        loss = loss + lam * ((p - snapshot[id(p)]) ** 2).sum()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter_params + head_params, 1.0)
                optimizer.step()
        
        if buffer is not None:
            buffer.add_from_loader(train_loader)
    
    @torch.no_grad()
    def _evaluate_all(self, backbone, head, dataset, up_to_task):
        backbone.eval(); head.eval()
        all_accs = {}
        for t, loader in enumerate(dataset.get_all_test_loaders(
            up_to_task, batch_size=self.config["inner_training"]["batch_size"],
            num_workers=self.config["dataset"]["num_workers"],
        )):
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                features = backbone.forward_features(images)
                if features.dim() == 3:
                    features = features[:, 0]
                correct += (head(features).argmax(-1) == labels).sum().item()
                total += labels.size(0)
            all_accs[t] = correct / max(total, 1)
        return all_accs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/split_cifar100_mvp.yaml")
    parser.add_argument("--policy_path", type=str, default=None,
                        help="Path to pretrained policy. If None, train first.")
    parser.add_argument("--num_orderings", type=int, default=10)
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = config["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    # ─── Step 1: Get trained policy ───
    if args.policy_path and os.path.exists(args.policy_path):
        print(f"Loading pretrained policy from {args.policy_path}")
        # Need to determine state_dim — use CIFAR-100 probe
        from trainers.rlda_trainer import RLDATrainer
        trainer = RLDATrainer(config)
        trainer.setup()
        policy = trainer.policy
        policy.load_state_dict(torch.load(args.policy_path, map_location=device))
    else:
        print("No pretrained policy found. Training on CIFAR-100 first...")
        from trainers.rlda_trainer import RLDATrainer
        trainer = RLDATrainer(config)
        trainer.setup()
        trainer.meta_train()
        policy = trainer.policy
        torch.save(policy.state_dict(), os.path.join(save_dir, "policy.pt"))
    
    policy.eval()
    
    # ─── Step 2: Deploy on TinyImageNet (zero-shot) ───
    print("\n" + "█" * 70)
    print("  CROSS-DATASET TRANSFER: CIFAR-100 → TinyImageNet")
    print("█" * 70)
    
    runner = CrossDatasetRunner(config, policy)
    transfer_results = runner.run_on_tinyimagenet(num_orderings=args.num_orderings)
    
    transfer_accs = [r["metrics"]["avg_accuracy"] for r in transfer_results]
    transfer_forgets = [r["metrics"]["forgetting"] for r in transfer_results]
    
    # ─── Step 3: Fixed baseline on TinyImageNet for comparison ───
    print("\n" + "█" * 70)
    print("  BASELINE: Fixed Balanced (r=4) on TinyImageNet")
    print("█" * 70)
    
    # Create a dummy policy that always selects "balanced"
    class FixedPolicy:
        def get_action(self, state, deterministic=False):
            return 2, 0.0  # balanced profile
    
    fixed_runner = CrossDatasetRunner(config, FixedPolicy())
    fixed_results = fixed_runner.run_on_tinyimagenet(num_orderings=args.num_orderings)
    
    fixed_accs = [r["metrics"]["avg_accuracy"] for r in fixed_results]
    
    # ─── Step 4: Summary ───
    print(f"\n{'═'*70}")
    print(f"  CROSS-DATASET TRANSFER RESULTS")
    print(f"{'═'*70}")
    print(f"  {'Method':<40} {'Avg Acc':>10} {'± std':>8}")
    print(f"  {'-'*58}")
    print(f"  {'RLDA (trained on CIFAR-100)':<40} {np.mean(transfer_accs):>10.3f} {np.std(transfer_accs):>7.3f}")
    print(f"  {'Fixed balanced r=4 (on TinyImageNet)':<40} {np.mean(fixed_accs):>10.3f} {np.std(fixed_accs):>7.3f}")
    
    gap = np.mean(transfer_accs) - np.mean(fixed_accs)
    print(f"\n  Transfer advantage: {gap:+.3f}")
    
    if gap > 0.01:
        print(f"  ✅ Cross-dataset transfer successful — policy generalizes!")
    elif gap > -0.01:
        print(f"  ⚠️  Comparable performance — partial transfer")
    else:
        print(f"  ❌ Transfer degradation — policy may be dataset-specific")
    
    # Save
    summary = {
        "experiment": "cross_dataset_transfer",
        "source": "split_cifar100",
        "target": "split_tinyimagenet",
        "rlda_transfer_acc_mean": float(np.mean(transfer_accs)),
        "rlda_transfer_acc_std": float(np.std(transfer_accs)),
        "fixed_baseline_acc_mean": float(np.mean(fixed_accs)),
        "fixed_baseline_acc_std": float(np.std(fixed_accs)),
        "transfer_advantage": float(gap),
    }
    with open(os.path.join(save_dir, "cross_dataset_transfer.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Results saved to {save_dir}/cross_dataset_transfer.json")


if __name__ == "__main__":
    main()
