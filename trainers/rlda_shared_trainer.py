"""
Path-B pilot: learned per-task adaptation intensity on a SHARED adapter.

Motivation (from the cross-regime analysis): a single continually-trained
shared adapter + replay outperforms the entire per-task adapter family
(naive 0.622, EWC@lambda=100 0.661 vs task-modular oracle 0.583 on CIFAR-100).
The pilot asks: can the SAME policy machinery (state encoder + contextual
bandit) learn a per-task consolidation strength lambda on this winning
architecture, and beat the best fixed lambda?

Setup:
  - Architecture: one shared balanced (r=4, all layers) adapter + head,
    trained continually with replay — identical to cl_ewc.
  - Action space: lambda in {0, 10, 100, 1000}  (4 discrete actions).
  - State: the same 210-dim task state used by RLDA (task embedding,
    similarity summary, gradient profile, budget block).
  - Fisher: accumulated after every task exactly as in EWCRunner; the
    CHOSEN lambda scales the penalty during that task's training.
  - Reward: same R = alpha*Acc - beta*Forget - gamma*Cost with running
    normalization (cost is constant here, so R reduces to acc/forget
    trade-off — exactly the quantity a consolidation knob controls).

Headroom references (already measured, fixed lambda for the whole
sequence): lambda=0 -> 0.622, lambda=100 -> 0.661. The pilot succeeds if
learned per-task lambda exceeds the best fixed lambda; fixed lambda in
{10, 1000} can be obtained via EWCRunner with config override.

Run:
  python main.py --config configs/split_cifar100_mvp.yaml --mode rlda_shared
Outputs:
  rlda_shared_train_summary.json / rlda_shared_eval_summary.json /
  policy_shared.pt in logging.save_dir.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List

from rl.profiles import resolve_profile, PROFILE_NAMES
from rl.state_encoder import StateEncoder
from rl.bandit import BanditPolicy, BanditTrainer
from continual.metrics import RewardComputer, ContinualMetrics, TaskResult
from continual.buffer import ReservoirBuffer
from trainers.cl_baselines import SharedAdapterCLRunner

LAMBDA_ACTIONS = [0.0, 10.0, 100.0, 1000.0]
_BALANCED_IDX = PROFILE_NAMES.index("balanced")


class SharedIntensityTrainer(SharedAdapterCLRunner):
    """Bandit-selected per-task EWC strength on a shared adapter."""

    method_name = "rlda_shared"

    def __init__(self, config: dict):
        super().__init__(config)
        bc = config["bandit"]
        self.policy = BanditPolicy(
            state_dim=int(config["meta"].get("state_dim", 210)),
            num_actions=len(LAMBDA_ACTIONS),
            hidden_dims=[256, 128],
        ).to(self.device)
        self.bandit_trainer = BanditTrainer(
            policy=self.policy,
            lr=float(bc["lr"]),
            entropy_coef=float(bc["entropy_coef"]),
            entropy_decay=float(bc["entropy_decay"]),
            entropy_min=float(bc["entropy_min"]),
        )
        # EWC machinery (per ordering)
        self.ewc_lambda_t = 0.0
        self.fisher: Dict[int, torch.Tensor] = {}
        self.star: Dict[int, torch.Tensor] = {}
        self.fisher_batches = int(self.clc.get("fisher_batches", 20))
        # Reward normalization shared across orderings (like RLDA)
        self.reward_computer = RewardComputer(budget_max=1)

    # ── EWC penalty with the CHOSEN lambda ────────────────────────
    def extra_loss(self, t, images, logits, params, feat_fn, head, ctx=None):
        if t == 0 or not self.fisher or self.ewc_lambda_t <= 0:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)
        for p in params:
            k = id(p)
            if k in self.fisher:
                penalty = penalty + (self.fisher[k] * (p - self.star[k]) ** 2).sum()
        return (self.ewc_lambda_t / 2.0) * penalty

    def after_task(self, t, backbone, head, params, feat_fn, train_loader):
        """Accumulate Fisher and snapshot theta* (same as EWCRunner)."""
        head.eval()
        loss_fn = nn.CrossEntropyLoss()
        new_f = {id(p): torch.zeros_like(p) for p in params}
        n = 0
        for bi, (images, labels) in enumerate(train_loader):
            if bi >= self.fisher_batches:
                break
            images = images.to(self.device)
            labels = labels.to(self.device)
            for p in params:
                if p.grad is not None:
                    p.grad = None
            loss = loss_fn(head(feat_fn(images)), labels)
            loss.backward()
            for p in params:
                if p.grad is not None:
                    new_f[id(p)] += p.grad.detach() ** 2
            n += 1
        for p in params:
            if p.grad is not None:
                p.grad = None
        if n:
            for p in params:
                k = id(p)
                self.fisher[k] = self.fisher.get(k, torch.zeros_like(p)) + new_f[k] / n
                self.star[k] = p.detach().clone()
        head.train()

    # ── sequence with bandit lambda selection ─────────────────────
    def run_sequence(self, ordering, ordering_id=0, ordering_seed=0,
                    logger=None, train_policy=True) -> Dict:
        from models.peft.injection import LoRAInjector

        # fresh consolidation per ordering
        self.fisher, self.star = {}, {}

        backbone, head = self._setup_backbone()
        dataset = self._setup_dataset()
        dataset.set_ordering(ordering)
        dc = self.config["dataset"]
        num_tasks = int(dc["num_tasks"])
        self.classes_per_task = int(dc["classes_per_task"])

        injector = LoRAInjector(backbone, self.config)
        buffer = ReservoirBuffer(capacity=int(self.config["buffer"]["size"]))
        state_encoder = StateEncoder(
            backbone=backbone,
            num_layers=self.config["backbone"]["num_layers"],
            budget_max=1,  # cost constant in shared regime
            t_ref=20,
            device=str(self.device),
        )

        alloc = resolve_profile(
            profile_idx=_BALANCED_IDX,
            num_layers=self.config["backbone"]["num_layers"],
            gradient_profile=np.ones(self.config["backbone"]["num_layers"]),
            most_similar_task=None,
        )
        injector.create_adapters(task_id=0, layer_mask=alloc.layer_mask,
                                 rank=alloc.rank, copy_from_task=None)
        injector.set_trainable(0)
        injector.inject()

        adapter_params = injector.get_trainable_params()
        head_params = list(head.parameters())
        params = adapter_params + head_params

        def feat_fn(x):
            f = backbone.forward_features(x)
            if f.dim() == 3:
                f = f[:, 0]
            return f

        metrics = ContinualMetrics(num_tasks)
        history = []

        for t in range(num_tasks):
            # State (same signals as RLDA)
            probe_images, probe_labels = dataset.get_probe_set(
                t, self.config["meta"]["probe_size"],
            )
            state, task_emb, grad_profile = state_encoder.construct_state(
                t, probe_images, probe_labels, head,
            )

            # Bandit selects lambda for this task
            state = state.to(next(self.policy.parameters()).device)
            action, log_prob = self.policy.get_action(
                state, deterministic=not train_policy,
            )
            self.ewc_lambda_t = LAMBDA_ACTIONS[action]

            train_loader, _, _ = dataset.get_task(
                t,
                batch_size=self.config["inner_training"]["batch_size"],
                num_workers=self.config["dataset"]["num_workers"],
            )
            self._inner_train_shared(backbone, head, params, feat_fn, t,
                                     train_loader, buffer)
            self.after_task(t, backbone, head, params, feat_fn, train_loader)

            all_accs = self._evaluate_all_tasks(backbone, head, dataset, t)
            metrics.update(t, all_accs)
            state_encoder.register_task(task_emb, 0)

            result = TaskResult(task_id=t, accuracy=all_accs[t],
                                all_accuracies=all_accs, param_cost=0)
            reward_info = self.reward_computer.compute(result)
            if train_policy:
                self.bandit_trainer.store_transition(
                    state=state, action=action,
                    reward=reward_info["reward"], log_prob=log_prob,
                )

            history.append({
                "task": t, "lambda": self.ewc_lambda_t, "action": int(action),
                "acc": all_accs[t],
                "avg_accuracy": float(np.mean(list(all_accs.values()))),
                "forgetting": reward_info["forgetting"],
                "reward": reward_info["reward"],
            })
            print(f"    Task {t}: lambda={self.ewc_lambda_t:6.0f}  "
                  f"acc={all_accs[t]:.3f}  forget={reward_info['forgetting']:.3f}  "
                  f"R={reward_info['reward']:.3f}", flush=True)

        # REINFORCE update over the sequence (via BanditTrainer)
        if train_policy:
            pm = self.bandit_trainer.update()
            print(f"  Policy loss: {pm.get('policy_loss', 0):.4f}  "
                  f"Entropy: {pm.get('entropy', 0):.3f}  "
                  f"Coef: {pm.get('entropy_coef', 0):.4f}", flush=True)

        injector.remove_hooks()
        metrics.print_final_row()
        return {
            "metrics": metrics.summary(),
            "allocations": history,
            "total_params": injector.total_params(),
            "final_accuracies": metrics.final_accuracies(),
        }


def run_rlda_shared(config):
    """Meta-train + meta-eval the shared-intensity policy (pilot)."""
    from main import _get_orderings, _save_summary  # reuse helpers

    save_dir = config["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    trainer = SharedIntensityTrainer(config)

    n_train = int(config["meta"].get("num_train_orderings_shared",
                  config["meta"]["num_train_orderings"]))
    n_eval = int(config["meta"]["num_eval_orderings"])

    print(f"\n{'█'*70}\n  SHARED-INTENSITY META-TRAINING (Path-B pilot)\n{'█'*70}")
    print(f"  Actions (lambda): {LAMBDA_ACTIONS}")
    train_orderings = _get_orderings(config, n_train, base_seed=0)
    train_results = []
    for i, ordering in enumerate(train_orderings):
        print(f"\n{'═'*70}\n Meta-train ordering {i+1}/{n_train}\n{'═'*70}")
        train_results.append(trainer.run_sequence(ordering, i, i, train_policy=True))
    _save_summary(save_dir, "rlda_shared_train_summary.json",
                  "rlda_shared_train", train_results)

    torch.save(trainer.policy.state_dict(),
               os.path.normpath(os.path.join(save_dir, "policy_shared.pt")))

    print(f"\n{'█'*70}\n  SHARED-INTENSITY META-EVAL (zero-shot)\n{'█'*70}")
    eval_orderings = _get_orderings(config, n_eval, base_seed=10000)
    eval_results = []
    for i, ordering in enumerate(eval_orderings):
        print(f"\n{'═'*70}\n Meta-eval ordering {i+1}/{n_eval}\n{'═'*70}")
        eval_results.append(trainer.run_sequence(ordering, i, 10000+i,
                                                 train_policy=False))
    _save_summary(save_dir, "rlda_shared_eval_summary.json",
                  "rlda_shared_eval", eval_results)

    # Per-position lambda profile (quick analysis)
    lam = np.zeros((len(eval_results), 10))
    for oi, r in enumerate(eval_results):
        for a in r["allocations"]:
            lam[oi, a["task"]] = a["lambda"]
    print("\n  Eval lambda by task position (mean over orderings):")
    print("  " + "  ".join(f"T{t}:{lam[:, t].mean():.0f}" for t in range(10)))
