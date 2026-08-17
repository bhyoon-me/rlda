"""
Standard continual-learning baselines: EWC and LwF.

These differ from the allocation baselines (fixed/heuristic) in a fundamental
way: they are not allocation strategies. Both use the classic single-model CL
setting — ONE shared set of LoRA adapters (balanced, r=4, all layers) created
at task 0 and kept trainable for the whole sequence — plus a regularization
objective from the continual-learning literature:

  EWCRunner : Elastic Weight Consolidation (Kirkpatrick et al., 2017).
              After each task, a diagonal Fisher estimate is accumulated over
              the trainable parameters; subsequent tasks pay a quadratic
              penalty lambda/2 * sum_i F_i (theta_i - theta*_i)^2.

  LwFRunner : Learning without Forgetting (Li & Hoiem, 2016).
              Before each new task, the model is snapshotted; during training,
              a knowledge-distillation loss matches the new model's logits on
              previously-seen classes to the snapshot's logits (temperature T).

Matched conditions: both runners share the SAME frozen backbone, the SAME
replay buffer, and the SAME inner-training schedule as every other method in
the paper — so they are EWC+replay and LwF+replay, i.e. deliberately strong
versions of these baselines. The comparison isolates the *objective* (their
regularizers, our learned allocation) rather than the training budget.

Config (all optional, read from config["cl_baselines"] with defaults):
  ewc_lambda:      EWC penalty strength           (default 100.0)
  fisher_batches:  batches used for Fisher est.   (default 20)
  lwf_alpha:       KD loss weight                 (default 1.0)
  lwf_temperature: KD softmax temperature         (default 2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

from rl.profiles import resolve_profile, PROFILE_NAMES
from continual.metrics import ContinualMetrics
from continual.buffer import ReservoirBuffer
from trainers.baselines import BaseRunner

# Profile index for "balanced" (r=4, all layers) — the shared adapter config.
_BALANCED_IDX = PROFILE_NAMES.index("balanced")


class SharedAdapterCLRunner(BaseRunner):
    """
    Shared infrastructure for classic single-model CL baselines.

    One set of adapters is created at task 0 and remains trainable for every
    task. Subclasses hook into the loop via:
        before_task(t, ...)   — e.g. snapshot the model (LwF)
        extra_loss(...)       — the method-specific regularizer
        after_task(t, ...)    — e.g. Fisher consolidation (EWC)
    """

    method_name = "shared_cl"

    def __init__(self, config: dict):
        super().__init__(config, run_type=f"baseline_{self.method_name}")
        self.clc = dict(config.get("cl_baselines", {}) or {})

    # ── subclass hooks ────────────────────────────────────────────
    def before_task(self, t, backbone, head, params, feat_fn):
        pass

    def pre_batch(self, t, images, params, feat_fn, head):
        """Called BEFORE any forward pass of the batch. May run no-grad
        forwards with swapped (snapshot) parameters; doing this before the
        training graph is built avoids autograd version conflicts."""
        return None

    def extra_loss(self, t, images, logits, params, feat_fn, head, ctx=None):
        return torch.tensor(0.0, device=self.device)

    def after_task(self, t, backbone, head, params, feat_fn, train_loader):
        pass

    # ── main loop ─────────────────────────────────────────────────
    def run_sequence(
        self,
        ordering: List[int],
        ordering_id: int = 0,
        ordering_seed: int = 0,
        logger=None,  # accepted for API compatibility; no allocation to log
    ) -> Dict:
        from models.peft.injection import LoRAInjector

        backbone, head = self._setup_backbone()
        dataset = self._setup_dataset()
        dataset.set_ordering(ordering)
        dc = self.config["dataset"]
        num_tasks = int(dc["num_tasks"])
        self.classes_per_task = int(dc["classes_per_task"])

        injector = LoRAInjector(backbone, self.config)
        buffer = ReservoirBuffer(capacity=int(self.config["buffer"]["size"]))

        # ── Create ONE shared adapter set at task 0 (balanced, all layers) ──
        alloc = resolve_profile(
            profile_idx=_BALANCED_IDX,
            num_layers=self.config["backbone"]["num_layers"],
            gradient_profile=np.ones(self.config["backbone"]["num_layers"]),
            most_similar_task=None,
        )
        injector.create_adapters(
            task_id=0,
            layer_mask=alloc.layer_mask,
            rank=alloc.rank,
            copy_from_task=None,
        )
        injector.set_trainable(0)
        injector.inject()

        adapter_params = injector.get_trainable_params()
        head_params = list(head.parameters())
        params = adapter_params + head_params  # stable objects across tasks

        def feat_fn(x):
            f = backbone.forward_features(x)
            if f.dim() == 3:
                f = f[:, 0]
            return f

        metrics = ContinualMetrics(num_tasks)
        allocation_history = []

        for t in range(num_tasks):
            self.before_task(t, backbone, head, params, feat_fn)

            train_loader, _, _ = dataset.get_task(
                t,
                batch_size=self.config["inner_training"]["batch_size"],
                num_workers=self.config["dataset"]["num_workers"],
            )
            self._inner_train_shared(
                backbone, head, params, feat_fn, t, train_loader, buffer,
            )
            self.after_task(t, backbone, head, params, feat_fn, train_loader)

            all_accs = self._evaluate_all_tasks(backbone, head, dataset, t)
            metrics.update(t, all_accs)

            allocation_history.append({
                "task": t,
                "profile": "shared_balanced",
                "rank": alloc.rank,
                "param_cost": injector.task_params(0) if t == 0 else 0,
                "avg_accuracy": float(np.mean(list(all_accs.values()))),
            })
            print(f"    Task {t}: method={self.method_name:8s} "
                  f"acc={all_accs[t]:.3f}  "
                  f"avg={np.mean(list(all_accs.values())):.3f}", flush=True)

        injector.remove_hooks()
        metrics.print_final_row()

        return {
            "metrics": metrics.summary(),
            "allocations": allocation_history,
            "total_params": injector.total_params(),
            "final_accuracies": metrics.final_accuracies(),
        }

    # ── training with method-specific extra loss ──────────────────
    def _inner_train_shared(self, backbone, head, params, feat_fn, t,
                            train_loader, buffer):
        tc = self.config["inner_training"]
        bc = self.config["buffer"]
        replay_weight = float(bc.get("replay_weight", 0.5))
        replay_batch = int(tc["batch_size"]) * 2

        optimizer = torch.optim.AdamW(
            params, lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]),
        )
        loss_fn = nn.CrossEntropyLoss()
        head.train()
        backbone.eval()

        for _ in range(int(tc["epochs_per_task"])):
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Snapshot-model forwards (e.g. LwF old logits) MUST happen
                # before the training graph is built — parameter swap-in/out
                # after the new forward corrupts autograd version counters.
                ctx = self.pre_batch(t, images, params, feat_fn, head)

                logits = head(feat_fn(images))
                loss = loss_fn(logits, labels)

                # Experience replay (matched with all other methods)
                if buffer is not None:
                    replay = buffer.sample(replay_batch, device=str(self.device))
                    if replay is not None:
                        rx, ry = replay
                        loss = loss + replay_weight * loss_fn(head(feat_fn(rx)), ry)

                # Method-specific regularizer (EWC penalty / LwF distillation)
                loss = loss + self.extra_loss(t, images, logits, params, feat_fn, head, ctx)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

        if buffer is not None:
            buffer.add_from_loader(train_loader)


# ═══════════════════════════════════════════════════════════════
#  EWC
# ═══════════════════════════════════════════════════════════════

class EWCRunner(SharedAdapterCLRunner):
    """EWC + replay on a shared balanced adapter set."""

    method_name = "ewc"

    def __init__(self, config: dict):
        super().__init__(config)
        self.ewc_lambda = float(self.clc.get("ewc_lambda", 100.0))
        self.fisher_batches = int(self.clc.get("fisher_batches", 20))
        self.fisher: Dict[int, torch.Tensor] = {}   # id(p) -> accumulated F
        self.star: Dict[int, torch.Tensor] = {}     # id(p) -> theta*

    def run_sequence(self, *args, **kwargs):
        # Fresh consolidation state per ordering.
        self.fisher, self.star = {}, {}
        return super().run_sequence(*args, **kwargs)

    def extra_loss(self, t, images, logits, params, feat_fn, head, ctx=None):
        if t == 0 or not self.fisher:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)
        for p in params:
            k = id(p)
            if k in self.fisher:
                penalty = penalty + (self.fisher[k] * (p - self.star[k]) ** 2).sum()
        return (self.ewc_lambda / 2.0) * penalty

    def after_task(self, t, backbone, head, params, feat_fn, train_loader):
        """Accumulate a diagonal empirical Fisher estimate and snapshot theta*."""
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
        if n == 0:
            return
        for p in params:
            k = id(p)
            f = new_f[k] / n
            self.fisher[k] = self.fisher.get(k, torch.zeros_like(p)) + f
            self.star[k] = p.detach().clone()
        head.train()


# ═══════════════════════════════════════════════════════════════
#  Naive shared control (architecture-effect decomposition)
# ═══════════════════════════════════════════════════════════════

class NaiveSharedRunner(SharedAdapterCLRunner):
    """Shared balanced adapter + replay, NO regularizer.

    Control experiment: if this matches EWC's accuracy, the gain over the
    per-task adapter family comes from the shared-adapter architecture (a
    single continually-trained adapter can rebalance old-task features),
    not from the EWC penalty itself.
    """

    method_name = "shared_naive"


# ═══════════════════════════════════════════════════════════════
#  LwF
# ═══════════════════════════════════════════════════════════════

class LwFRunner(SharedAdapterCLRunner):
    """LwF + replay on a shared balanced adapter set."""

    method_name = "lwf"

    def __init__(self, config: dict):
        super().__init__(config)
        self.alpha = float(self.clc.get("lwf_alpha", 1.0))
        self.T = float(self.clc.get("lwf_temperature", 2.0))
        self._old: Optional[List[torch.Tensor]] = None

    def run_sequence(self, *args, **kwargs):
        self._old = None
        return super().run_sequence(*args, **kwargs)

    def before_task(self, t, backbone, head, params, feat_fn):
        """Snapshot the model (adapter + head params) before training task t."""
        if t > 0:
            self._old = [p.detach().clone() for p in params]
        else:
            self._old = None

    @torch.no_grad()
    def pre_batch(self, t, images, params, feat_fn, head):
        """Compute snapshot-model logits BEFORE the training forward.
        Swap-in/swap-out is safe here because no autograd graph exists yet
        for this batch. LoRA + head params are small (~0.3M), so the
        per-batch copy cost is negligible."""
        if t == 0 or self._old is None:
            return None
        cur = [p.detach().clone() for p in params]
        for p, o in zip(params, self._old):
            p.copy_(o)
        logits_old = head(feat_fn(images))
        for p, c in zip(params, cur):
            p.copy_(c)
        return logits_old

    def extra_loss(self, t, images, logits, params, feat_fn, head, ctx=None):
        if t == 0 or ctx is None:
            return torch.tensor(0.0, device=self.device)
        logits_old = ctx
        # Distill only over classes seen BEFORE the current task. Labels are
        # remapped to a contiguous global index by ordering position, so tasks
        # 0..t-1 occupy exactly the first t*classes_per_task columns.
        n_seen = t * self.classes_per_task
        if n_seen <= 0:
            return torch.tensor(0.0, device=self.device)
        p_old = F.softmax(logits_old[:, :n_seen] / self.T, dim=1)
        log_p_new = F.log_softmax(logits[:, :n_seen] / self.T, dim=1)
        kd = F.kl_div(log_p_new, p_old, reduction="batchmean") * (self.T ** 2)
        return self.alpha * kd
