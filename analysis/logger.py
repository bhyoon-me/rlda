"""
Allocation Logger: Structured logging for all paper figures and analysis.

Saves task-level allocation decisions in a flat JSONL format.
Each line = one task allocation in one ordering.

Fields saved per record:
  run_type          : "train" | "eval" | "baseline_fixed_X" | "heuristic_X" | "oracle" | "best_fixed"
  ordering_id       : int (which ordering)
  ordering_seed     : int (seed that generated the ordering)
  task_idx          : int (position in sequence, 0-indexed)
  task_classes      : list[int] (original CIFAR-100 class IDs for this task)
  max_similarity    : float (max cosine sim to any previous task)
  mean_similarity   : float (mean cosine sim)
  min_similarity    : float
  recent_similarity : float (sim to most recent previous task)
  gradient_profile  : list[float] (per-layer gradient norms, normalized)
  budget_fraction   : float (cumulative params used / budget_max)
  selected_profile  : str (profile name)
  selected_profile_idx : int
  selected_rank     : int
  protection_level  : str ("low" / "med" / "high")
  copy_from_task    : int or null
  num_active_layers : int
  param_cost        : int (adapter params added for this task)
  cumulative_params : int (total adapter params after this task)
  acc_new           : float (accuracy on new task after training)
  forgetting        : float (average forgetting after this task)
  avg_accuracy      : float (average accuracy across all seen tasks)
  bwt               : float (backward transfer, if computable)
  reward            : float (bandit reward)
  reward_normalized : float (after running normalization)

Usage:
    logger = AllocationLogger(save_dir="results/", run_type="train")
    logger.log_task(...)
    logger.save()     # writes allocations.jsonl
    logger.summary()  # prints aggregate stats
    
    # Later, for analysis:
    records = AllocationLogger.load("results/allocations.jsonl")
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class AllocationRecord:
    """One task allocation in one ordering."""
    # --- Context ---
    run_type: str
    ordering_id: int
    ordering_seed: int
    task_idx: int
    task_classes: List[int]
    
    # --- State signals ---
    max_similarity: float
    mean_similarity: float
    min_similarity: float
    recent_similarity: float
    gradient_profile: List[float]
    budget_fraction: float
    
    # --- Action ---
    selected_profile: str
    selected_profile_idx: int
    selected_rank: int
    protection_level: str
    copy_from_task: Optional[int]
    num_active_layers: int
    
    # --- Outcome ---
    param_cost: int
    cumulative_params: int
    acc_new: float
    forgetting: float
    avg_accuracy: float
    bwt: float
    reward: float
    reward_normalized: float


class AllocationLogger:
    """Accumulates and saves allocation records."""
    
    def __init__(self, save_dir: str, run_type: str):
        # Normalize to an absolute, OS-correct path. This prevents mixed
        # forward/back-slash paths (e.g. './results\\file') that Windows
        # rejects with Errno 22, especially under synced folders.
        self.save_dir = os.path.abspath(os.path.normpath(save_dir))
        self.run_type = run_type
        self.records: List[AllocationRecord] = []
        os.makedirs(self.save_dir, exist_ok=True)
    
    def log_task(
        self,
        ordering_id: int,
        ordering_seed: int,
        task_idx: int,
        task_classes: List[int],
        state_info: Dict[str, Any],
        action_info: Dict[str, Any],
        outcome_info: Dict[str, Any],
    ):
        """Log one task allocation."""
        record = AllocationRecord(
            run_type=self.run_type,
            ordering_id=ordering_id,
            ordering_seed=ordering_seed,
            task_idx=task_idx,
            task_classes=task_classes,
            # State
            max_similarity=state_info.get("max_similarity", 0.0),
            mean_similarity=state_info.get("mean_similarity", 0.0),
            min_similarity=state_info.get("min_similarity", 0.0),
            recent_similarity=state_info.get("recent_similarity", 0.0),
            gradient_profile=state_info.get("gradient_profile", []),
            budget_fraction=state_info.get("budget_fraction", 0.0),
            # Action
            selected_profile=action_info.get("profile", ""),
            selected_profile_idx=action_info.get("profile_idx", -1),
            selected_rank=action_info.get("rank", 0),
            protection_level=action_info.get("protection", ""),
            copy_from_task=action_info.get("copy_from_task", None),
            num_active_layers=action_info.get("num_active_layers", 0),
            # Outcome
            param_cost=outcome_info.get("param_cost", 0),
            cumulative_params=outcome_info.get("cumulative_params", 0),
            acc_new=outcome_info.get("acc_new", 0.0),
            forgetting=outcome_info.get("forgetting", 0.0),
            avg_accuracy=outcome_info.get("avg_accuracy", 0.0),
            bwt=outcome_info.get("bwt", 0.0),
            reward=outcome_info.get("reward", 0.0),
            reward_normalized=outcome_info.get("reward_normalized", 0.0),
        )
        self.records.append(record)
    
    def save(self, filename: str = "allocations.jsonl"):
        """Save all records as JSONL."""
        path = os.path.normpath(os.path.join(self.save_dir, filename))
        with open(path, "a", encoding="utf-8") as f:  # append mode — multiple runs accumulate
            for record in self.records:
                f.write(json.dumps(asdict(record), default=_json_default) + "\n")
        print(f"[Logger] Saved {len(self.records)} records to {path}")
        self.records = []  # clear after save
    
    def summary(self) -> Dict[str, float]:
        """Aggregate stats across all records."""
        if not self.records:
            return {}
        accs = [r.acc_new for r in self.records]
        forgets = [r.forgetting for r in self.records]
        rewards = [r.reward for r in self.records]
        ranks = [r.selected_rank for r in self.records]
        return {
            "mean_acc": np.mean(accs),
            "mean_forgetting": np.mean(forgets),
            "mean_reward": np.mean(rewards),
            "mean_rank": np.mean(ranks),
            "num_records": len(self.records),
        }
    
    @staticmethod
    def load(path: str) -> List[Dict]:
        """Load JSONL records."""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
