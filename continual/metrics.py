"""
Reward computation for RLDA.

Reward: R_t = α * Acc_new − β * Forget − γ * Cost

All terms are self-contained (no external baseline dependency).
Running normalization applied per-term for stability.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TaskResult:
    """Results from training on one task."""
    task_id: int
    accuracy: float           # accuracy on this task after training
    all_accuracies: Dict[int, float]  # {task_id: accuracy} for all seen tasks
    param_cost: int           # adapter parameters added for this task


class RewardComputer:
    """
    Computes the reward signal after each task allocation + training.
    
    Tracks per-task best accuracies for forgetting computation.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 0.5,
        budget_max: int = 1_000_000,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.budget_max = budget_max
        
        # Per-task best accuracy (for forgetting)
        self.best_accuracies: Dict[int, float] = {}
        
        # Cumulative param cost
        self.total_params = 0
    
    def reset(self):
        """Reset for a new task sequence."""
        self.best_accuracies = {}
        self.total_params = 0
    
    def compute(self, result: TaskResult) -> Dict[str, float]:
        """
        Compute reward and its components.
        
        Args:
            result: TaskResult from training on the current task
        
        Returns:
            dict with 'reward', 'acc_new', 'forgetting', 'cost' and component values
        """
        # --- Acc_new: accuracy on the new task ---
        acc_new = result.accuracy
        
        # --- Forgetting: mean accuracy degradation on previous tasks ---
        forgetting = 0.0
        num_prev = 0
        for prev_task, best_acc in self.best_accuracies.items():
            if prev_task != result.task_id:
                current_acc = result.all_accuracies.get(prev_task, 0.0)
                drop = max(0.0, best_acc - current_acc)
                forgetting += drop
                num_prev += 1
        
        if num_prev > 0:
            forgetting /= num_prev
        
        # --- Cost: normalized parameter usage ---
        self.total_params += result.param_cost
        cost = result.param_cost / max(self.budget_max, 1)
        
        # --- Update best accuracies ---
        for task_id, acc in result.all_accuracies.items():
            if task_id not in self.best_accuracies or acc > self.best_accuracies[task_id]:
                self.best_accuracies[task_id] = acc
        
        # --- Compute reward ---
        reward = (
            self.alpha * acc_new
            - self.beta * forgetting
            - self.gamma * cost
        )
        
        return {
            "reward": reward,
            "acc_new": acc_new,
            "forgetting": forgetting,
            "cost": cost,
            "total_params": self.total_params,
        }


class ContinualMetrics:
    """
    Computes standard continual learning metrics across a full sequence.
    
    Maintains the accuracy matrix A[i,j] = accuracy on task j after learning task i.
    """
    
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        self.acc_matrix = np.zeros((num_tasks, num_tasks))
    
    def update(self, current_task: int, all_accuracies: Dict[int, float]):
        """Record accuracies after learning current_task."""
        for task_id, acc in all_accuracies.items():
            if task_id < self.num_tasks:
                self.acc_matrix[current_task, task_id] = acc
    
    @property
    def average_accuracy(self) -> float:
        """Average accuracy after learning all tasks."""
        return self.acc_matrix[-1, :].mean()
    
    @property
    def forgetting(self) -> float:
        """Average forgetting: mean of max accuracy drop per task."""
        forgetting = 0.0
        for j in range(self.num_tasks - 1):  # exclude last task
            best = self.acc_matrix[:, j].max()
            final = self.acc_matrix[-1, j]
            forgetting += max(0, best - final)
        return forgetting / max(1, self.num_tasks - 1)
    
    @property
    def backward_transfer(self) -> float:
        """BWT: average influence of learning new tasks on old tasks."""
        bwt = 0.0
        count = 0
        for j in range(self.num_tasks - 1):
            bwt += self.acc_matrix[-1, j] - self.acc_matrix[j, j]
            count += 1
        return bwt / max(1, count)
    
    def summary(self) -> Dict[str, float]:
        return {
            "avg_accuracy": self.average_accuracy,
            "forgetting": self.forgetting,
            "bwt": self.backward_transfer,
        }

    def final_accuracies(self) -> Dict[int, float]:
        """Per-task accuracy after learning the full sequence (last row)."""
        return {j: float(self.acc_matrix[-1, j]) for j in range(self.num_tasks)}

    def print_final_row(self, indent: str = "    "):
        """
        Print per-task final accuracy — the key diagnostic for forgetting.

        If only the last task is high and earlier tasks are near-random,
        catastrophic forgetting is occurring. If all are reasonable,
        retention is working.
        """
        final = self.final_accuracies()
        print(f"{indent}Final accuracy per task (after full sequence):")
        cells = []
        for j in range(self.num_tasks):
            cells.append(f"T{j}={final[j]:.2f}")
        # Print in rows of 5 for readability
        for i in range(0, len(cells), 5):
            print(f"{indent}  " + "  ".join(cells[i:i+5]))
        print(f"{indent}  → mean={self.average_accuracy:.3f}  "
              f"forget={self.forgetting:.3f}  bwt={self.backward_transfer:.3f}")

