"""
Contextual Bandit Policy for RLDA.

Architecture: 2-layer MLP (256 → 128 → 9)
Training: REINFORCE with running-mean baseline + entropy bonus
Exploration: Softmax sampling during training, argmax at deployment

Key design decision: softmax sampling + entropy bonus (not epsilon-greedy)
for cleaner implementation and better gradient flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from dataclasses import dataclass, field


@dataclass
class BanditTransition:
    """A single (state, action, reward) tuple."""
    state: torch.Tensor
    action: int
    reward: float
    log_prob: float


class BanditPolicy(nn.Module):
    """
    Contextual bandit policy network.
    
    Input: state vector (state_dim,)
    Output: distribution over 9 profiles
    
    ~50K parameters for state_dim ≈ 210.
    """
    
    def __init__(
        self,
        state_dim: int,
        num_actions: int = 9,
        hidden_dims: List[int] = [256, 128],
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # Build MLP
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h_dim))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize final layer with small weights for uniform initial policy
        nn.init.normal_(self.network[-1].weight, std=0.01)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns logits over actions."""
        return self.network(state)
    
    def get_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        """
        Select an action given a state.
        
        Args:
            state: (state_dim,) tensor
            deterministic: if True, use argmax (deployment mode)
        
        Returns:
            action: int (profile index)
            log_prob: float (log probability of selected action)
        """
        logits = self.forward(state.unsqueeze(0))  # (1, num_actions)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=-1).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        
        log_prob = F.log_softmax(logits, dim=-1)[0, action].item()
        
        return action, log_prob
    
    def get_entropy(self, state: torch.Tensor) -> float:
        """Compute entropy of the policy at a given state."""
        logits = self.forward(state.unsqueeze(0))
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum().item()
        return entropy


class BanditTrainer:
    """
    Trains the bandit policy using REINFORCE with baseline + entropy bonus.
    
    Training loop:
    1. Collect transitions across task sequence
    2. Update policy using policy gradient
    3. Decay entropy coefficient
    """
    
    def __init__(
        self,
        policy: BanditPolicy,
        lr: float = 3e-4,
        entropy_coef: float = 0.1,
        entropy_decay: float = 0.995,
        entropy_min: float = 0.01,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.entropy_coef = entropy_coef
        self.entropy_decay = entropy_decay
        self.entropy_min = entropy_min
        
        # Running baseline for variance reduction
        self.baseline_sum = 0.0
        self.baseline_count = 0
        
        # Reward normalization
        self.reward_running_mean = 0.0
        self.reward_running_var = 1.0
        self.reward_count = 0
        
        # Buffer for current sequence
        self.transitions: List[BanditTransition] = []
    
    @property
    def baseline(self) -> float:
        if self.baseline_count == 0:
            return 0.0
        return self.baseline_sum / self.baseline_count
    
    def normalize_reward(self, reward: float) -> float:
        """Running normalization of reward for stability."""
        self.reward_count += 1
        alpha = 1.0 / self.reward_count
        self.reward_running_mean = (1 - alpha) * self.reward_running_mean + alpha * reward
        self.reward_running_var = (1 - alpha) * self.reward_running_var + alpha * (reward - self.reward_running_mean) ** 2
        std = max(self.reward_running_var ** 0.5, 1e-8)
        return (reward - self.reward_running_mean) / std
    
    def store_transition(self, state: torch.Tensor, action: int, reward: float, log_prob: float):
        """Store a transition from one task allocation."""
        norm_reward = self.normalize_reward(reward)
        self.transitions.append(BanditTransition(
            state=state.detach(),
            action=action,
            reward=norm_reward,
            log_prob=log_prob,
        ))
        
        # Update baseline
        self.baseline_sum += norm_reward
        self.baseline_count += 1
    
    def update(self) -> dict:
        """
        Update policy from collected transitions.
        
        Returns dict of training metrics.
        """
        if len(self.transitions) == 0:
            return {"policy_loss": 0.0, "entropy": 0.0}
        
        # Compute policy gradient loss
        policy_loss = 0.0
        total_entropy = 0.0
        
        for t in self.transitions:
            # Advantage = reward - baseline
            advantage = t.reward - self.baseline
            
            # Recompute log_prob (for gradient)
            logits = self.policy(t.state.unsqueeze(0).to(next(self.policy.parameters()).device))
            log_probs = F.log_softmax(logits, dim=-1)
            log_prob = log_probs[0, t.action]
            
            # Policy gradient: -log_prob * advantage (negative for gradient ascent)
            policy_loss -= log_prob * advantage
            
            # Entropy bonus
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum()
            total_entropy += entropy
        
        n = len(self.transitions)
        policy_loss = policy_loss / n
        entropy_loss = -self.entropy_coef * total_entropy / n  # negative to maximize entropy
        
        total_loss = policy_loss + entropy_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Clear buffer
        metrics = {
            "policy_loss": policy_loss.item(),
            "entropy": (total_entropy / n).item(),
            "entropy_coef": self.entropy_coef,
            "baseline": self.baseline,
            "num_transitions": n,
        }
        self.transitions = []
        
        # Decay entropy
        self.entropy_coef = max(
            self.entropy_min,
            self.entropy_coef * self.entropy_decay,
        )
        
        return metrics
    
    def clear_sequence(self):
        """Clear transitions buffer (between sequences, without updating)."""
        self.transitions = []
