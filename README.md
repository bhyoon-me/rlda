# RLDA: Learning to Allocate — Dynamic Subspace Policies for Continual Learning

## Structure

```
rlda/
├── main.py                          # Unified entry point (rlda/baselines/references/analysis)
├── configs/split_cifar100_mvp.yaml  # MVP experiment config
├── data/split_cifar100.py           # Dataset + orderings
├── models/peft/
│   ├── lora.py                      # LoRALinear (dynamic rank, copy-init)
│   └── injection.py                 # LoRAInjector (hook-based forward injection)
├── rl/
│   ├── profiles.py                  # 9 allocation profiles
│   ├── state_encoder.py             # 4-component state construction
│   └── bandit.py                    # Contextual bandit + REINFORCE + entropy
├── continual/metrics.py             # Reward + accuracy matrix
├── trainers/
│   ├── rlda_trainer.py              # Algorithm 1 (full RLDA pipeline)
│   └── baselines.py                 # Fixed / Heuristic / BestFixed / Oracle
├── analysis/logger.py               # Structured JSONL logging
└── scripts/verify_injection.py      # 6-test LoRA verification
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/verify_injection.py                                    # verify hooks
python main.py --config configs/split_cifar100_mvp.yaml --mode all    # run everything
python main.py --config configs/split_cifar100_mvp.yaml --mode table  # print results
```

## Implementation Status

| Component | Status |
|-----------|--------|
| LoRA injection (hooks) | Done |
| Bandit policy (REINFORCE + entropy) | Done |
| State encoder (4 components) | Done |
| 9 allocation profiles | Done |
| Inner training (L2 protection) | Done |
| RLDA trainer (Algorithm 1) | Done |
| FixedProfileRunner | Done |
| HeuristicRunner (3 heuristics) | Done |
| BestFixedRunner (per-ordering) | Done |
| OracleRunner (per-task, 9x) | Done |
| Allocation logger (JSONL) | Done |
| Replay buffer (ER, reservoir) | Done |
| Full Fisher EWC | TODO |
| Figure generation scripts | TODO |
| PPO extension | TODO |
