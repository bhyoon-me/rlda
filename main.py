"""
RLDA Main Entry Point — Full Experiment Pipeline

Usage:
    python main.py --config configs/split_cifar100_mvp.yaml --mode rlda
    python main.py --config configs/split_cifar100_mvp.yaml --mode baselines
    python main.py --config configs/split_cifar100_mvp.yaml --mode references
    python main.py --config configs/split_cifar100_mvp.yaml --mode all
    python main.py --config configs/split_cifar100_mvp.yaml --mode analysis
    python main.py --config configs/split_cifar100_mvp.yaml --mode table
"""

import argparse, yaml, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _get_orderings(config, num_orderings, base_seed):
    """Generate orderings for the configured dataset."""
    dataset_name = config["dataset"].get("name", "split_cifar100")
    if dataset_name == "split_tinyimagenet":
        from data.split_tinyimagenet import generate_orderings_tinyimagenet
        return generate_orderings_tinyimagenet(int(num_orderings), base_seed)
    else:
        from data.split_cifar100 import generate_orderings
        return generate_orderings(int(num_orderings), base_seed)


def load_config(path):
    with open(path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    # Resolve data_root to an absolute path so dataset loading is robust
    # regardless of the current working directory (fixes Windows './data' errors).
    dc = config.get("dataset", {})
    data_root = dc.get("data_root", "./data")
    if not os.path.isabs(data_root):
        data_root = os.path.join(repo_dir, data_root)
    dc["data_root"] = os.path.normpath(os.path.abspath(data_root))
    os.makedirs(dc["data_root"], exist_ok=True)
    config["dataset"] = dc
    # Resolve logging.save_dir to an absolute, normalized path. This prevents
    # mixed-slash paths (e.g. './results\\file.jsonl') that Windows rejects
    # with Errno 22, especially inside synced folders (Google Drive, OneDrive).
    lc = config.get("logging", {})
    save_dir = lc.get("save_dir", "./results")
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(repo_dir, save_dir)
    lc["save_dir"] = os.path.normpath(os.path.abspath(save_dir))
    os.makedirs(lc["save_dir"], exist_ok=True)
    config["logging"] = lc
    return config


def run_rlda(config):
    import torch
    from trainers.rlda_trainer import RLDATrainer
    save_dir = config["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    trainer = RLDATrainer(config)
    trainer.setup()

    print("\n" + "█" * 70 + "\n  RLDA META-TRAINING\n" + "█" * 70)
    train_results = trainer.meta_train()
    _save_summary(save_dir, "rlda_train_summary.json", "rlda_bandit", train_results)
    torch.save(trainer.policy.state_dict(), os.path.normpath(os.path.join(save_dir, "policy.pt")))

    print("\n" + "█" * 70 + "\n  RLDA META-EVAL (Transfer)\n" + "█" * 70)
    eval_results = trainer.meta_eval()
    _save_summary(save_dir, "rlda_eval_summary.json", "rlda_transfer", eval_results)


def run_baselines(config):
    from trainers.baselines import FixedProfileRunner, HeuristicRunner
    from analysis.logger import AllocationLogger
    from rl.profiles import PROFILE_NAMES

    save_dir = config["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    orderings = _get_orderings(config, config["meta"]["num_eval_orderings"], base_seed=10000)

    baselines = {}
    for pi in [2, 3, 4]:
        baselines[f"fixed_{PROFILE_NAMES[pi]}"] = FixedProfileRunner(config, pi)
    for h in ["uniform", "similarity_proportional", "gradient_proportional"]:
        baselines[f"heuristic_{h}"] = HeuristicRunner(config, h)
    # Standard CL objectives (shared adapter + regularizer, matched replay)
    from trainers.cl_baselines import EWCRunner, LwFRunner, NaiveSharedRunner
    baselines["cl_ewc"] = EWCRunner(config)
    baselines["cl_lwf"] = LwFRunner(config)
    baselines["cl_shared_naive"] = NaiveSharedRunner(config)

    # Optional subset selection: set RLDA_BASELINES to a comma-separated list
    # of method names to run only those (e.g. "heuristic_uniform,heuristic_
    # similarity_proportional"). Useful to finish partial runs without
    # repeating completed methods.
    subset = os.environ.get("RLDA_BASELINES", "").strip()
    if subset:
        wanted = [s.strip() for s in subset.split(",") if s.strip()]
        unknown = [w for w in wanted if w not in baselines]
        if unknown:
            print(f"  [warn] unknown baseline names ignored: {unknown}")
            print(f"  [info] valid names: {list(baselines.keys())}")
        baselines = {k: v for k, v in baselines.items() if k in wanted}
        print(f"  [subset] running only: {list(baselines.keys())}")

    # Merge with any existing summary so partial runs accumulate rather than
    # overwrite (e.g. fixed_* done earlier, heuristic_* added now).
    summary_path = os.path.normpath(os.path.join(save_dir, "baselines_summary.json"))
    all_summaries = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            all_summaries = json.load(f)
        print(f"  [merge] loaded existing summary with methods: {list(all_summaries.keys())}")

    for name, runner in baselines.items():
        print(f"\n{'█' * 70}\n  BASELINE: {name}\n{'█' * 70}")
        logger = AllocationLogger(save_dir, run_type=name)
        results = []
        for i, ordering in enumerate(orderings):
            print(f"  Ordering {i+1}/{len(orderings)}...")
            results.append(runner.run_sequence(ordering, i, 10000+i, logger))
        logger.save()
        all_summaries[name] = _make_summary(name, results)
        print(f"  → {name}: {all_summaries[name]['avg_accuracy_mean']:.3f} "
              f"± {all_summaries[name]['avg_accuracy_std']:.3f}")
        # Incremental save after EVERY method — a later crash loses nothing.
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)

    print(f"\n  [done] baselines_summary.json contains: {list(all_summaries.keys())}")


def run_references(config):
    from trainers.baselines import BestFixedRunner, OracleRunner
    from analysis.logger import AllocationLogger

    save_dir = config["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    orderings = _get_orderings(config, config["meta"]["num_eval_orderings"], base_seed=10000)
    all_summaries = {}

    # Allow skipping best_fixed if already done (set RLDA_SKIP_BESTFIXED=1).
    # best_fixed is expensive (9 profiles × orderings); if it already ran,
    # don't repeat it — go straight to the oracle.
    skip_best_fixed = os.environ.get("RLDA_SKIP_BESTFIXED", "0") == "1"

    if not skip_best_fixed:
        # Best Fixed
        print(f"\n{'█' * 70}\n  Best Fixed Profile (per-ordering)\n{'█' * 70}")
        bf = BestFixedRunner(config)
        bf_logger = AllocationLogger(save_dir, run_type="best_fixed")
        bf_results = [bf.run_sequence(orderings[i], i, 10000+i, bf_logger)
                      for i in range(len(orderings))]
        bf_logger.save()
        all_summaries["best_fixed"] = _make_summary("best_fixed", bf_results)
        # Save incrementally so a later crash doesn't lose best_fixed.
        with open(os.path.normpath(os.path.join(save_dir, "best_fixed_summary.json")), "w", encoding="utf-8") as f:
            json.dump(all_summaries["best_fixed"], f, indent=2)
    else:
        print("  [skip] best_fixed (RLDA_SKIP_BESTFIXED=1) — loading existing summary if present")
        bf_path = os.path.normpath(os.path.join(save_dir, "best_fixed_summary.json"))
        if os.path.exists(bf_path):
            with open(bf_path, encoding="utf-8") as f:
                all_summaries["best_fixed"] = json.load(f)

    # Oracle (run fewer — 9× expensive)
    print(f"\n{'█' * 70}\n  Retrospective Oracle\n{'█' * 70}")
    oracle = OracleRunner(config)
    n_oracle = min(5, len(orderings))
    oracle_results = [oracle.run_sequence(orderings[i], i, 10000+i)
                      for i in range(n_oracle)]
    all_summaries["oracle"] = _make_summary("oracle", oracle_results)
    # Save oracle incrementally too.
    with open(os.path.normpath(os.path.join(save_dir, "oracle_summary.json")), "w", encoding="utf-8") as f:
        json.dump(all_summaries["oracle"], f, indent=2)

    with open(os.path.normpath(os.path.join(save_dir, "references_summary.json")), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    if "best_fixed" in all_summaries:
        print(f"\n  Best Fixed: {all_summaries['best_fixed']['avg_accuracy_mean']:.3f}")
    print(f"  Oracle:     {all_summaries['oracle']['avg_accuracy_mean']:.3f}")


def run_analysis(config):
    from analysis.logger import AllocationLogger
    save_dir = config["logging"]["save_dir"]
    alloc_path = os.path.normpath(os.path.join(save_dir, "allocations.jsonl"))
    if not os.path.exists(alloc_path):
        print(f"No allocation log at {alloc_path}. Run experiments first.")
        return

    records = AllocationLogger.load(alloc_path)
    print(f"Loaded {len(records)} records")

    by_type = {}
    for r in records:
        by_type.setdefault(r["run_type"], []).append(r)
    for rt, recs in by_type.items():
        accs = [r["acc_new"] for r in recs]
        print(f"  {rt}: n={len(recs)}, mean_acc={np.mean(accs):.3f}")

    # Figure 3 data
    rlda_recs = [r for r in records if r["run_type"] in ("train", "eval") and r["task_idx"] > 0]
    if rlda_recs:
        sims = [r["max_similarity"] for r in rlda_recs]
        ranks = [r["selected_rank"] for r in rlda_recs]
        corr = np.corrcoef(sims, ranks)[0, 1] if len(sims) > 1 else 0
        print(f"\n[Figure 3] sim-vs-rank correlation: {corr:.3f} (expect negative)")

    print_table(config)


def print_table(config):
    save_dir = config["logging"]["save_dir"]
    print(f"\n{'═'*73}")
    print(f"{'Method':<35} {'AvgAcc':>8} {'±std':>7} {'Forget':>8} {'Params':>10}")
    print("-" * 73)
    for fname in sorted(os.listdir(save_dir)):
        if not fname.endswith("_summary.json"):
            continue
        with open(os.path.normpath(os.path.join(save_dir, fname)), encoding='utf-8') as f:
            data = json.load(f)
        if "method" in data:
            _print_row(data)
        else:
            for info in data.values():
                _print_row(info)


def _print_row(d):
    print(f"  {d.get('method','?'):<33} "
          f"{d.get('avg_accuracy_mean',0):>8.3f} "
          f"{d.get('avg_accuracy_std',0):>6.3f} "
          f"{d.get('forgetting_mean',0):>8.3f} "
          f"{d.get('total_params_mean',0):>10.0f}")


def _save_summary(save_dir, filename, method_name, results):
    accs = [r["metrics"]["avg_accuracy"] for r in results]
    forgets = [r["metrics"]["forgetting"] for r in results]
    summary = {
        "method": method_name,
        "num_orderings": len(results),
        "avg_accuracy_mean": float(np.mean(accs)),
        "avg_accuracy_std": float(np.std(accs)),
        "forgetting_mean": float(np.mean(forgets)),
        "forgetting_std": float(np.std(forgets)),
        "bwt_mean": float(np.mean([r["metrics"]["bwt"] for r in results])),
        "total_params_mean": float(np.mean([r["total_params"] for r in results])),
        "per_ordering": [r["metrics"] for r in results],
    }
    with open(os.path.normpath(os.path.join(save_dir, filename)), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved → {filename}: {summary['avg_accuracy_mean']:.3f} ± {summary['avg_accuracy_std']:.3f}")


def _make_summary(name, results):
    accs = [r["metrics"]["avg_accuracy"] for r in results]
    forgets = [r["metrics"]["forgetting"] for r in results]
    return {
        "method": name,
        "num_orderings": len(results),
        "avg_accuracy_mean": float(np.mean(accs)),
        "avg_accuracy_std": float(np.std(accs)),
        "forgetting_mean": float(np.mean(forgets)),
        "forgetting_std": float(np.std(forgets)),
        "total_params_mean": float(np.mean([r["total_params"] for r in results])),
        "per_ordering": [r["metrics"] for r in results],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLDA: Learning to Allocate")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["rlda", "baselines", "references", "all", "analysis", "table", "transfer", "rlda_shared"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    config["seed"] = args.seed

    if args.mode == "rlda":     run_rlda(config)
    elif args.mode == "baselines": run_baselines(config)
    elif args.mode == "references": run_references(config)
    elif args.mode == "all":
        run_rlda(config); run_baselines(config); run_references(config); print_table(config)
    elif args.mode == "analysis": run_analysis(config)
    elif args.mode == "table":    print_table(config)
    elif args.mode == "rlda_shared":
        from trainers.rlda_shared_trainer import run_rlda_shared
        run_rlda_shared(config)
    elif args.mode == "transfer":
        # Cross-dataset transfer: train on CIFAR-100, deploy on TinyImageNet
        import subprocess
        policy_path = os.path.normpath(os.path.join(config["logging"]["save_dir"], "policy.pt"))
        cmd = [sys.executable, "scripts/cross_dataset_transfer.py",
               "--config", args.config, "--policy_path", policy_path]
        subprocess.run(cmd)
