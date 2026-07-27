"""
Main entry point for federated learning experiments.

Usage examples:

    # FedAvg, IID setting (alpha=100 ~ IID), 50 rounds
    python main.py --algorithm fedavg --alpha 100 --rounds 50 --seed 42

    # FedAvg, strong non-IID
    python main.py --algorithm fedavg --alpha 0.1 --rounds 100 --seed 42

    # Use ResNet-8 instead of ResNet-20 (faster for debugging)
    python main.py --algorithm fedavg --alpha 0.1 --rounds 30 --model resnet8

    # Skip data augmentation (faster, slightly worse accuracy)
    python main.py --algorithm fedavg --no-augment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import time
from typing import Optional

import torch

from data.cifar10 import get_cifar10_datasets, get_test_loader
from data.partition import iid_partition, dirichlet_partition
from models.resnet import build_resnet
from algorithms.fedavg import Client, FedAvgServer, cosine_lr
from algorithms.fedcurv import FedCurvClient, FedCurvServer
from utils.seed import set_seed
from utils.eval import evaluate, evaluate_extended
from utils.logger import ExperimentLogger, make_run_name

# Force UTF-8 stdout on Windows to handle Unicode characters in print statements
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description="Federated learning experiments on CIFAR-10")

    # Algorithm
    parser.add_argument("--algorithm", type=str, default="fedavg",
                        choices=["fedavg", "fedcurv"],
                        help="Federated algorithm to use")

    # Dataset / partitioning
    parser.add_argument("--num-clients", type=int, default=10,
                        help="Number of clients (cross-silo)")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Dirichlet concentration (small=non-IID, large=IID). "
                             "Use a special sentinel value <=0 to use IID partition instead.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable training data augmentation")

    # Model
    parser.add_argument("--model", type=str, default="resnet20",
                        choices=["resnet8", "resnet20", "resnet32"],
                        help="Model architecture")

    # Federated training schedule
    parser.add_argument("--rounds", type=int, default=100,
                        help="Number of communication rounds")
    parser.add_argument("--local-epochs", type=int, default=2,
                        help="Local training epochs per round")

    # Local optimizer
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Local SGD learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Local batch size")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (for partition + model init + training)")

    # Evaluation
    parser.add_argument("--eval-every", type=int, default=1,
                        help="Evaluate global model every N rounds (1 = every round)")
    parser.add_argument("--eval-per-class", action="store_true",
                        help="Also compute per-class accuracy at the end")
    parser.add_argument("--eval-extended-every", type=int, default=10,
                        help="Compute extended metrics (macro F1, fairness) every N rounds. "
                             "Set to 0 to disable. NOTE: must be a multiple of --eval-every.")

    # Logging
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override the auto-generated run name")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Optional suffix for run name (e.g. 'pilot')")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Mirror logs to Weights & Biases")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-client stats each round")

    # LR schedule
    parser.add_argument("--lr-schedule", type=str, default="cosine",
                        choices=["constant", "cosine"],
                        help="Learning rate schedule across global rounds. "
                             "'constant' keeps lr fixed (McMahan-style); "
                             "'cosine' decays lr from --lr to --lr-min over all rounds.")
    parser.add_argument("--lr-min", type=float, default=1e-4,
                        help="Minimum LR for cosine schedule (final round value)")

    # FedCurv-specific
    parser.add_argument("--fed-lambda", type=float, default=1000.0,
                        help="FedCurv regularization coefficient (ignored if algorithm != fedcurv)")
    parser.add_argument("--fisher-samples", type=int, default=200,
                        help="Number of samples used to estimate Fisher per client (FedCurv)")

    return parser.parse_args()


def model_depth_from_arg(model_arg: str) -> int:
    return {"resnet8": 8, "resnet20": 20, "resnet32": 32}[model_arg]


def main():
    args = parse_args()

    # Reproducibility
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Include lambda in suffix for FedCurv runs
    suffix = args.suffix
    if args.algorithm == "fedcurv":
        lambda_str = f"lambda{args.fed_lambda}".replace(".", "_")
        suffix = lambda_str if not suffix else f"{lambda_str}_{suffix}"

    run_name = args.run_name or make_run_name(
        algorithm=args.algorithm,
        alpha=args.alpha,
        local_epochs=args.local_epochs,
        seed=args.seed,
        suffix=suffix,
    )

    config = vars(args)
    config["device"] = str(device)
    logger = ExperimentLogger(
        run_name=run_name,
        config=config,
        use_wandb=args.use_wandb,
    )

    # Load datasets
    print("\nLoading CIFAR-10...")
    trainset, testset = get_cifar10_datasets(augment_train=not args.no_augment)
    test_loader = get_test_loader(testset, batch_size=256)
    print(f"  Train: {len(trainset)} samples, Test: {len(testset)} samples")

    # Partition train set across clients
    print(f"\nPartitioning train set across {args.num_clients} clients...")
    if args.alpha <= 0:
        client_indices = iid_partition(trainset, args.num_clients, seed=args.seed)
        print(f"  Strategy: IID (random uniform)")
    else:
        client_indices = dirichlet_partition(trainset, args.num_clients,
                                             alpha=args.alpha, seed=args.seed)
        print(f"  Strategy: Dirichlet(alpha={args.alpha})")
    sizes = [len(idx) for idx in client_indices]
    print(f"  Client sizes: min={min(sizes)}, max={max(sizes)}, "
          f"mean={sum(sizes)/len(sizes):.0f}")

    # Build clients (algorithm-specific)
    ClientClass = FedCurvClient if args.algorithm == "fedcurv" else Client
    clients = [
        ClientClass(
            client_id=k,
            dataset=trainset,
            indices=idx,
            batch_size=args.batch_size,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            device=device,
        )
        for k, idx in enumerate(client_indices)
    ]

    # Build global model and server
    depth = model_depth_from_arg(args.model)
    print(f"\nBuilding model: {args.model} (depth={depth}) on {device}")
    global_model = build_resnet(depth=depth, device=device)
    n_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,} (~{n_params/1e3:.1f} k)")

    if args.algorithm == "fedavg":
        server = FedAvgServer(global_model=global_model, clients=clients, device=device)
    elif args.algorithm == "fedcurv":
        server = FedCurvServer(
            global_model=global_model, clients=clients, device=device,
            fed_lambda=args.fed_lambda,
            fisher_samples=args.fisher_samples,
        )
        print(f"  FedCurv: lambda={args.fed_lambda}, fisher_samples={args.fisher_samples}")
    else:
        raise NotImplementedError(f"Algorithm {args.algorithm} not implemented yet")

    # Helper to build fresh local models on the right device
    def model_factory():
        return build_resnet(depth=depth, device=device)

    # Initial evaluation (round 0)
    print("\n" + "=" * 60)
    print("Starting federated training")
    print("=" * 60)

    if args.lr_schedule == "cosine":
        print(f"LR schedule: cosine from {args.lr} → {args.lr_min} over {args.rounds} rounds")
        print(f"  Sample: round 1 → lr={cosine_lr(1, args.rounds, args.lr, args.lr_min):.5f}")
        print(f"  Sample: round {args.rounds//2} → lr={cosine_lr(args.rounds//2, args.rounds, args.lr, args.lr_min):.5f}")
        print(f"  Sample: round {args.rounds} → lr={cosine_lr(args.rounds, args.rounds, args.lr, args.lr_min):.5f}")
    else:
        print(f"LR schedule: constant at {args.lr}")

    init_loss, init_acc = evaluate(global_model, test_loader, device)
    print(f"[Round 0] Initial: loss={init_loss:.4f}, acc={init_acc*100:.2f}%")

    # Build the initial log entry.
    init_log = {
        "round": 0,
        "train_loss": float("nan"),
        "test_loss": init_loss,
        "test_acc": init_acc,
        "elapsed_sec": 0.0,
        "lr": float("nan"),

        # Extended metrics (populated every --eval-extended-every rounds)
        "macro_f1": float("nan"),
        "std_acc": float("nan"),
        "fairness_gap": float("nan"),
        "min_acc": float("nan"),
    }
    if args.algorithm == "fedcurv":
        init_log["penalty_norm"] = 0.0
    logger.log(init_log)

    # Training loop
    best_acc = init_acc
    start_time = time.time()
    for r in range(1, args.rounds + 1):
        round_start = time.time()

        # Compute LR for this round according to the schedule
        if args.lr_schedule == "cosine":
            current_lr = cosine_lr(r, args.rounds, lr_max=args.lr, lr_min=args.lr_min)
        else:  # constant
            current_lr = args.lr

        stats = server.run_round(
            round_idx=r,
            model_factory=model_factory,
            local_epochs=args.local_epochs,
            verbose=args.verbose,
            lr_override=current_lr,
        )
        round_time = time.time() - round_start

        # Evaluate periodically
        if r % args.eval_every == 0 or r == args.rounds:
            test_loss, test_acc = evaluate(global_model, test_loader, device)
            elapsed = time.time() - start_time
            best_acc = max(best_acc, test_acc)
            base_msg = (f"[Round {r:3d}/{args.rounds}] "
                        f"train_loss={stats['weighted_client_loss']:.4f} "
                        f"test_loss={test_loss:.4f} "
                        f"test_acc={test_acc*100:5.2f}% "
                        f"(best={best_acc*100:5.2f}%) "
                        f"[{round_time:.1f}s/round, total {elapsed/60:.1f}min]")
            if "penalty_norm" in stats and stats["penalty_norm"] > 0:
                base_msg += f" | penalty={stats['penalty_norm']:.4f}"
            print(base_msg)
            log_entry = {
                "round": r,
                "train_loss": stats["weighted_client_loss"],
                "test_loss": test_loss,
                "test_acc": test_acc,
                "elapsed_sec": round(elapsed, 1),
                "lr": stats.get("lr", float("nan")),
            }

            # Extended evaluation: macro F1 + fairness, only every K rounds (expensive: full pass on test set + sklearn)
            do_extended = (
                    args.eval_extended_every > 0
                    and (r % args.eval_extended_every == 0 or r == args.rounds)
            )
            if do_extended:
                ext = evaluate_extended(global_model, test_loader, device, num_classes=10)
                log_entry["macro_f1"]     = ext["macro_f1"]
                log_entry["std_acc"]      = ext["std_acc"]
                log_entry["fairness_gap"] = ext["fairness_gap"]
                log_entry["min_acc"]      = ext["min_acc"]
                # Pretty print for the console too
                print(f"           ↳ macro_f1={ext['macro_f1']*100:5.2f}% "
                      f"std={ext['std_acc']*100:4.2f}% "
                      f"gap={ext['fairness_gap']*100:5.2f}pp "
                      f"worst_class={ext['min_acc']*100:5.2f}%")

            # FedCurv-specific: log the penalty term magnitude
            if "penalty_norm" in stats:
                log_entry["penalty_norm"] = stats["penalty_norm"]

            logger.log(log_entry)

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.1f} min")
    print(f"Best test accuracy: {best_acc*100:.2f}%")

    # Final per-class evaluation (optional)
    summary = {
        "final_test_acc": float(test_acc),
        "best_test_acc": float(best_acc),
        "final_test_loss": float(test_loss),
        "total_rounds": int(args.rounds),
        "total_time_min": round(total_time / 60, 2),
    }
    if args.eval_per_class:
        # Full extended evaluation: accuracy + macro F1 + per-class P/R/F1 + fairness
        final_metrics = evaluate_extended(global_model, test_loader, device, num_classes=10)

        # Add all extended metrics to summary
        for k, v in final_metrics.items():
            if k not in ("loss", "accuracy"):  # accuracy already in summary
                summary[f"final_{k}"] = v

        print("\n" + "=" * 60)
        print("Final evaluation (extended)")
        print("=" * 60)
        print(f"  Accuracy:      {final_metrics['accuracy']*100:6.2f}%")
        print(f"  Macro F1:      {final_metrics['macro_f1']*100:6.2f}%")
        print(f"\nFairness across classes:")
        print(f"  Std per-class: {final_metrics['std_acc']*100:6.2f}%")
        print(f"  Worst class:   {final_metrics['min_acc']*100:6.2f}%")
        print(f"  Best class:    {final_metrics['max_acc']*100:6.2f}%")
        print(f"  Fairness gap:  {final_metrics['fairness_gap']*100:6.2f}%")

        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        print(f"\nPer-class breakdown:")
        print(f"  {'Class':<12s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s}")
        for c, name in enumerate(class_names):
            print(f"  {name:<12s} "
                  f"{final_metrics[f'acc_class_{c}']*100:6.2f}% "
                  f"{final_metrics[f'precision_class_{c}']*100:6.2f}% "
                  f"{final_metrics[f'recall_class_{c}']*100:6.2f}% "
                  f"{final_metrics[f'f1_class_{c}']*100:6.2f}%")

    logger.save_summary(summary)
    logger.close()


if __name__ == "__main__":
    main()