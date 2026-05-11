"""
Logging utilities for federated learning experiments.

ExperimentLogger does three things:
1. Creates a unique results folder for each run.
2. Logs per-round metrics to a CSV file.
3. Saves the run configuration (hyperparameters) to a YAML file.
4. Optionally mirrors logs to Weights & Biases (wandb) for interactive plots.

Usage:
    logger = ExperimentLogger(
        run_name="fedavg_alpha0.1_seed42",
        config={"algorithm": "fedavg", "alpha": 0.1, "seed": 42, ...},
        use_wandb=False,
    )
    for round in range(num_rounds):
        # ... do training ...
        logger.log({"round": round, "test_loss": ..., "test_acc": ...})
    logger.close()
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import json
from datetime import datetime
from typing import Any, Dict, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


class ExperimentLogger:
    """
    Logs metrics for one experiment run.

    Layout on disk:
        experiments/results/
            <run_name>/
                config.yaml      # full run configuration
                metrics.csv      # one row per round, columns are auto-detected
                summary.json     # final summary (best acc, final loss, etc.)
    """

    def __init__(
            self,
            run_name: str,
            config: Dict[str, Any],
            use_wandb: bool = True,
            wandb_project: str = "fedcl-project",
            results_dir: Optional[Path] = None,
    ):
        self.run_name = run_name
        self.config = config
        self.use_wandb = use_wandb

        # Setup the run folder
        results_dir = results_dir or RESULTS_DIR
        results_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = results_dir / run_name
        self.run_dir.mkdir(exist_ok=True)

        # Save config
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        # Open CSV file (header written lazily on first log call)
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_file = None
        self.csv_writer = None
        self._csv_columns: Optional[list] = None

        # Wandb (optional)
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb.init(
                    project=wandb_project,
                    name=run_name,
                    config=config,
                    reinit=True,
                )
            except ImportError:
                print("[Logger] WARNING: wandb not installed, falling back to CSV only")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

        print(f"[Logger] Run: {run_name}")
        print(f"[Logger] Results dir: {self.run_dir}")
        if self.use_wandb:
            print(f"[Logger] wandb logging enabled, project={wandb_project}")

    def log(self, metrics: Dict[str, Any]) -> None:
        """
        Log one row of metrics. The first call defines the CSV columns;
        subsequent calls must use the same keys.
        """
        # Initialize CSV header on first call
        if self.csv_writer is None:
            self._csv_columns = list(metrics.keys())
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self._csv_columns)
            self.csv_writer.writeheader()

        # Sanity check: same keys as first call
        if set(metrics.keys()) != set(self._csv_columns):
            missing = set(self._csv_columns) - set(metrics.keys())
            extra = set(metrics.keys()) - set(self._csv_columns)
            raise ValueError(
                f"Inconsistent log keys. Missing: {missing}. Extra: {extra}"
            )

        # Write to CSV (and flush so we don't lose data if the process crashes)
        self.csv_writer.writerow(metrics)
        self.csv_file.flush()

        # Mirror to wandb
        if self.use_wandb and self.wandb is not None:
            self.wandb.log(metrics)

    def save_summary(self, summary: Dict[str, Any]) -> None:
        """Save a final summary (best metrics, time elapsed, etc.) as JSON."""
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        if self.use_wandb and self.wandb is not None:
            self.wandb.summary.update(summary)

    def close(self) -> None:
        """Close file handles and finish the wandb run."""
        if self.csv_file is not None:
            self.csv_file.close()
        if self.use_wandb and self.wandb is not None:
            self.wandb.finish()


def make_run_name(
        algorithm: str,
        alpha: float,
        local_epochs: int,
        seed: int,
        suffix: Optional[str] = None,
) -> str:
    """
    Build a standardized run name from key hyperparameters.

    Example: fedavg_alpha0.1_E2_seed42  ->  experiments/results/fedavg_alpha0.1_E2_seed42/
    """
    name = f"{algorithm}_alpha{alpha}_E{local_epochs}_seed{seed}"
    if suffix:
        name += f"_{suffix}"
    return name


if __name__ == "__main__":
    # Sanity check: create a fake run, log a few rounds, save summary
    import time
    import random

    print("=" * 60)
    print("Sanity check: ExperimentLogger")
    print("=" * 60)

    config = {
        "algorithm": "fedavg",
        "dataset": "cifar10",
        "model": "resnet20",
        "num_clients": 10,
        "alpha": 0.1,
        "local_epochs": 2,
        "lr": 0.01,
        "seed": 42,
    }

    timestamp = datetime.now().strftime("%H%M%S")
    run_name = make_run_name("fedavg", 0.1, 2, 42, suffix=f"sanity_{timestamp}")
    logger = ExperimentLogger(run_name=run_name, config=config, use_wandb=False)

    # Simulate 5 rounds of fake metrics
    for r in range(1, 6):
        fake_test_loss = 2.5 - 0.3 * r + random.uniform(-0.05, 0.05)
        fake_test_acc = 0.10 + 0.07 * r + random.uniform(-0.01, 0.01)
        fake_train_loss = fake_test_loss - 0.1
        logger.log({
            "round": r,
            "train_loss": round(fake_train_loss, 4),
            "test_loss": round(fake_test_loss, 4),
            "test_acc": round(fake_test_acc, 4),
        })
        time.sleep(0.05)

    logger.save_summary({
        "best_test_acc": 0.42,
        "final_test_acc": 0.42,
        "final_test_loss": 1.10,
        "total_rounds": 5,
    })
    logger.close()

    print(f"\n✓ Sanity check completed.")
    print(f"  Check the run folder: {logger.run_dir}")
    print(f"  Files inside:")
    for f in logger.run_dir.iterdir():
        print(f"    - {f.name}")