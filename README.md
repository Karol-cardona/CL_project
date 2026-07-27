# FedAvg vs FedCurv on Non-IID CIFAR-10

A from-scratch PyTorch
implementation of federated learning under controlled label-skew heterogeneity.

The project frames **client drift** in non-IID federated learning as a form of
**catastrophic forgetting**: each client's local SGD pulls the shared model toward its own
minimum, so the averaged model drifts away from the region that works for *all* clients.
We compare a **FedAvg** baseline against **FedCurv**, which adds an EWC-style
Fisher-weighted penalty during local training to anchor clients to the globally important
parameters.

Everything is implemented from scratch (no Flower / FedML / OpenFL), so every design
choice is explicit and inspectable.

---

## Repository structure

```
CL-Project/
├── main.py                     # CLI entry point for a single run
├── requirements.txt
├── data/
│   ├── cifar10.py              # dataset loading, transforms, test loader
│   ├── partition.py            # IID and Dirichlet (label-skew) partitioning
│   └── CIFAR10/                # dataset (downloaded on first run)
├── models/
│   └── resnet.py               # CIFAR ResNet (6n+2) with GroupNorm
├── algorithms/
│   ├── fedavg.py               # Client, FedAvgServer, cosine LR schedule
│   └── fedcurv.py              # FedCurvClient/Server, diagonal Fisher
├── utils/
│   ├── seed.py                 # reproducibility across all RNGs
│   ├── eval.py                 # accuracy, macro F1, fairness metrics
│   └── logger.py               # per-run folder, CSV, YAML config, summary
├── experiments/
│   ├── results/<run_name>/     # config.yaml, metrics.csv, summary.json
│   ├── plot_fedavg_results.py
│   └── plot_fedcurv_results.py
└── plots/                      # generated figures
```

---

## Setup

Requires an NVIDIA GPU (experiments were run on an RTX 5070 Laptop, ~8 GB VRAM).

```bash
conda create -n fedcl python=3.11
conda activate fedcl
pip install -r requirements.txt
```

**Notes on the environment**

- `torch` / `torchvision` are pinned to **nightly cu128 builds**, required for Blackwell
  (RTX 50-series) GPU support.
- Install with **pip only** (not mixing conda and pip) to avoid the OpenMP duplicate-library
  conflict. As a safety net, `data/cifar10.py` and `data/partition.py` set
  `KMP_DUPLICATE_LIB_OK=TRUE`.
- `requirements.txt` is currently missing two packages that the code imports:
  **`scikit-learn`** (used by `utils/eval.py` for precision/recall/F1) and **`pandas`**
  (used by `experiments/plot_fedavg_results.py`). Add them before installing on a clean
  machine.

CIFAR-10 downloads automatically on first run into `data/CIFAR10/`; subsequent runs skip
the (slow) MD5 verification.

---

## Usage

A single experiment is one invocation of `main.py`:

```bash
# FedAvg baseline, strong non-IID
python main.py --algorithm fedavg --alpha 0.1 --rounds 100 --local-epochs 2 \
               --seed 42 --eval-per-class

# Near-IID reference (alpha large)
python main.py --algorithm fedavg --alpha 100 --rounds 100 --seed 42 --eval-per-class

# True IID partition (sentinel: alpha <= 0)
python main.py --algorithm fedavg --alpha 0 --rounds 100 --seed 42

# FedCurv with Fisher penalty
python main.py --algorithm fedcurv --alpha 0.1 --rounds 100 --local-epochs 5 \
               --fed-lambda 1 --fisher-samples 200 --seed 42 --eval-per-class
```

### Main arguments

| Flag | Default | Description |
|---|---|---|
| `--algorithm` | `fedavg` | `fedavg` or `fedcurv` |
| `--alpha` | `0.1` | Dirichlet concentration; small = strongly non-IID, large → IID. `<= 0` selects the IID partition |
| `--num-clients` | `10` | Number of simulated clients (cross-silo) |
| `--rounds` | `100` | Communication rounds |
| `--local-epochs` | `2` | Local epochs per round (E) |
| `--model` | `resnet20` | `resnet8`, `resnet20`, `resnet32` |
| `--lr` / `--lr-min` | `0.01` / `1e-4` | Cosine schedule endpoints across rounds |
| `--lr-schedule` | `cosine` | `cosine` or `constant` (McMahan-style) |
| `--batch-size` | `64` | Local batch size |
| `--momentum` / `--weight-decay` | `0.9` / `1e-4` | Local SGD settings |
| `--seed` | `42` | Seeds partition, model init and training |
| `--eval-extended-every` | `10` | Rounds between macro-F1 / fairness evaluations (0 disables) |
| `--eval-per-class` | off | Full per-class report at the end |
| `--fed-lambda` | `1000.0` | FedCurv regularization coefficient λ |
| `--fisher-samples` | `200` | Samples per client for the Fisher estimate |
| `--use-wandb` | off | Mirror metrics to Weights & Biases |

### Output

Each run creates `experiments/results/<run_name>/` containing:

- `config.yaml` — every hyperparameter of the run (reproducibility)
- `metrics.csv` — one row per round; fairness columns are filled every K rounds and `NaN` otherwise
- `summary.json` — best/final accuracy, per-class metrics, wall-clock time

Run names are auto-generated as `{algorithm}_alpha{α}_E{E}_seed{s}[_lambda{λ}]`, e.g.
`fedcurv_alpha0.1_E5_seed42_lambda1_0`.

### Plots

```bash
python experiments/plot_fedavg_results.py    # reads metrics.csv / summary.json
python experiments/plot_fedcurv_results.py   # numbers hard-coded from run summaries
```

Figures are written to `plots/`. The FedAvg script loads results programmatically; the
FedCurv script embeds the final numbers directly, because the λ suffix in run names
(`lambda1_0` for λ=1 vs `lambda0_1` for λ=0.1) makes automatic lookup error-prone.

---

## Metrics

Beyond accuracy, the evaluation focuses on **class-level disparity**, which is the
observable signature of client drift:

- **Macro F1** — unweighted mean of per-class F1; penalizes models that abandon difficult classes.
- **std_acc** — standard deviation of per-class accuracies.
- **fairness_gap** — `max − min` per-class accuracy (in percentage points).
- **min_acc** — worst-class accuracy.

Two models with identical average accuracy can differ sharply here: a uniform
`[73, 73, …, 73]` and a skewed `[94, 91, 47, 51, …]` share the same mean but not the same
gap. Fairness metrics should always be read *jointly with accuracy*, since a uniformly poor
model trivially achieves a small gap.

Lightweight evaluation (loss + accuracy) runs every round; the expensive extended suite
runs every `--eval-extended-every` rounds and at the end.
