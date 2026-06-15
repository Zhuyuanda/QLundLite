# SRJ Training — Quantum LundNet

A teaching and research repo for exploring **quantum graph neural networks** applied
to jet tagging in particle physics.

The central model is **QLundNet** — a hybrid quantum-classical GNN that replaces the
first EdgeConv layer of the classical LundNet with a PennyLane quantum circuit.
It is trained on **Small Radius Jets (SRJ)** represented as Lund-plane graphs to
separate quark-initiated jets (parton truth labels 1–5) from gluon-initiated jets
(labels −1, 21), using ATLAS AnalysisTree ROOT files.

The classical **LundNet** baseline is also included as a reference point.

Designed to run **on a laptop, CPU-only, with a small QCD dataset**.

---

## What is QLundNet?

```
Lund-plane graph (nodes = splittings, edges = parent links)
        │
        ▼  QuantumEdgeConv  (layer 1)   ← quantum circuit (PennyLane)
        │     encodes pairs of node features into a variational quantum circuit
        │     and reads out expectation values as new node embeddings
        │
        ▼  EdgeConv ×5                  ← classical layers 2–6
        │
        ▼  Global mean-pool
        │
        ▼  Ntrk concatenation + MLP → sigmoid score ∈ [0, 1]
```

The quantum circuit uses `n_qubits` qubits and `n_quantum_layers` variational
layers (default: 4 qubits, 2 layers).

> **Current status — work in progress.**
> Only the first EdgeConv layer has been replaced by a quantum circuit.
> Replacing more layers was found to make training prohibitively slow on
> classical simulators, so the hybrid single-layer design is used for now.
> A fully quantum architecture remains an open direction for future work.
> One natural starting point for student projects is to experiment with
> replacing different layers, or using faster PennyLane backends
> (e.g. `lightning.qubit`).

The classical **LundNet** uses standard EdgeConv in all six layers and serves
as the natural performance baseline.

---

## Pipeline

```
ROOT (AnalysisTree, SRJ_* branches)
  │
  ├─ Step 1  Make_data_SRJ.py          ROOT → PyG graphs + metadata ROOT + mean/std JSON
  │
  ├─ Step 2  preprocess_SRJ_CPU.py     standardise node features, flatten pT/η weights
  │
  ├─ Step 3  weight_ONLY_TRAINS_SRJ.py train QLundNet (or LundNet), save .pt checkpoints
  │
  ├─ Step 4  test_make_scores_SRJ.py   run inference, write scores into ROOT
  │
  ├─ Plot    notebooks/plot_SRJ_scores.ipynb       ROC, score distributions, rejection vs pT
  │
  ├─ Step 5  weight_ONLY_TRAINS_COMBINER_SRJ.py   (optional) train Combiner MLP
  │
  └─ Step 6  make_scores_combiner_SRJ.py           (optional) combined scoring
```

---

## Installation

### Option A — conda (recommended)

```bash
conda env create -f environment.yml   # creates env, installs all deps + pip install -e .
conda activate srj-training
```

### Option B — pip

```bash
# 1. CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. torch-geometric + all other deps (including pennylane)
pip install torch_geometric
pip install -r requirements.txt

# 3. Register tools/ and plotting/ as importable packages
pip install -e .
```

> **torch-geometric note:** if pip install fails, follow the
> [official PyG guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
> or use the conda route.

### GPU (optional)

Replace `cpuonly` in `environment.yml` with `pytorch-cuda=12.1` (adjust to your
driver), or change the pip index URL to the CUDA variant. PennyLane's default
simulator (`default.qubit`) runs on CPU; GPU acceleration for the quantum layers
requires a PennyLane plugin (e.g. `pennylane-lightning-gpu`).

---

## Quick start

### 1. Set your paths

Every config in `configs/` contains `/path/to/your/...` placeholders. Replace them
with real paths before running:

| Step | Config | Fields to set |
|------|--------|---------------|
| 1 | `config_make_data_SRJ.yaml` | `path_to_rootfiles`, `out_dir` |
| 2 | `config_preprocess_SRJ.yaml` | `data.train_graphs`, `data.test_graphs`, `data.out_dir`, `data.meanstd_json` |
| 3 | `config_ONLY_TRAIN_SRJ.yaml` | `data.path_to_trainfiles`, `data.path_to_save` |
| 4 | `config_make_scores_SRJ.yaml` | `data.paths_to_test_file_root`, `data.paths_to_test_file_graphs`, `data.path_to_outdir`, `test.models_to_run[].ckpt` |

Any field can be overridden at the command line:

```bash
python Make_data_SRJ.py configs/config_make_data_SRJ.yaml \
    --override out_dir=/tmp/out n_files=3
```

### 2. Run

```bash
# Step 1 — build Lund-plane graphs from ROOT
python Make_data_SRJ.py configs/config_make_data_SRJ.yaml

# Step 2 — standardise + flatten pT/η
python preprocess_SRJ_CPU.py configs/config_preprocess_SRJ.yaml

# Step 3 — train QLundNet (default) or LundNet
python weight_ONLY_TRAINS_SRJ.py configs/config_ONLY_TRAIN_SRJ.yaml

# Step 4 — write scores to ROOT
python test_make_scores_SRJ.py configs/config_make_scores_SRJ.yaml
```

### 3. Switch between models

In `configs/config_ONLY_TRAIN_SRJ.yaml`:

```yaml
choose_model: QLundNet   # quantum-classical hybrid  ← default
# choose_model: LundNet  # classical baseline
```

To change the quantum circuit size:
```yaml
# (add under architecture:)
n_qubits: 4         # number of qubits per circuit
n_quantum_layers: 2 # variational layers in the circuit
```

---

## Laptop defaults

| Parameter | Value | Reason |
|-----------|-------|--------|
| `batch_size` | 512 | fits in typical laptop RAM |
| `num_workers` | 0 | avoids multiprocessing issues |
| `event_fractions` | `{0.7: 1, 0.3: 1}` | 70% train + 30% test, single chunk |
| `n_files` | `null` | set to a small int (e.g. `5`) to speed up testing |
| `gpu` | `null` | auto-detects; falls back to CPU if no GPU found |

> **QLundNet training speed:** the quantum simulation layer is slower than classical
> EdgeConv. With `n_qubits=4` and `n_quantum_layers=2`, expect roughly 5–20× slower
> per-epoch time compared to LundNet on CPU. Reduce `n_epochs` (e.g. to `10`) and
> `batch_size` (e.g. to `128`) for exploratory runs.

---

## Input data format

`Make_data_SRJ.py` reads ROOT files with an `AnalysisTree` tree.

| Branch | Description |
|--------|-------------|
| `SRJ_pt`, `SRJ_eta`, `SRJ_phi`, `SRJ_mass` | Jet kinematics |
| `SRJ_partonTruthLabel` | Truth label (quarks 1–5 = signal; gluons −1, 21 = background) |
| `SRJ_Nconst`, `SRJ_Nconst_Charged` | Constituent multiplicity |
| `SRJ_jetLundZ`, `SRJ_jetLundKt`, `SRJ_jetLundDeltaR` | Lund-plane node features (z, kT, ΔR) |
| `SRJ_jetLundIDParent1`, `SRJ_jetLundIDParent2` | Parent-node indices (graph edges) |
| `mcEventWeight` | Per-event MC weight |
| `dsid` | Dataset ID |

Default kinematic selection (`configs/config_signal_SRJ.yaml`):
pT ∈ [20, 3200] GeV, η ∈ [2.0, 4.0], ≥ 3 Lund-plane splittings.

---

## Repository structure

```
srj-training/
├── Make_data_SRJ.py                      # Step 1: ROOT → PyG graphs
├── preprocess_SRJ_CPU.py                 # Step 2: standardise + flatten
├── weight_ONLY_TRAINS_SRJ.py             # Step 3: train QLundNet / LundNet
├── test_make_scores_SRJ.py               # Step 4: inference → ROOT
├── weight_ONLY_TRAINS_COMBINER_SRJ.py    # Step 5: Combiner MLP (optional)
├── make_scores_combiner_SRJ.py           # Step 6: combined scoring (optional)
│
├── configs/
│   ├── config_signal_SRJ.yaml            # truth-label & kinematic selection
│   ├── config_make_data_SRJ.yaml         # Step 1
│   ├── config_preprocess_SRJ.yaml        # Step 2
│   ├── config_ONLY_TRAIN_SRJ.yaml        # Step 3  (choose_model: QLundNet by default)
│   ├── config_make_scores_SRJ.yaml       # Step 4
│   ├── config_ONLY_TRAIN_COMBINER_SRJ.yaml    # Step 5 (optional)
│   └── config_make_scores_combiner_SRJ.yaml   # Step 6 (optional)
│
├── tools/
│   ├── utils_config.py                   # YAML recursive-update + dot-arg CLI parser
│   └── GNN_model_weight/
│       ├── models.py                     # QLundNet, LundNet, GATNet, GINNet, Combiner, …
│       ├── utils_newdata.py              # graph builder, training loops, reweighting
│       └── quantum_layers.py             # QuantumEdgeConv — PennyLane quantum circuit
│
├── notebooks/
│   └── plot_SRJ_scores.ipynb             # score distributions, ROC, rejection vs pT, correlations
│
├── plotting/
│   └── utils_plots_matplotlib.py         # histogram-with-error-bars helper
│
├── pyproject.toml                        # makes tools/ and plotting/ pip-installable
├── requirements.txt                      # pip dependencies (includes pennylane + jupyter)
└── environment.yml                       # conda environment (includes pennylane + jupyter)
```

---

## Model reference

| `choose_model` | Description |
|----------------|-------------|
| `QLundNet` | **Default.** Hybrid quantum-classical: quantum circuit for layer 1, classical EdgeConv for layers 2–6 |
| `LundNet` | Classical baseline: all 6 layers are EdgeConv |
| `GATNet` | Graph Attention Network |
| `GINNet` | Graph Isomorphism Network |
| `EdgeGinNet` | Hybrid EdgeConv + GIN |
| `PNANet` | Principal Neighbourhood Aggregation |

---

## Project ideas for students

- Compare QLundNet vs LundNet ROC curves and AUC as a function of training set size
- Vary `n_qubits` (2, 4, 6, 8) and `n_quantum_layers` (1, 2, 3) — how does expressivity change?
- Try different PennyLane ansätze in `tools/GNN_model_weight/quantum_layers.py`
- Replace only selected EdgeConv layers with quantum circuits and measure the trade-off
- Study convergence speed: does the quantum layer need more or fewer epochs?
- Adversarial mass-decorrelation with QLundNet (`do_combined_training: True`)
