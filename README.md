# SRJ Training — Standalone Repo

Graph-neural-network (GNN) tagger for **Semi-central Rapidity Jets (SRJ)** using the
Lund-plane representation. The model is a **LundNet** (EdgeConv-based GNN) trained to
separate quark jets (truth labels 1–5) from gluon jets (truth labels −1, 21) using
`SRJ_*` branches from ATLAS AnalysisTree ROOT files.

An optional **Combiner** MLP can be trained on top of LundNet + ParT scores for
improved performance.

---

## Pipeline overview

```
ROOT files (AnalysisTree, SRJ branches)
        │
        ▼  Step 1 – Make data
        │  python Make_data_SRJ.py configs/config_make_data_SRJ.yaml
        │  → graphs_*.pt  (PyG graphs)
        │  → data_*.root  (jet-level metadata)
        │  → slice_meanstd_*.json  (Welford mean/std)
        │
        ▼  Step 2 – Preprocess
        │  python preprocess_SRJ_CPU.py configs/config_preprocess_SRJ.yaml
        │  → processed_SRJ_train.pt
        │  → processed_SRJ_test.pt
        │
        ▼  Step 3 – Train LundNet
        │  python weight_ONLY_TRAINS_SRJ.py configs/config_ONLY_TRAIN_SRJ.yaml
        │  → LundNet_*_eXXX_X.XXXXX.pt  (checkpoints)
        │  → losses_*.txt
        │
        ▼  Step 4 – Score
        │  python test_make_scores_SRJ.py configs/config_make_scores_SRJ.yaml
        │  → ROOT files with fjet_{tag}_score branch
        │
        ▼  Step 5 (optional) – Train Combiner MLP
        │  python weight_ONLY_TRAINS_COMBINER_SRJ.py configs/config_ONLY_TRAIN_COMBINER_SRJ.yaml
        │  → Combiner_*_eXXX_X.XXXXX.pt
        │
        ▼  Step 6 (optional) – Combined scoring
           python make_scores_combiner_SRJ.py configs/config_make_scores_combiner_SRJ.yaml
           → ROOT files with fjet_LundNet_score + fjet_Combined_score
```

---

## Installation

### Option A – pip (any Python environment)

```bash
pip install -r requirements.txt
```

For GPU support, install the CUDA-enabled PyTorch build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install -r requirements.txt
```

### Option B – conda

```bash
conda env create -f environment.yml
conda activate srj-training
```

### Optional: QLundNet (quantum layers)

Only needed if `choose_model: QLundNet` in the training config:

```bash
pip install pennylane
```

---

## Configuration

All paths in the `configs/` directory are **template placeholders** —
replace every `/path/to/your/...` with your actual paths before running.

| Config file | Step | Key paths to set |
|---|---|---|
| `config_make_data_SRJ.yaml` | Step 1 | `path_to_rootfiles`, `out_dir` |
| `config_preprocess_SRJ.yaml` | Step 2 | `data.train_graphs`, `data.test_graphs`, `data.out_dir`, `data.meanstd_json` |
| `config_ONLY_TRAIN_SRJ.yaml` | Step 3 | `data.path_to_trainfiles`, `data.path_to_save` |
| `config_make_scores_SRJ.yaml` | Step 4 | `data.paths_to_test_file_root`, `data.paths_to_test_file_graphs`, `data.path_to_outdir`, `test.models_to_run[].ckpt` |
| `config_ONLY_TRAIN_COMBINER_SRJ.yaml` | Step 5 | `data.root_dir`, `output.path_to_save` |
| `config_make_scores_combiner_SRJ.yaml` | Step 6 | all `ckpt` and path fields |

Any config value can be overridden at the command line using dot-notation:

```bash
python Make_data_SRJ.py configs/config_make_data_SRJ.yaml \
    --override out_dir=/my/output event_fraction_idx=3
```

---

## Running a full example

```bash
# 1 — Build graphs (runs all event fractions by default)
python Make_data_SRJ.py configs/config_make_data_SRJ.yaml

# 2 — Preprocess (standardize + flatten pT/η)
python preprocess_SRJ_CPU.py configs/config_preprocess_SRJ.yaml

# 3 — Train
python weight_ONLY_TRAINS_SRJ.py configs/config_ONLY_TRAIN_SRJ.yaml

# 4 — Score
python test_make_scores_SRJ.py configs/config_make_scores_SRJ.yaml

# 5 (optional) — Train combiner
python weight_ONLY_TRAINS_COMBINER_SRJ.py configs/config_ONLY_TRAIN_COMBINER_SRJ.yaml

# 6 (optional) — Combined scoring
python make_scores_combiner_SRJ.py configs/config_make_scores_combiner_SRJ.yaml
```

---

## Input data format

Step 1 (`Make_data_SRJ.py`) expects ROOT files containing an `AnalysisTree` tree with
the following branches:

| Branch | Description |
|---|---|
| `SRJ_pt`, `SRJ_eta`, `SRJ_phi`, `SRJ_mass` | Jet kinematics |
| `SRJ_partonTruthLabel` | Integer truth label (signal: 1–5; background: −1, 21) |
| `SRJ_Nconst`, `SRJ_Nconst_Charged` | Constituent multiplicity |
| `SRJ_jetLundZ`, `SRJ_jetLundKt`, `SRJ_jetLundDeltaR` | Lund-plane node features |
| `SRJ_jetLundIDParent1`, `SRJ_jetLundIDParent2` | Parent-node indices (graph edges) |
| `mcEventWeight` | Per-event MC weight |
| `dsid` | Dataset ID |

---

## Repository structure

```
srj-training/
├── Make_data_SRJ.py                  # Step 1: ROOT → PyG graphs
├── preprocess_SRJ_CPU.py             # Step 2: standardize + flatten
├── weight_ONLY_TRAINS_SRJ.py         # Step 3: train LundNet
├── test_make_scores_SRJ.py           # Step 4: inference → ROOT
├── weight_ONLY_TRAINS_COMBINER_SRJ.py  # Step 5: train Combiner MLP
├── make_scores_combiner_SRJ.py         # Step 6: combined inference
├── configs/
│   ├── config_signal_SRJ.yaml        # Truth-label & kinematic definitions
│   ├── config_make_data_SRJ.yaml     # Step 1 config
│   ├── config_preprocess_SRJ.yaml    # Step 2 config
│   ├── config_ONLY_TRAIN_SRJ.yaml    # Step 3 config
│   ├── config_make_scores_SRJ.yaml   # Step 4 config
│   ├── config_ONLY_TRAIN_COMBINER_SRJ.yaml   # Step 5 config
│   └── config_make_scores_combiner_SRJ.yaml  # Step 6 config
├── tools/
│   ├── utils_config.py               # YAML recursive-update & dot-arg parser
│   └── GNN_model_weight/
│       ├── models.py                 # LundNet, GATNet, GINNet, Combiner, …
│       ├── utils_newdata.py          # Graph creation, training loops, weighting
│       └── quantum_layers.py         # PennyLane quantum layers (QLundNet, optional)
├── plotting/
│   └── utils_plots_matplotlib.py     # Histogram-with-error-bars helper
├── requirements.txt
└── environment.yml
```

---

## Model options

Set `choose_model` in `config_ONLY_TRAIN_SRJ.yaml`:

| Value | Description |
|---|---|
| `LundNet` | Default: 6-layer EdgeConv GNN with skip connections + Ntrk |
| `GATNet` | Graph Attention Network |
| `GINNet` | Graph Isomorphism Network |
| `EdgeGinNet` | Hybrid EdgeConv + GIN |
| `PNANet` | Principal Neighbourhood Aggregation |
| `QLundNet` | Hybrid quantum-classical LundNet (requires PennyLane) |

---

## Notes

- Graphs produced by `Make_data_SRJ.py` are **not yet standardized**.
  The `preprocess_SRJ_CPU.py` step applies mean/std normalization from the
  `slice_meanstd_*.json` files produced in Step 1.
- pT/η flattening is applied only to the **training** set.
- The `event_fraction_idx` option in `config_make_data_SRJ.yaml` enables
  embarrassingly-parallel array-job submission (one job per fraction index).
- Model checkpoints are named `{model_name}_e{epoch:03d}_{val_loss:.5f}.pt`.
