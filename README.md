# SRJ Training

Graph-neural-network (GNN) tagger for **Small Radius Jets (SRJ)** using the Lund-plane
representation. The network is a **LundNet** (EdgeConv-based GNN) trained to separate
quark-initiated jets (parton truth labels 1–5) from gluon-initiated jets (labels −1, 21)
using Lund-plane splittings stored in ATLAS AnalysisTree ROOT files.

An optional **Combiner** MLP can be trained on top of LundNet + ParT scores.

The repo is designed to run **entirely on a laptop** (CPU-only, small datasets).

---

## Pipeline overview

```
ROOT files (AnalysisTree, SRJ_* branches)
        │
        ▼  Step 1 – Make data
        │  python Make_data_SRJ.py configs/config_make_data_SRJ.yaml
        │  → graphs_*.pt          (PyTorch Geometric graph objects)
        │  → data_*.root          (jet-level metadata: pT, η, labels, …)
        │  → slice_meanstd_*.json (Welford mean/std for normalisation)
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
        │  → ROOT files with fjet_{tag}_score branch added
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

### Step 1 — Create the environment

**Option A: conda (recommended — handles PyTorch + torch-geometric cleanly)**

```bash
conda env create -f environment.yml
conda activate srj-training
```

**Option B: pip**

```bash
# CPU-only PyTorch (laptop default)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install -r requirements.txt
```

> **Note:** `torch-geometric` can be tricky to install via pip alone.
> If you hit errors, try the conda route or the
> [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Step 2 — Install the package (required)

After activating the environment, install the repo itself so that the
`tools` and `plotting` modules can be imported from anywhere:

```bash
cd srj-training
pip install -e .
```

This registers `tools` and `plotting` as packages. Without this step the
scripts can only be run from the repo root directory.

### Optional: QLundNet (quantum layers)

Only needed if you set `choose_model: QLundNet` in the training config:

```bash
pip install pennylane
# or: pip install -e ".[quantum]"
```

---

## Quick start

### 1 — Edit the configs

Open each config file in `configs/` and replace every `/path/to/your/...` placeholder
with your actual paths. The table below shows the minimum set you must edit:

| Config | Step | Fields to fill in |
|---|---|---|
| `config_make_data_SRJ.yaml` | 1 | `path_to_rootfiles`, `out_dir` |
| `config_preprocess_SRJ.yaml` | 2 | `data.train_graphs`, `data.test_graphs`, `data.out_dir`, `data.meanstd_json` |
| `config_ONLY_TRAIN_SRJ.yaml` | 3 | `data.path_to_trainfiles`, `data.path_to_save` |
| `config_make_scores_SRJ.yaml` | 4 | paths + `test.models_to_run[].ckpt` |

Any config value can also be overridden at the command line:

```bash
python Make_data_SRJ.py configs/config_make_data_SRJ.yaml \
    --override out_dir=/my/output n_files=5
```

### 2 — Run the pipeline

```bash
# Step 1 — Build PyG graphs from ROOT
python Make_data_SRJ.py configs/config_make_data_SRJ.yaml

# Step 2 — Standardise node features and flatten pT/η
python preprocess_SRJ_CPU.py configs/config_preprocess_SRJ.yaml

# Step 3 — Train LundNet
python weight_ONLY_TRAINS_SRJ.py configs/config_ONLY_TRAIN_SRJ.yaml

# Step 4 — Write scores back to ROOT
python test_make_scores_SRJ.py configs/config_make_scores_SRJ.yaml
```

---

## Running on a laptop with a small dataset

The default configs are already tuned for laptop use:

- `batch_size: 512` (small enough to fit comfortably in RAM)
- `num_workers: 0` (avoids multiprocessing issues on some systems)
- `event_fractions: {1.0: 1}` — uses 100% of each file in a single chunk
  (set `n_files: 5` or similar to limit how many ROOT files are loaded)

If you have only a few ROOT files of QCD jets, a minimal `config_make_data_SRJ.yaml` looks like:

```yaml
path_to_rootfiles:
  - "/path/to/your/qcd_sample/*.root"

n_files: 5            # load only 5 files for a quick test
event_fractions: {0.7: 1, 0.3: 1}   # 70% train, 30% test
event_fraction_idx: null

out_dir: "/path/to/output/data_{id}{frac}"
out_file_name_graphs: "graphs_{id}{frac}_ln_kT_cut_{kT_cut}{include_pt}"
out_file_name_root: "data_{id}{frac}_ln_kT_cut_{kT_cut}.root"
id: "qcd"

signal_config_file: "configs/config_signal_SRJ.yaml"
signal: srj
signal_name_in_weight: False
kT_cut: null
include_pt: True
```

---

## Input data format

`Make_data_SRJ.py` reads ROOT files with an `AnalysisTree` tree containing
these branches:

| Branch | Description |
|---|---|
| `SRJ_pt`, `SRJ_eta`, `SRJ_phi`, `SRJ_mass` | Jet kinematics |
| `SRJ_partonTruthLabel` | Parton truth label (quark signal: 1–5; gluon background: −1, 21) |
| `SRJ_Nconst`, `SRJ_Nconst_Charged` | Constituent multiplicity |
| `SRJ_jetLundZ`, `SRJ_jetLundKt`, `SRJ_jetLundDeltaR` | Lund-plane node features (z, kT, ΔR per splitting) |
| `SRJ_jetLundIDParent1`, `SRJ_jetLundIDParent2` | Parent-node indices (graph edges) |
| `mcEventWeight` | Per-event MC weight |
| `dsid` | Dataset ID |

Jets are selected in the kinematic window defined in `configs/config_signal_SRJ.yaml`
(default: pT ∈ [20, 160] GeV, |η| ∈ [3.2, 4.5], min 3 Lund-plane splittings).

---

## Repository structure

```
srj-training/
├── Make_data_SRJ.py                     # Step 1: ROOT → PyG graphs
├── preprocess_SRJ_CPU.py                # Step 2: standardise + flatten
├── weight_ONLY_TRAINS_SRJ.py            # Step 3: train LundNet
├── test_make_scores_SRJ.py              # Step 4: inference → ROOT
├── weight_ONLY_TRAINS_COMBINER_SRJ.py   # Step 5: train Combiner MLP (optional)
├── make_scores_combiner_SRJ.py          # Step 6: combined inference (optional)
├── configs/
│   ├── config_signal_SRJ.yaml           # Truth-label & kinematic selection
│   ├── config_make_data_SRJ.yaml        # Step 1 config
│   ├── config_preprocess_SRJ.yaml       # Step 2 config
│   ├── config_ONLY_TRAIN_SRJ.yaml       # Step 3 config
│   ├── config_make_scores_SRJ.yaml      # Step 4 config
│   ├── config_ONLY_TRAIN_COMBINER_SRJ.yaml   # Step 5 config (optional)
│   └── config_make_scores_combiner_SRJ.yaml  # Step 6 config (optional)
├── tools/
│   ├── utils_config.py                  # YAML recursive-update & dot-arg parser
│   └── GNN_model_weight/
│       ├── models.py                    # LundNet, GATNet, GINNet, Combiner, …
│       ├── utils_newdata.py             # Graph builder, training loops, reweighting
│       └── quantum_layers.py            # PennyLane quantum layers (QLundNet, optional)
├── plotting/
│   └── utils_plots_matplotlib.py        # Histogram-with-error-bars helper
├── requirements.txt
└── environment.yml
```

---

## Available models

Set `choose_model` in `config_ONLY_TRAIN_SRJ.yaml`:

| Value | Description |
|---|---|
| `LundNet` | Default: 6-layer EdgeConv GNN with skip connections + Ntrk input |
| `GATNet` | Graph Attention Network |
| `GINNet` | Graph Isomorphism Network |
| `EdgeGinNet` | Hybrid EdgeConv + GIN |
| `PNANet` | Principal Neighbourhood Aggregation |
| `QLundNet` | Hybrid quantum-classical LundNet (requires PennyLane) |
