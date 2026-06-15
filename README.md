# SRJ Training

GNN tagger for **Small Radius Jets (SRJ)** using the Lund-plane representation.
The primary model is **LundNet** — a 6-layer EdgeConv graph neural network that
learns to separate quark-initiated jets (parton truth labels 1–5) from
gluon-initiated jets (labels −1, 21) from Lund-plane splittings stored in ATLAS
AnalysisTree ROOT files.

An optional **Combiner** MLP can be trained on top of LundNet + ParT scores.

Designed to run **on a laptop, CPU-only, with a small QCD dataset**.

---

## Pipeline at a glance

```
ROOT (AnalysisTree, SRJ_* branches)
  │
  ├─ Step 1  Make_data_SRJ.py          ROOT → PyG graphs + metadata ROOT + mean/std JSON
  │
  ├─ Step 2  preprocess_SRJ_CPU.py     standardise node features, flatten pT/η weights
  │
  ├─ Step 3  weight_ONLY_TRAINS_SRJ.py train LundNet, save .pt checkpoints
  │
  ├─ Step 4  test_make_scores_SRJ.py   run inference, write scores into ROOT
  │
  ├─ Step 5  weight_ONLY_TRAINS_COMBINER_SRJ.py   (optional) train Combiner MLP
  │
  └─ Step 6  make_scores_combiner_SRJ.py           (optional) combined scoring
```

---

## Installation

### Option A — conda (recommended)

```bash
conda env create -f environment.yml   # creates env + runs pip install -e . automatically
conda activate srj-training
```

### Option B — pip

```bash
# 1. Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install torch-geometric and other deps
pip install torch_geometric
pip install -r requirements.txt

# 3. Register tools/ and plotting/ as importable packages
pip install -e .
```

> **torch-geometric note:** if pip install fails, follow the
> [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
> or use the conda route above.

### GPU (optional)

Replace `cpuonly` in `environment.yml` with `pytorch-cuda=12.1` (adjust to your driver),
or change the PyTorch pip index URL to the CUDA variant.

### QLundNet (optional)

Only needed when `choose_model: QLundNet`:

```bash
pip install -e ".[quantum]"
```

---

## Quick start

### 1. Configure paths

Every config file in `configs/` contains `/path/to/your/...` placeholders.
Replace them with your actual paths before running. The table below shows what
each step needs:

| Step | Config | Key fields to set |
|------|--------|-------------------|
| 1 | `config_make_data_SRJ.yaml` | `path_to_rootfiles`, `out_dir` |
| 2 | `config_preprocess_SRJ.yaml` | `data.train_graphs`, `data.test_graphs`, `data.out_dir`, `data.meanstd_json` |
| 3 | `config_ONLY_TRAIN_SRJ.yaml` | `data.path_to_trainfiles`, `data.path_to_save` |
| 4 | `config_make_scores_SRJ.yaml` | `data.paths_to_test_file_root`, `data.paths_to_test_file_graphs`, `data.path_to_outdir`, `test.models_to_run[].ckpt` |

> Any field can be overridden at the command line with `--override key=value` (dot-notation for nested keys):
> ```bash
> python Make_data_SRJ.py configs/config_make_data_SRJ.yaml --override out_dir=/tmp/out n_files=3
> ```

### 2. Concrete example (small QCD dataset)

The path names below show how each step's outputs feed into the next.

```bash
# Step 1 — build graphs
#   reads:  /data/qcd/*.root  (AnalysisTree)
#   writes: /out/data_qcd_part0_70.00percent/graphs_qcd_part0_70.00percent_ln_kT_cut_None_with_pt
#           /out/data_qcd_part0_70.00percent/data_qcd_part0_70.00percent_ln_kT_cut_None.root
#           /out/data_qcd_part0_70.00percent/slice_meanstd_part0.json
#           /out/data_qcd_part1_30.00percent/...  (test chunk)
python Make_data_SRJ.py configs/config_make_data_SRJ.yaml

# Step 2 — preprocess (standardise + flatten pT/η)
#   reads:  graphs from step 1 + slice_meanstd_part0.json
#   writes: /out/preprocessed/processed_SRJ_train.pt
#           /out/preprocessed/processed_SRJ_test.pt
#           /out/preprocessed/plots_before_after/  (diagnostic plots)
python preprocess_SRJ_CPU.py configs/config_preprocess_SRJ.yaml

# Step 3 — train LundNet
#   reads:  /out/preprocessed/processed_SRJ_train.pt
#   writes: /out/training/LundNet_dijet_ln_kT_cut_None_e001_0.XXXXX.pt  (per epoch)
#           /out/training/losses_LundNet_..._DDMM-HHMM.txt
python weight_ONLY_TRAINS_SRJ.py configs/config_ONLY_TRAIN_SRJ.yaml

# Step 4 — score
#   reads:  /out/preprocessed/processed_SRJ_test.pt + best checkpoint from step 3
#   writes: /out/scores/Final_Scores_test_kTNone.root  (with fjet_LundNet_SRJ_score branch)
python test_make_scores_SRJ.py configs/config_make_scores_SRJ.yaml
```

---

## Laptop defaults

All configs are pre-tuned for laptop use:

| Parameter | Value | Why |
|-----------|-------|-----|
| `batch_size` | 512 | fits in typical laptop RAM |
| `num_workers` | 0 | avoids multiprocessing issues |
| `event_fractions` | `{0.7: 1, 0.3: 1}` | one 70% train chunk + one 30% test chunk |
| `n_files` | `null` (= all) | set to a small int (e.g. `5`) to limit I/O |
| `gpu` | `null` | auto-detects; falls back to CPU if no GPU |

---

## Input data format

`Make_data_SRJ.py` reads ROOT files with an `AnalysisTree` tree.
Required branches:

| Branch | Description |
|--------|-------------|
| `SRJ_pt`, `SRJ_eta`, `SRJ_phi`, `SRJ_mass` | Jet kinematics |
| `SRJ_partonTruthLabel` | Truth label (signal quarks: 1–5; background gluons: −1, 21) |
| `SRJ_Nconst`, `SRJ_Nconst_Charged` | Constituent multiplicity |
| `SRJ_jetLundZ`, `SRJ_jetLundKt`, `SRJ_jetLundDeltaR` | Lund-plane node features |
| `SRJ_jetLundIDParent1`, `SRJ_jetLundIDParent2` | Parent-node indices (graph edges) |
| `mcEventWeight` | Per-event MC weight |
| `dsid` | Dataset ID |

Default kinematic selection (in `configs/config_signal_SRJ.yaml`):
pT ∈ [20, 160] GeV, |η| ∈ [3.2, 4.5], ≥ 3 Lund-plane splittings.

---

## Repository structure

```
srj-training/
├── Make_data_SRJ.py                      # Step 1: ROOT → PyG graphs
├── preprocess_SRJ_CPU.py                 # Step 2: standardise + flatten
├── weight_ONLY_TRAINS_SRJ.py             # Step 3: train LundNet
├── test_make_scores_SRJ.py               # Step 4: inference → ROOT
├── weight_ONLY_TRAINS_COMBINER_SRJ.py    # Step 5: train Combiner MLP (optional)
├── make_scores_combiner_SRJ.py           # Step 6: combined inference (optional)
│
├── configs/
│   ├── config_signal_SRJ.yaml            # truth-label & kinematic selection
│   ├── config_make_data_SRJ.yaml         # Step 1
│   ├── config_preprocess_SRJ.yaml        # Step 2
│   ├── config_ONLY_TRAIN_SRJ.yaml        # Step 3
│   ├── config_make_scores_SRJ.yaml       # Step 4
│   ├── config_ONLY_TRAIN_COMBINER_SRJ.yaml    # Step 5 (optional)
│   └── config_make_scores_combiner_SRJ.yaml   # Step 6 (optional)
│
├── tools/
│   ├── utils_config.py                   # YAML recursive-update + dot-arg CLI parser
│   └── GNN_model_weight/
│       ├── models.py                     # LundNet, GATNet, GINNet, Combiner, …
│       ├── utils_newdata.py              # graph builder, training loops, reweighting
│       └── quantum_layers.py             # PennyLane layers for QLundNet (optional)
│
├── plotting/
│   └── utils_plots_matplotlib.py         # histogram-with-error-bars helper
│
├── pyproject.toml                        # makes tools/ and plotting/ pip-installable
├── requirements.txt
└── environment.yml
```

---

## Available models

Set `choose_model` in `configs/config_ONLY_TRAIN_SRJ.yaml`:

| Value | Description |
|-------|-------------|
| `LundNet` | **Default.** 6-layer EdgeConv GNN with skip connections and Ntrk input |
| `GATNet` | Graph Attention Network |
| `GINNet` | Graph Isomorphism Network |
| `EdgeGinNet` | Hybrid EdgeConv + GIN |
| `PNANet` | Principal Neighbourhood Aggregation |
| `QLundNet` | Hybrid quantum-classical LundNet (requires `pip install -e ".[quantum]"`) |
