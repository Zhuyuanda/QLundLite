import gc
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from tools.GNN_model_weight.utils_newdata import load_yaml
from plotting.utils_plots_matplotlib import hist_with_errors


# ============================================================
#  Vectorised standardization — avoids Python per-graph loop
# ============================================================
def _standardize_graphs(dataset, mean_x, std_x, mean_ntrk, std_ntrk):
    t_mean_x   = torch.from_numpy(mean_x).float()
    t_std_x    = torch.from_numpy(std_x).float()
    t_mean_ntrk = torch.tensor(mean_ntrk).float()
    t_std_ntrk  = torch.tensor(std_ntrk).float()

    for g in dataset:
        g.x    = (g.x - t_mean_x) / t_std_x
        g.Ntrk = (g.Ntrk - t_mean_ntrk) / t_std_ntrk


# ============================================================
#  2-D histogram flattening — replaces KDE (O(N²) → O(N))
# ============================================================
def _hist2d_weights(etas, pts, n_eta=50, n_pt=100, eps=1e-9):
    """
    Assign per-jet weights so the 2-D (|η|, pT) distribution becomes flat.
    Uses a histogram lookup — O(N) time, no KDE overhead.
    """
    eta_edges = np.linspace(etas.min(), etas.max(), n_eta + 1)
    pt_edges  = np.linspace(pts.min(),  pts.max(),  n_pt  + 1)

    H, _, _ = np.histogram2d(etas, pts, bins=[eta_edges, pt_edges])
    # bin indices (clipped to [0, n-1])
    i_eta = np.clip(np.searchsorted(eta_edges[1:], etas), 0, n_eta - 1)
    i_pt  = np.clip(np.searchsorted(pt_edges[1:],  pts),  0, n_pt  - 1)
    density = H[i_eta, i_pt]
    weights = 1.0 / np.maximum(density, eps)
    weights /= weights.mean()
    return weights


# ============================================================
#  Process one dataset (train or test)
# ============================================================
def process_one_split(graph_paths, config,
                      mean_x, std_x, mean_ntrk, std_ntrk,
                      do_flatten, tag, out_dir):

    is_train = (tag == "train")

    # ------------------------------------
    # Load graph files
    # ------------------------------------
    if isinstance(graph_paths, str):
        graph_paths = [graph_paths]

    dataset = []
    for path in graph_paths:
        print(f"\nLoading {tag} graph file:", path)
        dataset += torch.load(path, weights_only=False)
    print(f"Total {tag} graphs loaded: {len(dataset):,}")

    # ------------------------------------
    # Jet selection (consistent with training)
    # ------------------------------------
    if config["cut_pt_eta_mass"]:
        sig_cfg = load_yaml(config["config_signal_path"])[config["signal"]]
        pt_min, pt_max = sig_cfg["pt_range"]
        m_min,  m_max  = sig_cfg["mass_range"]
        eta_min = sig_cfg.get("eta_min", 0)
        eta_max = sig_cfg.get("eta_max", 4.5)

        before = len(dataset)
        dataset = [
            g for g in dataset
            if pt_min < g.pt < pt_max
            and m_min < g.mass < m_max
            and eta_min <= abs(g.eta) <= eta_max
        ]
        print(f"{tag}: {len(dataset):,} remaining after cuts (removed {before - len(dataset):,})")

    # ------------------------------------
    # Standardization (vectorised per graph, unavoidable due to variable node count)
    # ------------------------------------
    print(f"\nStandardizing {tag} graphs...")
    _standardize_graphs(dataset, mean_x, std_x, mean_ntrk, std_ntrk)

    # ------------------------------------
    # Collect attributes (single pass, pre-allocated)
    # ------------------------------------
    n = len(dataset)
    labels = np.empty(n, dtype=np.int32)
    pts    = np.empty(n, dtype=np.float32)
    etas   = np.empty(n, dtype=np.float32)
    for i, g in enumerate(dataset):
        labels[i] = int(g.y)
        pts[i]    = float(g.pt)
        etas[i]   = float(g.eta)

    # ------------------------------------
    # Flatten (train only)
    # ------------------------------------
    if do_flatten:
        print(f"\nFlattening {tag} dataset (histogram 2-D, n_pt={config['n_bins_pt']}, n_eta={config['n_bins_eta']})...")

        sig_mask = (labels == 1)
        bkg_mask = (labels == 0)

        use_kde = config.get("use_kde", False)

        if use_kde:
            # Original KDE — correct but O(N²); only feasible for N < ~50k
            from tools.GNN_model_weight.utils_newdata import assign_2d_flat_weights_kde
            print("  WARNING: use_kde=True is O(N²) and slow for large datasets.")
            weights_sig = assign_2d_flat_weights_kde(etas[sig_mask], pts[sig_mask])
            weights_bkg = assign_2d_flat_weights_kde(etas[bkg_mask], pts[bkg_mask])
        else:
            # Histogram-based — O(N), recommended for laptop
            weights_sig = _hist2d_weights(
                etas[sig_mask], pts[sig_mask],
                n_eta=config["n_bins_eta"], n_pt=config["n_bins_pt"]
            )
            weights_bkg = _hist2d_weights(
                etas[bkg_mask], pts[bkg_mask],
                n_eta=config["n_bins_eta"], n_pt=config["n_bins_pt"]
            )

        sig_idx = bkg_idx = 0
        for g in dataset:
            if g.y == 1:
                g.weights = float(weights_sig[sig_idx]); sig_idx += 1
            else:
                g.weights = float(weights_bkg[bkg_idx]); bkg_idx += 1
    else:
        print(f"\nSkipping flattening for {tag} dataset.")
        weights_sig = np.ones(int((labels == 1).sum()))
        weights_bkg = np.ones(int((labels == 0).sum()))

    # ============================================================
    #  Visualization (TRAIN ONLY)
    # ============================================================
    if is_train:
        print("\n=== Saving pT/η distributions before and after flattening ===")
        plot_dir = os.path.join(config["data"]["out_dir"], "plots_before_after")
        os.makedirs(plot_dir, exist_ok=True)

        sig_mask = (labels == 1)
        bkg_mask = (labels == 0)
        pts_sig,  pts_bkg  = pts[sig_mask],  pts[bkg_mask]
        etas_sig, etas_bkg = etas[sig_mask], etas[bkg_mask]

        # pT before
        plt.figure()
        hist_with_errors(pts_bkg, bins=config["n_bins_pt"], density=True,
                         fmt=".", capsize=2, label="Background")
        hist_with_errors(pts_sig, bins=config["n_bins_pt"], density=True,
                         fmt=".", label="Signal")
        plt.xlabel("SRJ pT [GeV]"); plt.ylabel("Density"); plt.legend()
        plt.savefig(os.path.join(plot_dir, "pT_before.png")); plt.close()

        # η before
        plt.figure()
        hist_with_errors(etas_bkg, bins=config["n_bins_eta"], density=True,
                         fmt=".", capsize=2, label="Background")
        hist_with_errors(etas_sig, bins=config["n_bins_eta"], density=True,
                         fmt=".", label="Signal")
        plt.xlabel("SRJ η"); plt.ylabel("Density"); plt.legend()
        plt.savefig(os.path.join(plot_dir, "eta_before.png")); plt.close()

        # 2D before
        plt.figure()
        H, xe, ye = np.histogram2d(pts_sig, etas_sig,
                                   bins=(config["n_bins_pt"], config["n_bins_eta"]),
                                   density=True)
        cmin = H[H > 0].min()
        plt.hist2d(pts_sig, etas_sig,
                   bins=(config["n_bins_pt"], config["n_bins_eta"]),
                   density=True, cmin=cmin)
        plt.colorbar(label="Density")
        plt.xlabel("SRJ pT [GeV]"); plt.ylabel("SRJ η")
        plt.savefig(os.path.join(plot_dir, "2D_sig_before.png")); plt.close()

        # 2D after
        plt.figure()
        H2, _, _ = np.histogram2d(pts_sig, etas_sig,
                                  bins=(config["n_bins_pt"], config["n_bins_eta"]),
                                  weights=weights_sig, density=True)
        cmin2 = H2[H2 > 0].min()
        plt.hist2d(pts_sig, etas_sig,
                   bins=(config["n_bins_pt"], config["n_bins_eta"]),
                   weights=weights_sig, density=True, cmin=cmin2)
        plt.colorbar(label="Density")
        plt.xlabel("SRJ pT [GeV]"); plt.ylabel("SRJ η")
        plt.savefig(os.path.join(plot_dir, "2D_sig_after.png")); plt.close()

        print("✓ Plots saved to", plot_dir)

    # ============================================================
    #  Save processed dataset
    # ============================================================
    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f"processed_SRJ_{tag}.pt")
    torch.save(dataset, outfile)
    del dataset; gc.collect()
    print(f"\n✓ Saved {tag} dataset → {outfile}")
    print("-" * 60)


# ============================================================
#  Main
# ============================================================
def preprocess_SRJ_CPU(config):
    stats      = json.load(open(config["data"]["meanstd_json"]))
    mean_x     = np.array(stats["mean_x"],  dtype=np.float32)
    std_x      = np.array(stats["std_x"],   dtype=np.float32)
    mean_ntrk  = float(stats["mean_ntrk"])
    std_ntrk   = float(stats["std_ntrk"])
    out_dir    = config["data"]["out_dir"]

    process_one_split(
        graph_paths=config["data"]["train_graphs"],
        config=config,
        mean_x=mean_x, std_x=std_x, mean_ntrk=mean_ntrk, std_ntrk=std_ntrk,
        do_flatten=True, tag="train", out_dir=out_dir,
    )

    if "test_graphs" in config["data"]:
        process_one_split(
            graph_paths=config["data"]["test_graphs"],
            config=config,
            mean_x=mean_x, std_x=std_x, mean_ntrk=mean_ntrk, std_ntrk=std_ntrk,
            do_flatten=False, tag="test", out_dir=out_dir,
        )


if __name__ == "__main__":
    import argparse
    from tools.utils_config import recursive_update, parse_dot_args

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config YAML")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Overrides in form key.subkey=value")
    args = parser.parse_args()

    config = load_yaml(args.config)
    config = recursive_update(config, parse_dot_args(args.override))
    preprocess_SRJ_CPU(config)
