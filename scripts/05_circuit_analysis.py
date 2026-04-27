import itertools
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "results", "checkpoints")
METRICS_DIR = os.path.join(ROOT, "results", "metrics")
PLOTS_DIR = os.path.join(ROOT, "results", "plots")

for d in [METRICS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Architecture & constants ──────────────────────────────────────────────────
ARCH = dict(
    n_layers=1, d_model=128, n_heads=4, d_head=32,
    d_mlp=None, act_fn=None, normalization_type=None,
    d_vocab=115, d_vocab_out=115, n_ctx=3, attn_only=True,
)
N_HEADS = 4
N_POS = 3
TOP_SV = 10
TOP_SUBSPACE = 5
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ADDITION_GROKKED_SEEDS = [0, 1, 2, 3, 4, 6, 7, 9, 10]
SUBTRACTION_GROKKED_SEED = 5
SUBTRACTION_NEAR_MISS_SEEDS = [2, 4]

POS_LABELS_FROM = ["from_a", "from_=", "from_b"]
POS_LABELS_TO = ["to_a", "to_=", "to_b"]


# ── Loading ───────────────────────────────────────────────────────────────────
def build_model(seed: int) -> HookedTransformer:
    cfg = HookedTransformerConfig(**ARCH, seed=seed)
    return HookedTransformer(cfg)


def load_model(task: str, seed: int) -> HookedTransformer:
    ckpt = torch.load(
        os.path.join(CKPT_DIR, f"{task}_seed{seed}_final.pt"),
        weights_only=False,
    )
    model = build_model(seed)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model


def load_test_loader(task: str) -> DataLoader:
    d = torch.load(os.path.join(DATA_DIR, f"{task}_test.pt"), weights_only=True)
    a, b, label = d["a"], d["b"], d["label"]
    sep = torch.full_like(a, 113)
    inputs = torch.stack([a, sep, b], dim=1)
    return DataLoader(TensorDataset(inputs, label), batch_size=BATCH_SIZE, shuffle=False)


# ── Circuit decomposition ─────────────────────────────────────────────────────
@torch.no_grad()
def head_circuits(model: HookedTransformer) -> dict:
    """Return per-head OV and QK matrices, both shape (d_model, d_model)."""
    out = {}
    for h in range(N_HEADS):
        W_Q = model.W_Q[0, h]   # (d_model, d_head)
        W_K = model.W_K[0, h]   # (d_model, d_head)
        W_V = model.W_V[0, h]   # (d_model, d_head)
        W_O = model.W_O[0, h]   # (d_head, d_model)
        out[h] = {
            "OV": (W_V @ W_O).detach().cpu(),       # (d_model, d_model)
            "QK": (W_Q @ W_K.T).detach().cpu(),     # (d_model, d_model)
        }
    return out


def svd_summary(M: torch.Tensor) -> dict:
    """SVD-based summary: top-10 singular values, effective rank, frobenius norm."""
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    s_np = S.numpy()
    eff_rank = float((s_np.sum() ** 2) / (np.square(s_np).sum() + 1e-12))
    frob = float(torch.linalg.norm(M, ord="fro").item())
    return {
        "singular_values_top10": s_np[:TOP_SV].tolist(),
        "effective_rank": eff_rank,
        "frobenius_norm": frob,
        "U_top5": U[:, :TOP_SUBSPACE].clone(),  # kept in dict for alignment, not serialised
    }


def subspace_alignment(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """Frobenius norm of U1.T @ U2 normalised by sqrt(k). Returns [0,1]."""
    k = U1.size(1)
    M = U1.T @ U2
    return float(torch.linalg.norm(M, ord="fro").item() / math.sqrt(k))


# ── Attention patterns ────────────────────────────────────────────────────────
@torch.no_grad()
def mean_attention_pattern(model: HookedTransformer, loader: DataLoader) -> np.ndarray:
    """Returns (n_heads, n_ctx, n_ctx) mean attention patterns over the test set."""
    sum_pat = None
    n_total = 0
    for inputs, _ in loader:
        inputs = inputs.to(DEVICE)
        _, cache = model.run_with_cache(inputs)
        pat = cache["blocks.0.attn.hook_pattern"]  # (B, n_heads, n_ctx, n_ctx)
        if sum_pat is None:
            sum_pat = pat.sum(dim=0).detach().cpu().numpy().astype(np.float64)
        else:
            sum_pat += pat.sum(dim=0).detach().cpu().numpy().astype(np.float64)
        n_total += inputs.size(0)
    return sum_pat / n_total


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    # Load all models and compute per-head OV / QK summaries + subspaces
    models_to_load = [
        ("addition", s) for s in ADDITION_GROKKED_SEEDS
    ] + [
        ("subtraction", SUBTRACTION_GROKKED_SEED),
        ("subtraction", SUBTRACTION_NEAR_MISS_SEEDS[0]),
        ("subtraction", SUBTRACTION_NEAR_MISS_SEEDS[1]),
    ]

    # Storage
    ov_summaries = {}      # mkey -> {head_idx -> svd_summary dict (with U_top5)}
    qk_summaries = {}
    attention_patterns = {}  # for plotting & JSON

    for task, seed in models_to_load:
        mkey = f"{task}_seed{seed}"
        model = load_model(task, seed)
        circuits = head_circuits(model)

        ov_summaries[mkey] = {h: svd_summary(circuits[h]["OV"]) for h in range(N_HEADS)}
        qk_summaries[mkey] = {h: svd_summary(circuits[h]["QK"]) for h in range(N_HEADS)}

        eff_ranks = [round(ov_summaries[mkey][h]["effective_rank"], 2) for h in range(N_HEADS)]
        print(f"Analysing {mkey}... OV effective ranks: {eff_ranks}")

        # Attention patterns only for selected models
        if (task == "addition" and seed in ADDITION_GROKKED_SEEDS) or \
           (task == "subtraction" and seed == SUBTRACTION_GROKKED_SEED):
            loader = load_test_loader(task)
            pat = mean_attention_pattern(model, loader)  # (n_heads, n_ctx, n_ctx)
            attention_patterns[mkey] = pat

        # Free GPU memory before next model
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ── Cross-model alignments for HEAD 2 ─────────────────────────────────────
    head = 2
    sub_key = f"subtraction_seed{SUBTRACTION_GROKKED_SEED}"

    head2_ov_cross = []
    head2_qk_cross = []
    for s in ADDITION_GROKKED_SEEDS:
        mk = f"addition_seed{s}"
        head2_ov_cross.append(subspace_alignment(
            ov_summaries[mk][head]["U_top5"], ov_summaries[sub_key][head]["U_top5"]
        ))
        head2_qk_cross.append(subspace_alignment(
            qk_summaries[mk][head]["U_top5"], qk_summaries[sub_key][head]["U_top5"]
        ))

    # Within-addition consistency: 45 pairs, per head
    within_addition_ov = {h: [] for h in range(N_HEADS)}
    within_addition_qk = {h: [] for h in range(N_HEADS)}
    for s1, s2 in itertools.combinations(ADDITION_GROKKED_SEEDS, 2):
        m1, m2 = f"addition_seed{s1}", f"addition_seed{s2}"
        for h in range(N_HEADS):
            within_addition_ov[h].append(subspace_alignment(
                ov_summaries[m1][h]["U_top5"], ov_summaries[m2][h]["U_top5"]
            ))
            within_addition_qk[h].append(subspace_alignment(
                qk_summaries[m1][h]["U_top5"], qk_summaries[m2][h]["U_top5"]
            ))

    head2_ov_within_mean = float(np.mean(within_addition_ov[head]))
    head2_ov_within_std  = float(np.std(within_addition_ov[head]))
    head2_qk_within_mean = float(np.mean(within_addition_qk[head]))
    head2_qk_within_std  = float(np.std(within_addition_qk[head]))

    # ── Dominant head comparison: addition Head 3 vs subtraction Head 1 ──────
    dom_ov_pairs = []
    dom_qk_pairs = []
    for s in ADDITION_GROKKED_SEEDS:
        mk = f"addition_seed{s}"
        dom_ov_pairs.append(subspace_alignment(
            ov_summaries[mk][3]["U_top5"], ov_summaries[sub_key][1]["U_top5"]
        ))
        dom_qk_pairs.append(subspace_alignment(
            qk_summaries[mk][3]["U_top5"], qk_summaries[sub_key][1]["U_top5"]
        ))

    # ── Near-miss comparison (Head 2 OV) ──────────────────────────────────────
    # vs addition mean: take mean over 9 addition seeds of alignment(near-miss, add seed)
    def mean_align_to_addition(near_miss_key: str, head: int, summaries: dict) -> float:
        return float(np.mean([
            subspace_alignment(
                summaries[near_miss_key][head]["U_top5"],
                summaries[f"addition_seed{s}"][head]["U_top5"],
            ) for s in ADDITION_GROKKED_SEEDS
        ]))

    nm_seed2_key = f"subtraction_seed{SUBTRACTION_NEAR_MISS_SEEDS[0]}"
    nm_seed4_key = f"subtraction_seed{SUBTRACTION_NEAR_MISS_SEEDS[1]}"

    nm_seed2_vs_add_head2 = mean_align_to_addition(nm_seed2_key, 2, ov_summaries)
    nm_seed4_vs_add_head2 = mean_align_to_addition(nm_seed4_key, 2, ov_summaries)
    nm_seed2_vs_sub5_head2 = subspace_alignment(
        ov_summaries[nm_seed2_key][2]["U_top5"], ov_summaries[sub_key][2]["U_top5"]
    )
    nm_seed4_vs_sub5_head2 = subspace_alignment(
        ov_summaries[nm_seed4_key][2]["U_top5"], ov_summaries[sub_key][2]["U_top5"]
    )

    # ── Mean attention patterns for plotting ──────────────────────────────────
    addition_attention_mean = np.mean(
        [attention_patterns[f"addition_seed{s}"] for s in ADDITION_GROKKED_SEEDS], axis=0
    )

    # ── Build JSON output (strip non-serialisable U_top5 tensors) ────────────
    def strip_summary(s: dict) -> dict:
        return {k: v for k, v in s.items() if k != "U_top5"}

    ov_circuits_json = {
        mk: {f"head{h}": strip_summary(ov_summaries[mk][h]) for h in range(N_HEADS)}
        for mk in ov_summaries
    }
    qk_circuits_json = {
        mk: {f"head{h}": strip_summary(qk_summaries[mk][h]) for h in range(N_HEADS)}
        for mk in qk_summaries
    }

    attention_json = {
        "addition_mean": {
            f"head{h}": addition_attention_mean[h].tolist() for h in range(N_HEADS)
        },
        sub_key: {
            f"head{h}": attention_patterns[sub_key][h].tolist() for h in range(N_HEADS)
        },
    }

    cross_model_similarity = {
        "head2_ov_alignment": {
            "addition_vs_subtraction5_mean": float(np.mean(head2_ov_cross)),
            "addition_vs_subtraction5_std":  float(np.std(head2_ov_cross)),
            "within_addition_mean":          head2_ov_within_mean,
            "within_addition_std":           head2_ov_within_std,
        },
        "head2_qk_alignment": {
            "addition_vs_subtraction5_mean": float(np.mean(head2_qk_cross)),
            "addition_vs_subtraction5_std":  float(np.std(head2_qk_cross)),
            "within_addition_mean":          head2_qk_within_mean,
            "within_addition_std":           head2_qk_within_std,
        },
        "dominant_head_ov_alignment": {
            "addition_head3_vs_subtraction_head1_mean": float(np.mean(dom_ov_pairs)),
            "addition_head3_vs_subtraction_head1_std":  float(np.std(dom_ov_pairs)),
        },
        "dominant_head_qk_alignment": {
            "addition_head3_vs_subtraction_head1_mean": float(np.mean(dom_qk_pairs)),
            "addition_head3_vs_subtraction_head1_std":  float(np.std(dom_qk_pairs)),
        },
        "near_miss_vs_addition_head2_ov": {
            "seed2_mean": nm_seed2_vs_add_head2,
            "seed4_mean": nm_seed4_vs_add_head2,
        },
        "near_miss_vs_subtraction5_head2_ov": {
            "seed2": nm_seed2_vs_sub5_head2,
            "seed4": nm_seed4_vs_sub5_head2,
        },
    }

    output = {
        "ov_circuits": ov_circuits_json,
        "qk_circuits": qk_circuits_json,
        "attention_patterns": attention_json,
        "cross_model_similarity": cross_model_similarity,
    }
    out_path = os.path.join(METRICS_DIR, "circuit_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    def plot_singular_values(circ_summaries: dict, circ_name: str, path: str):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(1, TOP_SV + 1)

        all_add_sv = []
        for s in ADDITION_GROKKED_SEEDS:
            sv = circ_summaries[f"addition_seed{s}"][2]["singular_values_top10"]
            all_add_sv.append(sv)
            ax.plot(x, sv, color="lightsteelblue", alpha=0.7, linewidth=1)

        add_mean = np.mean(all_add_sv, axis=0)
        ax.plot(x, add_mean, color="darkblue", linewidth=2.5, label="Addition mean")

        sub5_sv = circ_summaries[sub_key][2]["singular_values_top10"]
        ax.plot(x, sub5_sv, color="green", linewidth=2.5, label="Subtraction seed 5 (grokked)")

        for nm_key, color, alpha, label in [
            (nm_seed2_key, "darkorange", 1.0, "Subtraction seed 2 (near-miss 0.95)"),
            (nm_seed4_key, "orange",     0.6, "Subtraction seed 4 (near-miss 0.92)"),
        ]:
            sv = circ_summaries[nm_key][2]["singular_values_top10"]
            ax.plot(x, sv, color=color, linestyle="dashed", alpha=alpha, linewidth=1.8, label=label)

        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Singular value magnitude")
        ax.set_title(f"{circ_name} Circuit Singular Values - Head 2")
        ax.set_xticks(x)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)

    plot_singular_values(ov_summaries, "OV", os.path.join(PLOTS_DIR, "ov_singular_values_head2.png"))
    plot_singular_values(qk_summaries, "QK", os.path.join(PLOTS_DIR, "qk_singular_values_head2.png"))

    def plot_attention_patterns(pattern_4x3x3: np.ndarray, title: str, path: str):
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.5))
        for h in range(N_HEADS):
            ax = axes[h // 2, h % 2]
            mat = pattern_4x3x3[h]
            im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for i in range(N_POS):
                for j in range(N_POS):
                    color = "white" if mat[i, j] > 0.5 else "black"
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            color=color, fontsize=10)
            ax.set_xticks(range(N_POS))
            ax.set_xticklabels(POS_LABELS_TO)
            ax.set_yticks(range(N_POS))
            ax.set_yticklabels(POS_LABELS_FROM)
            ax.set_title(f"Head {h}")
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)

    plot_attention_patterns(
        addition_attention_mean,
        "Mean Attention Patterns - Addition (9 grokked seeds)",
        os.path.join(PLOTS_DIR, "attention_patterns_addition_mean.png"),
    )
    plot_attention_patterns(
        attention_patterns[sub_key],
        "Mean Attention Patterns - Subtraction (seed 5, grokked)",
        os.path.join(PLOTS_DIR, "attention_patterns_subtraction_seed5.png"),
    )

    # Plot 5: Subspace alignment summary
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    width = 0.35
    within_means = [head2_ov_within_mean, head2_qk_within_mean]
    within_stds  = [head2_ov_within_std,  head2_qk_within_std]
    cross_means  = [float(np.mean(head2_ov_cross)), float(np.mean(head2_qk_cross))]
    cross_stds   = [float(np.std(head2_ov_cross)),  float(np.std(head2_qk_cross))]

    ax.bar(x - width/2, within_means, width, yerr=within_stds, capsize=5,
           label="Within-addition (45 pairs)", color="steelblue")
    ax.bar(x + width/2, cross_means, width, yerr=cross_stds, capsize=5,
           label="Addition vs subtraction_seed5 (9 pairs)", color="darkorange")
    ax.axhline(within_means[0], color="steelblue", linestyle="--", alpha=0.5,
               label="OV within-addition mean")

    ax.set_xticks(x)
    ax.set_xticklabels(["OV Circuit", "QK Circuit"])
    ax.set_ylabel("Subspace alignment (top-5)")
    ax.set_title("Head 2 Subspace Alignment: Within-Task vs Cross-Task")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "subspace_alignment_summary.png"), dpi=150)
    plt.close(fig)

    print("Saved 5 plots")

    # ── Console summary ───────────────────────────────────────────────────────
    def interpret(cross_mean: float, within_mean: float) -> str:
        if within_mean < 1e-9:
            return "UNDEFINED"
        ratio = cross_mean / within_mean
        if ratio >= 0.80:
            return "SHARED"
        elif ratio >= 0.40:
            return "PARTIAL"
        else:
            return "DIVERGENT"

    print("\n=== Circuit Similarity Summary ===")
    print("\nHead 2 OV alignment:")
    print(f"  Within addition (45 pairs):     {head2_ov_within_mean:.3f} +/- {head2_ov_within_std:.3f}")
    cmean, cstd = float(np.mean(head2_ov_cross)), float(np.std(head2_ov_cross))
    print(f"  Addition vs subtraction_seed5:  {cmean:.3f} +/- {cstd:.3f}")
    print(f"  Interpretation: {interpret(cmean, head2_ov_within_mean)}")

    print("\nHead 2 QK alignment:")
    print(f"  Within addition (45 pairs):     {head2_qk_within_mean:.3f} +/- {head2_qk_within_std:.3f}")
    cmean, cstd = float(np.mean(head2_qk_cross)), float(np.std(head2_qk_cross))
    print(f"  Addition vs subtraction_seed5:  {cmean:.3f} +/- {cstd:.3f}")
    print(f"  Interpretation: {interpret(cmean, head2_qk_within_mean)}")

    print("\nDominant head OV alignment (addition Head3 vs subtraction Head1):")
    dmean, dstd = float(np.mean(dom_ov_pairs)), float(np.std(dom_ov_pairs))
    print(f"  Mean: {dmean:.3f} +/- {dstd:.3f}")
    # Use within-addition Head 3 OV alignment as the within-task baseline
    h3_within = float(np.mean(within_addition_ov[3]))
    print(f"  Within-addition Head3 baseline: {h3_within:.3f}")
    print(f"  Interpretation: {interpret(dmean, h3_within)}")

    print("\nNear-miss subtraction Head 2 OV alignment:")
    print(f"  seed2 vs addition mean:    {nm_seed2_vs_add_head2:.3f}")
    print(f"  seed2 vs subtraction5:     {nm_seed2_vs_sub5_head2:.3f}")
    print(f"  seed4 vs addition mean:    {nm_seed4_vs_add_head2:.3f}")
    print(f"  seed4 vs subtraction5:     {nm_seed4_vs_sub5_head2:.3f}")


if __name__ == "__main__":
    main()
