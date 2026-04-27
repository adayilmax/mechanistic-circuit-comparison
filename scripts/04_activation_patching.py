import json
import os

import einops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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
P = 113
N_HEADS = 4
N_POS = 3
N_SAMPLES = 500
SAMPLE_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ADDITION_GROKKED_SEEDS = [0, 1, 2, 3, 4, 6, 7, 9, 10]
SUBTRACTION_GROKKED_SEEDS = [5]
SUBTRACTION_NEAR_MISS_SEEDS = [2, 4]

POS_LABELS = ["pos_a", "pos_=", "pos_b"]
HEAD_LABELS = [f"Head {h}" for h in range(N_HEADS)]


# ── Model loading & data ──────────────────────────────────────────────────────
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


def load_test_split(task: str):
    d = torch.load(os.path.join(DATA_DIR, f"{task}_test.pt"), weights_only=True)
    return d["a"], d["b"], d["label"]


def task_label(task: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if task == "addition":
        return (a + b) % P
    return (a - b) % P


def make_inputs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    sep = torch.full_like(a, 113)
    return torch.stack([a, sep, b], dim=1)


def sample_indices(n_total: int, n_sample: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_total, generator=g)[:n_sample]


# ── Per-head attention contribution ───────────────────────────────────────────
def per_head_attn_contribution(model: HookedTransformer, z: torch.Tensor) -> torch.Tensor:
    """
    z: (B, n_ctx, n_heads, d_head) from cache['blocks.0.attn.hook_z']
    Returns per-head contribution to attn_out: (B, n_ctx, n_heads, d_model).
    Sum over heads then + b_O equals model's hook_attn_out.
    """
    W_O = model.W_O[0]  # (n_heads, d_head, d_model)
    return einops.einsum(z, W_O, "b n h d, h d m -> b n h m")


# ── Patching core ─────────────────────────────────────────────────────────────
@torch.no_grad()
def get_run_state(model: HookedTransformer, inputs: torch.Tensor) -> dict:
    """
    Run model and return embed (pre-attn resid), per-head attn contributions,
    and final logits at position 2.
    """
    _, cache = model.run_with_cache(inputs)
    embed = cache["hook_embed"]
    if "hook_pos_embed" in cache:
        embed = embed + cache["hook_pos_embed"]
    z = cache["blocks.0.attn.hook_z"]
    per_head = per_head_attn_contribution(model, z)  # (B, n_ctx, n_heads, d_model)
    b_O = model.b_O[0]                               # (d_model,)
    attn_out = per_head.sum(dim=2) + b_O             # (B, n_ctx, d_model)
    resid_post = embed + attn_out
    logits_pos2 = resid_post[:, 2, :] @ model.W_U + model.b_U  # (B, d_vocab)
    return {
        "embed": embed,
        "per_head": per_head,
        "b_O": b_O,
        "logits": logits_pos2,
    }


@torch.no_grad()
def logit_diff(logits: torch.Tensor, c_clean: torch.Tensor, c_corrupt: torch.Tensor) -> torch.Tensor:
    """logit_diff per example = logits[c_clean] - logits[c_corrupt]"""
    idx = torch.arange(logits.size(0), device=logits.device)
    return logits[idx, c_clean] - logits[idx, c_corrupt]


@torch.no_grad()
def compute_patching_heatmap(
    target_model: HookedTransformer,
    source_state: dict,
    target_clean_state: dict,
    target_corrupt_state: dict,
    c_clean: torch.Tensor,
    c_corrupt: torch.Tensor,
) -> np.ndarray:
    """
    For each (head, position), patch source_state['per_head'] into target's
    corrupted run and measure LDR with respect to (c_clean, c_corrupt) as
    answered by the TARGET model (these are the target task's labels).

    Returns heatmap of shape (n_heads, n_pos) with mean LDR per cell.
    """
    W_U = target_model.W_U
    b_U = target_model.b_U
    b_O = target_corrupt_state["b_O"]

    clean_diff = logit_diff(target_clean_state["logits"], c_clean, c_corrupt)
    corrupt_diff = logit_diff(target_corrupt_state["logits"], c_clean, c_corrupt)
    denom = (clean_diff - corrupt_diff)  # (B,)

    # Avoid division by zero — guard with small epsilon-aware mask
    safe_denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)

    src_per_head = source_state["per_head"]              # (B, n_ctx, n_heads, d_model)
    corrupt_per_head = target_corrupt_state["per_head"]  # (B, n_ctx, n_heads, d_model)
    corrupt_embed = target_corrupt_state["embed"]        # (B, n_ctx, d_model)

    heatmap = np.zeros((N_HEADS, N_POS), dtype=np.float64)

    for h in range(N_HEADS):
        for p in range(N_POS):
            # Build patched per-head: copy corrupted, replace cell (h, p) with source
            patched = corrupt_per_head.clone()
            patched[:, p, h, :] = src_per_head[:, p, h, :]
            patched_attn_out = patched.sum(dim=2) + b_O  # (B, n_ctx, d_model)
            patched_resid = corrupt_embed + patched_attn_out
            patched_logits = patched_resid[:, 2, :] @ W_U + b_U
            patched_diff = logit_diff(patched_logits, c_clean, c_corrupt)
            ldr = (patched_diff - corrupt_diff) / safe_denom
            # Drop examples where denom was unsafe
            mask = denom.abs() >= 1e-8
            heatmap[h, p] = ldr[mask].mean().item()

    return heatmap


# ── Top-K & Jaccard ───────────────────────────────────────────────────────────
def top_k_cells(heatmap: np.ndarray, k: int = 3) -> list:
    """Return list of (head, position) tuples for the top-k cells by LDR."""
    flat = heatmap.flatten()
    top_idx = np.argsort(-flat)[:k]
    return [(int(i // N_POS), int(i % N_POS)) for i in top_idx]


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def consensus_circuit(seed_top_lists: list, threshold: int) -> list:
    """Cells appearing in top-3 of >= threshold seed lists."""
    counts = {}
    for top in seed_top_lists:
        for cell in top:
            counts[cell] = counts.get(cell, 0) + 1
    return sorted([cell for cell, c in counts.items() if c >= threshold])


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_heatmap(heatmap: np.ndarray, top_cells: list, title: str, path: str,
                 vmin: float = -1.0, vmax: float = 1.0):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(heatmap, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    for h in range(N_HEADS):
        for p in range(N_POS):
            ax.text(p, h, f"{heatmap[h, p]:.2f}", ha="center", va="center",
                    color="black", fontsize=10)
    # Black border around top cells
    for (h, p) in top_cells:
        rect = plt.Rectangle((p - 0.5, h - 0.5), 1, 1, fill=False,
                             edgecolor="black", linewidth=2.5)
        ax.add_patch(rect)
    ax.set_xticks(range(N_POS))
    ax.set_xticklabels(POS_LABELS)
    ax.set_yticks(range(N_HEADS))
    ax.set_yticklabels(HEAD_LABELS)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="LDR")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(heatmaps: list, titles: list, top_cells_list: list, path: str):
    vmax = max(np.abs(h).max() for h in heatmaps)
    vmin = -vmax
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    im = None
    for ax, heatmap, title, top_cells in zip(axes, heatmaps, titles, top_cells_list):
        im = ax.imshow(heatmap, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        for h in range(N_HEADS):
            for p in range(N_POS):
                ax.text(p, h, f"{heatmap[h, p]:.2f}", ha="center", va="center",
                        color="black", fontsize=9)
        for (h, p) in top_cells:
            rect = plt.Rectangle((p - 0.5, h - 0.5), 1, 1, fill=False,
                                 edgecolor="black", linewidth=2.5)
            ax.add_patch(rect)
        ax.set_xticks(range(N_POS))
        ax.set_xticklabels(POS_LABELS)
        ax.set_yticks(range(N_HEADS))
        ax.set_yticklabels(HEAD_LABELS)
        ax.set_title(title)
    fig.suptitle("Circuit Comparison: Addition | Subtraction | Cross-Task")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, label="LDR")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    # ── Sample subsets per task ───────────────────────────────────────────────
    task_inputs = {}
    for task in ["addition", "subtraction"]:
        a, b, _ = load_test_split(task)
        idx = sample_indices(len(a), N_SAMPLES, SAMPLE_SEED)
        a_clean = a[idx].to(DEVICE)
        b_clean = b[idx].to(DEVICE)
        b_corrupt = (b_clean + 1) % P
        c_clean = task_label(task, a_clean, b_clean).to(DEVICE)
        c_corrupt = task_label(task, a_clean, b_corrupt).to(DEVICE)
        task_inputs[task] = {
            "clean": make_inputs(a_clean, b_clean),
            "corrupt": make_inputs(a_clean, b_corrupt),
            "c_clean": c_clean,
            "c_corrupt": c_corrupt,
            "a_clean": a_clean,
            "b_clean": b_clean,
            "b_corrupt": b_corrupt,
        }

    results = {}

    # ── Within-task patching ──────────────────────────────────────────────────
    def patch_model(task: str, seed: int):
        model = load_model(task, seed)
        ti = task_inputs[task]
        clean_state = get_run_state(model, ti["clean"])
        corrupt_state = get_run_state(model, ti["corrupt"])
        heatmap = compute_patching_heatmap(
            target_model=model,
            source_state=clean_state,
            target_clean_state=clean_state,
            target_corrupt_state=corrupt_state,
            c_clean=ti["c_clean"],
            c_corrupt=ti["c_corrupt"],
        )
        top = top_k_cells(heatmap, 3)
        key = f"{task}_seed{seed}"
        results[key] = {
            "heatmap": heatmap.tolist(),
            "top3_cells": [list(c) for c in top],
            "task": task,
            "seed": seed,
        }
        print(f"Patching {key}... done (top-3: {top})")
        return model, clean_state

    addition_models = {}  # cache models for cross-task experiment

    for s in ADDITION_GROKKED_SEEDS:
        m, _ = patch_model("addition", s)
        addition_models[s] = m

    sub5_model, _ = patch_model("subtraction", 5)

    for s in SUBTRACTION_NEAR_MISS_SEEDS:
        patch_model("subtraction", s)

    # ── Cross-task patching: addition activations → subtraction model ─────────
    sub_ti = task_inputs["subtraction"]
    sub_clean_state = get_run_state(sub5_model, sub_ti["clean"])
    sub_corrupt_state = get_run_state(sub5_model, sub_ti["corrupt"])

    cross_heatmaps = []
    for s in ADDITION_GROKKED_SEEDS:
        add_model = addition_models[s]
        # Run addition model on the SAME clean inputs (subtraction test subset)
        add_clean_state = get_run_state(add_model, sub_ti["clean"])
        cross_hm = compute_patching_heatmap(
            target_model=sub5_model,
            source_state=add_clean_state,           # source: addition activations
            target_clean_state=sub_clean_state,     # subtraction's own clean baseline
            target_corrupt_state=sub_corrupt_state,
            c_clean=sub_ti["c_clean"],              # subtraction labels
            c_corrupt=sub_ti["c_corrupt"],
        )
        cross_heatmaps.append(cross_hm)
        print(f"Cross-task patching addition_seed{s} -> subtraction_seed5... done")

    cross_mean = np.mean(cross_heatmaps, axis=0)
    cross_top3 = top_k_cells(cross_mean, 3)
    results["cross_task_mean"] = {
        "heatmap": cross_mean.tolist(),
        "top3_cells": [list(c) for c in cross_top3],
    }

    # ── Save patching_results.json ────────────────────────────────────────────
    out_path = os.path.join(METRICS_DIR, "patching_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # ── Circuit overlap analysis ──────────────────────────────────────────────
    addition_top_lists = [
        [tuple(c) for c in results[f"addition_seed{s}"]["top3_cells"]]
        for s in ADDITION_GROKKED_SEEDS
    ]
    add_consensus = consensus_circuit(addition_top_lists, threshold=7)
    sub5_top = [tuple(c) for c in results["subtraction_seed5"]["top3_cells"]]
    cross_top = [tuple(c) for c in cross_top3]

    j_add_sub = jaccard(set(add_consensus), set(sub5_top))
    j_add_cross = jaccard(set(add_consensus), set(cross_top))

    if j_add_sub >= 0.5:
        interp = "shared"
    elif j_add_sub >= 0.2:
        interp = "partial"
    else:
        interp = "divergent"

    overlap = {
        "addition_consensus_circuit": [list(c) for c in add_consensus],
        "subtraction_seed5_top3": [list(c) for c in sub5_top],
        "cross_task_top3": [list(c) for c in cross_top],
        "jaccard_addition_vs_subtraction": j_add_sub,
        "jaccard_addition_vs_cross_task": j_add_cross,
        "interpretation": interp,
    }
    overlap_path = os.path.join(METRICS_DIR, "circuit_overlap.json")
    with open(overlap_path, "w") as f:
        json.dump(overlap, f, indent=2)
    print(f"Saved {overlap_path}")

    # ── Compute mean addition heatmap for plotting ────────────────────────────
    add_heatmaps = [
        np.array(results[f"addition_seed{s}"]["heatmap"]) for s in ADDITION_GROKKED_SEEDS
    ]
    add_mean = np.mean(add_heatmaps, axis=0)
    sub5_hm = np.array(results["subtraction_seed5"]["heatmap"])

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_heatmap(
        add_mean, add_consensus,
        f"Activation Patching — Addition Mean ({len(ADDITION_GROKKED_SEEDS)} seeds)",
        os.path.join(PLOTS_DIR, "patching_heatmap_addition_mean.png"),
    )
    plot_heatmap(
        sub5_hm, sub5_top,
        "Activation Patching — Subtraction (seed 5, grokked)",
        os.path.join(PLOTS_DIR, "patching_heatmap_subtraction_seed5.png"),
    )
    plot_heatmap(
        cross_mean, cross_top,
        "Activation Patching — Cross-Task (Addition->Subtraction Mean)",
        os.path.join(PLOTS_DIR, "patching_heatmap_cross_task_mean.png"),
    )
    plot_comparison(
        [add_mean, sub5_hm, cross_mean],
        ["Addition (mean of 9 seeds)", "Subtraction (seed 5)", "Cross-Task (mean)"],
        [add_consensus, sub5_top, cross_top],
        os.path.join(PLOTS_DIR, "patching_heatmap_comparison.png"),
    )
    print("Saved 4 heatmap plots")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n=== Patching Summary ===")
    print(f"Addition consensus circuit (>=7/9 seeds): {add_consensus}")
    print(f"Subtraction seed5 top-3:                  {sub5_top}")
    print(f"Cross-task mean top-3:                    {cross_top}")
    print(f"Jaccard (addition vs subtraction):        {j_add_sub:.3f}")
    print(f"Jaccard (addition vs cross-task):         {j_add_cross:.3f}")
    print(f"Interpretation:                           {interp}")


if __name__ == "__main__":
    main()
