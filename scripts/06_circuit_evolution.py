import json
import os
import re

import einops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft
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
P = 113
N_HEADS = 4
N_POS = 3
N_FREQ = P // 2 + 1  # 57
N_SAMPLES = 500
SAMPLE_SEED = 42
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

POS_LABELS = ["a", "=", "b"]
HEAD_LABELS = [f"H{h}" for h in range(N_HEADS)]
STAGES = ["pre", "grokked", "final"]


# ── Discovery & loading ──────────────────────────────────────────────────────
def discover_seeds(task: str) -> list:
    """Return sorted list of seed numbers that have all 3 checkpoints."""
    files = os.listdir(CKPT_DIR)
    pat = re.compile(rf"^{task}_seed(\d+)_(pre_grokking|grokked|final)\.pt$")
    by_seed = {}
    for f in files:
        m = pat.match(f)
        if m:
            seed = int(m.group(1))
            stage = m.group(2)
            by_seed.setdefault(seed, set()).add(stage)
    # Need all three stages
    full_seeds = sorted(s for s, stages in by_seed.items()
                        if {"pre_grokking", "grokked", "final"} <= stages)
    return full_seeds


def build_model(seed: int) -> HookedTransformer:
    return HookedTransformer(HookedTransformerConfig(**ARCH, seed=seed))


def load_ckpt(task: str, seed: int, stage: str) -> dict:
    fname = f"{task}_seed{seed}_{stage}.pt"
    return torch.load(os.path.join(CKPT_DIR, fname), weights_only=False)


def load_model_from_ckpt(seed: int, ckpt: dict) -> HookedTransformer:
    """Load with strict=False — pre-grokking checkpoints omit non-trainable buffers."""
    model = build_model(seed)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(DEVICE).eval()
    return model


def load_test_split(task: str):
    d = torch.load(os.path.join(DATA_DIR, f"{task}_test.pt"), weights_only=True)
    return d["a"], d["b"], d["label"]


def make_inputs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    sep = torch.full_like(a, 113)
    return torch.stack([a, sep, b], dim=1)


def task_label(task: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if task == "addition":
        return (a + b) % P
    return (a - b) % P


def sample_indices(n_total: int, n_sample: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_total, generator=g)[:n_sample]


def load_test_loader(task: str) -> DataLoader:
    a, b, label = load_test_split(task)
    inputs = make_inputs(a, b)
    return DataLoader(TensorDataset(inputs, label), batch_size=BATCH_SIZE, shuffle=False)


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


# ── Weight L2 diff verification ──────────────────────────────────────────────
def state_l2_diff(state_a: dict, state_b: dict) -> float:
    """L2 diff over the intersection of parameter names, skipping non-finite buffers
    (e.g. attn.IGNORE = -inf would otherwise yield NaN under (-inf) - (-inf))."""
    common = sorted(set(state_a.keys()) & set(state_b.keys()))
    parts_a, parts_b = [], []
    for k in common:
        va = state_a[k].float().cpu().flatten()
        vb = state_b[k].float().cpu().flatten()
        if not (torch.isfinite(va).all() and torch.isfinite(vb).all()):
            continue
        parts_a.append(va)
        parts_b.append(vb)
    if not parts_a:
        return 0.0
    return (torch.cat(parts_a) - torch.cat(parts_b)).norm().item()


# ── Patching (replicated from Phase 4) ───────────────────────────────────────
def per_head_attn_contribution(model: HookedTransformer, z: torch.Tensor) -> torch.Tensor:
    W_O = model.W_O[0]  # (n_heads, d_head, d_model)
    return einops.einsum(z, W_O, "b n h d, h d m -> b n h m")


@torch.no_grad()
def get_run_state(model: HookedTransformer, inputs: torch.Tensor) -> dict:
    _, cache = model.run_with_cache(inputs)
    embed = cache["hook_embed"]
    if "hook_pos_embed" in cache:
        embed = embed + cache["hook_pos_embed"]
    z = cache["blocks.0.attn.hook_z"]
    per_head = per_head_attn_contribution(model, z)
    b_O = model.b_O[0]
    attn_out = per_head.sum(dim=2) + b_O
    resid_post = embed + attn_out
    logits_pos2 = resid_post[:, 2, :] @ model.W_U + model.b_U
    return {"embed": embed, "per_head": per_head, "b_O": b_O, "logits": logits_pos2}


@torch.no_grad()
def logit_diff(logits: torch.Tensor, c_clean: torch.Tensor, c_corrupt: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(logits.size(0), device=logits.device)
    return logits[idx, c_clean] - logits[idx, c_corrupt]


@torch.no_grad()
def compute_patching_heatmap(model: HookedTransformer, clean_state: dict,
                              corrupt_state: dict, c_clean: torch.Tensor,
                              c_corrupt: torch.Tensor) -> np.ndarray:
    W_U, b_U, b_O = model.W_U, model.b_U, corrupt_state["b_O"]
    clean_diff   = logit_diff(clean_state["logits"], c_clean, c_corrupt)
    corrupt_diff = logit_diff(corrupt_state["logits"], c_clean, c_corrupt)
    denom = clean_diff - corrupt_diff
    safe_denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)
    mask = denom.abs() >= 1e-8

    src_per_head     = clean_state["per_head"]
    corrupt_per_head = corrupt_state["per_head"]
    corrupt_embed    = corrupt_state["embed"]

    heatmap = np.zeros((N_HEADS, N_POS), dtype=np.float64)
    for h in range(N_HEADS):
        for p in range(N_POS):
            patched = corrupt_per_head.clone()
            patched[:, p, h, :] = src_per_head[:, p, h, :]
            patched_attn_out = patched.sum(dim=2) + b_O
            patched_resid = corrupt_embed + patched_attn_out
            patched_logits = patched_resid[:, 2, :] @ W_U + b_U
            patched_diff = logit_diff(patched_logits, c_clean, c_corrupt)
            ldr = (patched_diff - corrupt_diff) / safe_denom
            heatmap[h, p] = ldr[mask].mean().item()
    return heatmap


def top_k_cells(heatmap: np.ndarray, k: int = 3) -> list:
    flat = heatmap.flatten()
    top_idx = np.argsort(-flat)[:k]
    return [(int(i // N_POS), int(i % N_POS)) for i in top_idx]


# ── Asymmetry & Fourier ──────────────────────────────────────────────────────
@torch.no_grad()
def head2_asymmetry(model: HookedTransformer, loader: DataLoader) -> float:
    sum_attn = np.zeros(3, dtype=np.float64)
    n_total = 0
    for inputs, _ in loader:
        inputs = inputs.to(DEVICE)
        _, cache = model.run_with_cache(inputs)
        from_b_h2 = cache["blocks.0.attn.hook_pattern"][:, 2, 2, :]  # (B, n_ctx)
        sum_attn += from_b_h2.sum(dim=0).detach().cpu().numpy().astype(np.float64)
        n_total += inputs.size(0)
    mean_attn = sum_attn / n_total
    return float(mean_attn[0] - mean_attn[2])


def fourier_concentration(model: HookedTransformer, k: int = 3) -> float:
    W_E = model.W_E[:P, :].detach().cpu()
    F = torch.fft.rfft(W_E, dim=0)
    spectrum = ((F.real ** 2 + F.imag ** 2).sum(dim=-1)).numpy()
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    return float(np.sort(spectrum)[-k:].sum() / total)


# ── Per-checkpoint analysis ──────────────────────────────────────────────────
def analyse_stage(task: str, seed: int, stage: str, ckpt: dict,
                  patching_inputs: dict, loader: DataLoader) -> dict:
    model = load_model_from_ckpt(seed, ckpt)
    clean_state   = get_run_state(model, patching_inputs["clean"])
    corrupt_state = get_run_state(model, patching_inputs["corrupt"])
    heatmap = compute_patching_heatmap(model, clean_state, corrupt_state,
                                        patching_inputs["c_clean"],
                                        patching_inputs["c_corrupt"])
    top3 = top_k_cells(heatmap, 3)
    asym = head2_asymmetry(model, loader)
    conc = fourier_concentration(model)
    epoch = int(ckpt.get("epoch", -1))

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "epoch": epoch,
        "heatmap": heatmap,
        "top3": top3,
        "asymmetry": asym,
        "fourier_concentration": conc,
    }


def build_patching_inputs(task: str) -> dict:
    a, b, _ = load_test_split(task)
    idx = sample_indices(len(a), N_SAMPLES, SAMPLE_SEED)
    a_clean = a[idx].to(DEVICE)
    b_clean = b[idx].to(DEVICE)
    b_corrupt = (b_clean + 1) % P
    return {
        "clean":     make_inputs(a_clean, b_clean),
        "corrupt":   make_inputs(a_clean, b_corrupt),
        "c_clean":   task_label(task, a_clean, b_clean).to(DEVICE),
        "c_corrupt": task_label(task, a_clean, b_corrupt).to(DEVICE),
    }


# ── Plotting ─────────────────────────────────────────────────────────────────
def draw_heatmap(ax, heatmap: np.ndarray, top_cells: list, title: str,
                 vmin: float = -0.5, vmax: float = 1.0):
    im = ax.imshow(heatmap, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    for h in range(N_HEADS):
        for p in range(N_POS):
            ax.text(p, h, f"{heatmap[h, p]:.1f}", ha="center", va="center",
                    fontsize=7, color="black")
    for (h, p) in top_cells:
        rect = plt.Rectangle((p - 0.5, h - 0.5), 1, 1, fill=False,
                             edgecolor="black", linewidth=1.8)
        ax.add_patch(rect)
    ax.set_xticks(range(N_POS))
    ax.set_xticklabels(POS_LABELS, fontsize=7)
    ax.set_yticks(range(N_HEADS))
    ax.set_yticklabels(HEAD_LABELS, fontsize=7)
    ax.set_title(title, fontsize=8)
    return im


def plot_addition_evolution(addition_results: dict, path: str):
    seeds = sorted(addition_results.keys())
    n = len(seeds)
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 1.8 * n))
    if n == 1:
        axes = axes[None, :]
    im = None
    for i, seed in enumerate(seeds):
        r = addition_results[seed]
        ep_pre = r["pre"]["epoch"] if r["pre"] is not None else "—"
        ep_g = r["grokked"]["epoch"]
        ep_f = r["final"]["epoch"]
        axes[i, 0].set_ylabel(f"seed {seed}\n(pre:{ep_pre} grok:{ep_g} fin:{ep_f})",
                              fontsize=7, rotation=0, labelpad=42, ha="right", va="center")
        if r["pre"] is not None:
            im = draw_heatmap(axes[i, 0], r["pre"]["heatmap"], r["pre"]["top3"],
                              "Pre-grokking" if i == 0 else "")
        else:
            axes[i, 0].text(0.5, 0.5, "skipped\n(L2<1.0)",
                            ha="center", va="center", transform=axes[i, 0].transAxes)
            axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
            if i == 0:
                axes[i, 0].set_title("Pre-grokking", fontsize=8)
        im = draw_heatmap(axes[i, 1], r["grokked"]["heatmap"], r["grokked"]["top3"],
                          "Grokked" if i == 0 else "")
        im = draw_heatmap(axes[i, 2], r["final"]["heatmap"], r["final"]["top3"],
                          "Final" if i == 0 else "")
    fig.suptitle("Circuit Evolution Across Training Stages - Addition", fontsize=11)
    fig.subplots_adjust(right=0.88, hspace=0.5, wspace=0.4, left=0.18)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="LDR")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_subtraction_evolution(sub_result: dict, path: str):
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.2))
    im = None
    if sub_result["pre"] is not None:
        im = draw_heatmap(axes[0], sub_result["pre"]["heatmap"], sub_result["pre"]["top3"],
                          f"Pre-grokking (ep {sub_result['pre']['epoch']})")
    else:
        axes[0].text(0.5, 0.5, "skipped (L2<1.0)", ha="center", va="center")
    im = draw_heatmap(axes[1], sub_result["grokked"]["heatmap"], sub_result["grokked"]["top3"],
                      f"Grokked (ep {sub_result['grokked']['epoch']})")
    im = draw_heatmap(axes[2], sub_result["final"]["heatmap"], sub_result["final"]["top3"],
                      f"Final (ep {sub_result['final']['epoch']})")
    fig.suptitle("Circuit Evolution Across Training Stages - Subtraction (seed5)", fontsize=11)
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="LDR")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(addition_results: dict, sub_result: dict,
                    metric_key: str, ylabel: str, title_suffix: str,
                    q2_annotation: str | None, path: str):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(3)

    # Addition: per-seed lines + mean
    addition_traces = []
    for seed in sorted(addition_results.keys()):
        r = addition_results[seed]
        if r["pre"] is None:
            continue
        trace = [r["pre"][metric_key], r["grokked"][metric_key], r["final"][metric_key]]
        addition_traces.append(trace)
        ax_l.plot(x, trace, color="lightsteelblue", alpha=0.6, linewidth=1)
    if addition_traces:
        mean_trace = np.mean(addition_traces, axis=0)
        ax_l.plot(x, mean_trace, color="darkblue", linewidth=2.5, marker="o",
                  markersize=8, label=f"Mean (n={len(addition_traces)})")
    ax_l.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(["pre", "grokked", "final"])
    ax_l.set_ylabel(ylabel)
    ax_l.set_title(f"{ylabel} Trajectory - Addition")
    ax_l.legend(fontsize=9)

    # Subtraction
    if sub_result["pre"] is not None:
        sub_trace = [sub_result["pre"][metric_key],
                     sub_result["grokked"][metric_key],
                     sub_result["final"][metric_key]]
        ax_r.plot(x, sub_trace, color="green", linewidth=2.5, marker="o",
                  markersize=10, label="seed5")
    else:
        sub_trace = [None,
                     sub_result["grokked"][metric_key],
                     sub_result["final"][metric_key]]
        ax_r.plot(x[1:], sub_trace[1:], color="green", linewidth=2.5, marker="o",
                  markersize=10, label="seed5 (no pre)")
    ax_r.axhline(0, color="grey", linestyle="--", linewidth=1)
    if q2_annotation is not None:
        ax_r.annotate(q2_annotation,
                      xy=(1, sub_result["grokked"][metric_key]),
                      xytext=(0.5, sub_result["grokked"][metric_key] + 0.05),
                      fontsize=8, color="darkgreen")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(["pre", "grokked", "final"])
    ax_r.set_ylabel(ylabel)
    ax_r.set_title(f"{ylabel} Trajectory - Subtraction (seed5)")
    ax_r.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_jaccard_summary(addition_results: dict, path: str):
    j_pg, j_gf, j_pf = [], [], []
    for r in addition_results.values():
        if r["pre"] is not None:
            j_pg.append(jaccard(set(r["pre"]["top3"]), set(r["grokked"]["top3"])))
            j_pf.append(jaccard(set(r["pre"]["top3"]), set(r["final"]["top3"])))
        j_gf.append(jaccard(set(r["grokked"]["top3"]), set(r["final"]["top3"])))

    means = [np.mean(j_pg) if j_pg else 0,
             np.mean(j_gf) if j_gf else 0,
             np.mean(j_pf) if j_pf else 0]
    stds  = [np.std(j_pg) if j_pg else 0,
             np.std(j_gf) if j_gf else 0,
             np.std(j_pf) if j_pf else 0]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(3)
    ax.bar(x, means, yerr=stds, capsize=8, color="steelblue", edgecolor="black")
    ax.axhline(0.33, color="grey", linestyle="--", alpha=0.5, label="0.33 threshold")
    ax.axhline(0.67, color="grey", linestyle="--", alpha=0.7, label="0.67 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Pre->Grokked", "Grokked->Final", "Pre->Final"])
    ax.set_ylabel("Jaccard similarity (top-3 cells)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Circuit Stability Across Training Transitions - Addition")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def per_seed_block(task: str, seed: int, ckpts: dict,
                   patching_inputs: dict, loader: DataLoader,
                   skip_pre: bool) -> tuple[dict, dict]:
    """Run all three (or two) stages for a seed; return (raw_results, json_block)."""
    pre_data     = None if skip_pre else analyse_stage(task, seed, "pre", ckpts["pre"], patching_inputs, loader)
    grokked_data = analyse_stage(task, seed, "grokked", ckpts["grokked"], patching_inputs, loader)
    final_data   = analyse_stage(task, seed, "final", ckpts["final"], patching_inputs, loader)

    raw = {"pre": pre_data, "grokked": grokked_data, "final": final_data}

    j_pg = jaccard(set(pre_data["top3"]), set(grokked_data["top3"])) if pre_data else None
    j_gf = jaccard(set(grokked_data["top3"]), set(final_data["top3"]))
    j_pf = jaccard(set(pre_data["top3"]), set(final_data["top3"])) if pre_data else None

    block = {
        "pre_grokking_epoch": pre_data["epoch"] if pre_data else None,
        "grokked_epoch": grokked_data["epoch"],
        "final_epoch": final_data["epoch"],
        "patching": {
            "pre_heatmap": pre_data["heatmap"].tolist() if pre_data else None,
            "grokked_heatmap": grokked_data["heatmap"].tolist(),
            "final_heatmap": final_data["heatmap"].tolist(),
            "pre_top3": [list(c) for c in pre_data["top3"]] if pre_data else None,
            "grokked_top3": [list(c) for c in grokked_data["top3"]],
            "final_top3":   [list(c) for c in final_data["top3"]],
            "jaccard_pre_grokked": j_pg,
            "jaccard_grokked_final": j_gf,
            "jaccard_pre_final": j_pf,
        },
        "asymmetry": {
            "pre": pre_data["asymmetry"] if pre_data else None,
            "grokked": grokked_data["asymmetry"],
            "final": final_data["asymmetry"],
            "delta_pre_to_grokked": (grokked_data["asymmetry"] - pre_data["asymmetry"]) if pre_data else None,
            "delta_grokked_to_final": final_data["asymmetry"] - grokked_data["asymmetry"],
        },
        "fourier": {
            "pre_concentration": pre_data["fourier_concentration"] if pre_data else None,
            "grokked_concentration": grokked_data["fourier_concentration"],
            "final_concentration": final_data["fourier_concentration"],
            "delta_pre_to_grokked": (grokked_data["fourier_concentration"] - pre_data["fourier_concentration"]) if pre_data else None,
            "delta_grokked_to_final": final_data["fourier_concentration"] - grokked_data["fourier_concentration"],
        },
    }
    return raw, block


def main():
    print(f"Device: {DEVICE}\n")
    print("=== Phase 6: Circuit Evolution Analysis ===\n")

    # ── Discover seeds ────────────────────────────────────────────────────────
    add_seeds = discover_seeds("addition")
    sub_seeds = discover_seeds("subtraction")
    print(f"Discovered addition seeds with all 3 checkpoints: {add_seeds}")
    print(f"Discovered subtraction seeds with all 3 checkpoints: {sub_seeds}\n")

    # Patching inputs (seeded reproducibility — sample once per task)
    add_patching = build_patching_inputs("addition")
    sub_patching = build_patching_inputs("subtraction")
    add_loader = load_test_loader("addition")
    sub_loader = load_test_loader("subtraction")

    addition_results = {}
    addition_blocks  = {}

    for seed in add_seeds:
        ckpts = {s: load_ckpt("addition", seed, s) for s in ["pre_grokking", "grokked", "final"]}
        ckpts = {"pre": ckpts["pre_grokking"], "grokked": ckpts["grokked"], "final": ckpts["final"]}

        # Verify weight L2 diffs
        l2_pre_grok = state_l2_diff(ckpts["pre"]["model_state_dict"],
                                    ckpts["grokked"]["model_state_dict"])
        l2_grok_fin = state_l2_diff(ckpts["grokked"]["model_state_dict"],
                                    ckpts["final"]["model_state_dict"])

        skip_pre = l2_pre_grok < 1.0
        if skip_pre:
            print(f"  WARNING: addition seed{seed} pre/grokked L2={l2_pre_grok:.3f} < 1.0 "
                  f"— skipping pre-grokking stage")

        raw, block = per_seed_block("addition", seed, ckpts, add_patching,
                                    add_loader, skip_pre)
        block["weight_l2_diff_pre_grokked"] = l2_pre_grok
        block["weight_l2_diff_grokked_final"] = l2_grok_fin
        addition_results[seed] = raw
        addition_blocks[f"seed{seed}"] = block

        print(f"[addition | seed {seed}]")
        print(f"  Pre-grokking epoch: {block['pre_grokking_epoch']} | "
              f"Grokked epoch: {block['grokked_epoch']} | "
              f"Final epoch: {block['final_epoch']}")
        print(f"  Weight L2 diff (pre->grokked): {l2_pre_grok:.2f}")
        print(f"  Weight L2 diff (grokked->final): {l2_grok_fin:.2f}")
        if not skip_pre:
            print(f"  Patching Jaccard pre->grokked: {block['patching']['jaccard_pre_grokked']:.3f}")
        print(f"  Patching Jaccard grokked->final: {block['patching']['jaccard_grokked_final']:.3f}")
        pre_a = block["asymmetry"]["pre"]
        pre_f = block["fourier"]["pre_concentration"]
        pre_a_str = f"{pre_a:.3f}" if pre_a is not None else "N/A"
        pre_f_str = f"{pre_f:.3f}" if pre_f is not None else "N/A"
        print(f"  Head 2 asymmetry: pre={pre_a_str} "
              f"grokked={block['asymmetry']['grokked']:.3f} "
              f"final={block['asymmetry']['final']:.3f}")
        print(f"  Fourier conc: pre={pre_f_str} "
              f"grokked={block['fourier']['grokked_concentration']:.3f} "
              f"final={block['fourier']['final_concentration']:.3f}\n")

    # ── Subtraction (seed 5 only typically) ──────────────────────────────────
    subtraction_results = {}
    subtraction_blocks  = {}
    for seed in sub_seeds:
        ckpts = {s: load_ckpt("subtraction", seed, s) for s in ["pre_grokking", "grokked", "final"]}
        ckpts = {"pre": ckpts["pre_grokking"], "grokked": ckpts["grokked"], "final": ckpts["final"]}

        l2_pre_grok = state_l2_diff(ckpts["pre"]["model_state_dict"],
                                    ckpts["grokked"]["model_state_dict"])
        l2_grok_fin = state_l2_diff(ckpts["grokked"]["model_state_dict"],
                                    ckpts["final"]["model_state_dict"])

        skip_pre = l2_pre_grok < 1.0
        if skip_pre:
            print(f"  WARNING: subtraction seed{seed} pre/grokked L2={l2_pre_grok:.3f} < 1.0 "
                  f"— skipping pre-grokking stage")

        raw, block = per_seed_block("subtraction", seed, ckpts, sub_patching,
                                    sub_loader, skip_pre)
        block["weight_l2_diff_pre_grokked"] = l2_pre_grok
        block["weight_l2_diff_grokked_final"] = l2_grok_fin
        subtraction_results[seed] = raw
        subtraction_blocks[f"seed{seed}"] = block

        print(f"[subtraction | seed {seed}]")
        print(f"  Pre-grokking epoch: {block['pre_grokking_epoch']} | "
              f"Grokked epoch: {block['grokked_epoch']} | "
              f"Final epoch: {block['final_epoch']}")
        print(f"  Weight L2 diff (pre->grokked): {l2_pre_grok:.2f}")
        print(f"  Weight L2 diff (grokked->final): {l2_grok_fin:.2f}")
        if not skip_pre:
            print(f"  Patching Jaccard pre->grokked: {block['patching']['jaccard_pre_grokked']:.3f}")
        print(f"  Patching Jaccard grokked->final: {block['patching']['jaccard_grokked_final']:.3f}")
        pre_a = block["asymmetry"]["pre"]
        if pre_a is None:
            print(f"  Head 2 asymmetry: pre=N/A "
                  f"grokked={block['asymmetry']['grokked']:.3f} "
                  f"final={block['asymmetry']['final']:.3f}")
        else:
            print(f"  Head 2 asymmetry: pre={pre_a:.3f} "
                  f"grokked={block['asymmetry']['grokked']:.3f} "
                  f"final={block['asymmetry']['final']:.3f}")
        pre_f = block["fourier"]["pre_concentration"]
        if pre_f is None:
            print(f"  Fourier conc: pre=N/A "
                  f"grokked={block['fourier']['grokked_concentration']:.3f} "
                  f"final={block['fourier']['final_concentration']:.3f}\n")
        else:
            print(f"  Fourier conc: pre={pre_f:.3f} "
                  f"grokked={block['fourier']['grokked_concentration']:.3f} "
                  f"final={block['fourier']['final_concentration']:.3f}\n")

    # ── Aggregates & key questions ───────────────────────────────────────────
    j_pg_all = [b["patching"]["jaccard_pre_grokked"] for b in addition_blocks.values()
                if b["patching"]["jaccard_pre_grokked"] is not None]
    j_gf_all = [b["patching"]["jaccard_grokked_final"] for b in addition_blocks.values()]
    j_pf_all = [b["patching"]["jaccard_pre_final"] for b in addition_blocks.values()
                if b["patching"]["jaccard_pre_final"] is not None]

    j_pg_mean, j_pg_std = (float(np.mean(j_pg_all)), float(np.std(j_pg_all))) if j_pg_all else (None, None)
    j_gf_mean, j_gf_std = float(np.mean(j_gf_all)), float(np.std(j_gf_all))
    j_pf_mean, j_pf_std = (float(np.mean(j_pf_all)), float(np.std(j_pf_all))) if j_pf_all else (None, None)

    fc_pre = [b["fourier"]["pre_concentration"] for b in addition_blocks.values()
              if b["fourier"]["pre_concentration"] is not None]
    fc_grok = [b["fourier"]["grokked_concentration"] for b in addition_blocks.values()]
    fc_fin  = [b["fourier"]["final_concentration"] for b in addition_blocks.values()]

    fc_pre_mean = float(np.mean(fc_pre)) if fc_pre else None
    fc_grok_mean = float(np.mean(fc_grok))
    fc_fin_mean = float(np.mean(fc_fin))

    # Q1
    if j_pg_mean is not None and j_pg_mean < 0.2 and j_gf_mean > 0.5:
        q1 = "SUDDEN: circuit absent pre-grokking, stable post-grokking"
    elif j_pg_mean is not None and j_pg_mean > 0.5:
        q1 = "GRADUAL: circuit partially present before grokking"
    else:
        q1 = "MIXED: partial circuit formation, report per-seed details"
    if j_pg_mean is not None:
        q1 += f" (mean pre->grokked Jaccard = {j_pg_mean:.3f})"

    # Q2 (subtraction seed 5)
    sub5_block = subtraction_blocks.get("seed5")
    if sub5_block is not None:
        sub5_grok_asym = sub5_block["asymmetry"]["grokked"]
        sub5_delta_gf  = sub5_block["asymmetry"]["delta_grokked_to_final"]
        if abs(sub5_grok_asym) > 0.3:
            q2 = "PRESENT_AT_GROKKING: asymmetric circuit forms with grokking"
        elif abs(sub5_delta_gf) > 0.2:
            q2 = "DEVELOPS_AFTER_GROKKING: asymmetry grows post-grokking"
        else:
            q2 = "NO_SIGNIFICANT_CHANGE"
    else:
        q2 = "NO_SUBTRACTION_DATA"

    # Q3
    if fc_pre_mean is not None and fc_grok_mean > fc_pre_mean + 0.1:
        q3 = "YES: Fourier structure sharpens at grokking transition"
    elif fc_pre_mean is not None:
        q3 = "NO: Fourier structure present before grokking"
    else:
        q3 = "INSUFFICIENT_DATA"
    if fc_pre_mean is not None:
        q3 += f" (pre={fc_pre_mean:.3f}, grokked={fc_grok_mean:.3f})"

    # Q4
    if j_gf_mean > 0.67:
        q4 = "STABLE: circuit unchanged after grokking"
    elif j_gf_mean > 0.33:
        q4 = "PARTIALLY_STABLE"
    else:
        q4 = "UNSTABLE: circuit continues to evolve after grokking"
    q4 += f" (mean grokked->final Jaccard = {j_gf_mean:.3f})"

    print("\n=== Phase 6 Key Findings ===")
    print(f"Q1 Circuit formation:      {q1}")
    print(f"Q2 Subtraction asymmetry:  {q2}")
    print(f"Q3 Fourier concentration:  {q3}")
    print(f"Q4 Post-grokking stability:{q4}")

    print("\n=== Jaccard Summary (Addition) ===")
    if j_pg_mean is not None:
        print(f"Pre->Grokked:   {j_pg_mean:.3f} +/- {j_pg_std:.3f}")
    else:
        print(f"Pre->Grokked:   N/A")
    print(f"Grokked->Final: {j_gf_mean:.3f} +/- {j_gf_std:.3f}")
    if j_pf_mean is not None:
        print(f"Pre->Final:     {j_pf_mean:.3f} +/- {j_pf_std:.3f}")

    if sub5_block is not None:
        print("\n=== Subtraction seed5 Asymmetry ===")
        pre_a = sub5_block["asymmetry"]["pre"]
        print(f"Pre-grokking: {pre_a if pre_a is None else f'{pre_a:.3f}'}")
        print(f"Grokked:      {sub5_block['asymmetry']['grokked']:.3f}")
        print(f"Final:        {sub5_block['asymmetry']['final']:.3f}")
        print(f"Interpretation: {q2}")

    # ── JSON output ──────────────────────────────────────────────────────────
    summary_block = {
        "addition_jaccard_pre_grokked_mean": j_pg_mean,
        "addition_jaccard_pre_grokked_std":  j_pg_std,
        "addition_jaccard_grokked_final_mean": j_gf_mean,
        "addition_jaccard_grokked_final_std":  j_gf_std,
        "addition_fourier_concentration_pre_mean":     fc_pre_mean,
        "addition_fourier_concentration_grokked_mean": fc_grok_mean,
        "addition_fourier_concentration_final_mean":   fc_fin_mean,
        "subtraction_seed5_asymmetry_pre":     sub5_block["asymmetry"]["pre"] if sub5_block else None,
        "subtraction_seed5_asymmetry_grokked": sub5_block["asymmetry"]["grokked"] if sub5_block else None,
        "subtraction_seed5_asymmetry_final":   sub5_block["asymmetry"]["final"] if sub5_block else None,
    }

    output = {
        "addition": addition_blocks,
        "subtraction": subtraction_blocks,
        "summary": summary_block,
        "key_question_answers": {
            "Q1_circuit_formation": q1,
            "Q2_subtraction_asymmetry_timing": q2,
            "Q3_fourier_concentration": q3,
            "Q4_post_grokking_stability": q4,
        },
    }
    out_path = os.path.join(METRICS_DIR, "phase6_circuit_evolution.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_addition_evolution(addition_results,
                            os.path.join(PLOTS_DIR, "phase6_patching_evolution.png"))
    if sub5_block is not None:
        plot_subtraction_evolution(subtraction_results[5],
                                   os.path.join(PLOTS_DIR, "phase6_patching_subtraction.png"))
        plot_trajectory(addition_results, subtraction_results[5],
                        "asymmetry", "Head 2 asymmetry", "Head 2 Asymmetry",
                        q2.split(":")[0], os.path.join(PLOTS_DIR, "phase6_asymmetry_trajectories.png"))
        plot_trajectory(addition_results, subtraction_results[5],
                        "fourier_concentration", "Fourier concentration",
                        "Fourier Concentration",
                        None, os.path.join(PLOTS_DIR, "phase6_fourier_trajectories.png"))
    plot_jaccard_summary(addition_results,
                          os.path.join(PLOTS_DIR, "phase6_jaccard_summary.png"))
    print("Saved 5 plots.")


if __name__ == "__main__":
    main()
