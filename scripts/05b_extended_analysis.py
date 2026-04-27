import json
import os

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
N_FREQ = P // 2 + 1  # 57
TOP_K_FREQ = 5
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ADDITION_GROKKED_SEEDS = [0, 1, 2, 3, 4, 6, 7, 9, 10]
SUB5_KEY = "subtraction_seed5"
NEAR_MISS_SUBTRACTION_SEEDS = [2, 4]


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_model(seed: int) -> HookedTransformer:
    return HookedTransformer(HookedTransformerConfig(**ARCH, seed=seed))


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


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


# ── Analysis 1: Fourier ──────────────────────────────────────────────────────
def fourier_power_spectrum(model: HookedTransformer) -> np.ndarray:
    """Returns (N_FREQ,) total power per frequency over the embedding matrix."""
    W_E = model.W_E[:P, :].detach().cpu()  # (113, 128)
    F = torch.fft.rfft(W_E, dim=0)         # (57, 128) complex
    power = (F.real ** 2 + F.imag ** 2).sum(dim=-1)  # (57,)
    return power.numpy()


def top_k_freqs(spectrum: np.ndarray, k: int) -> tuple:
    idx = np.argsort(-spectrum)[:k]
    return idx.tolist(), spectrum[idx].tolist()


def concentration_ratio(spectrum: np.ndarray, k: int = 3) -> float:
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    top_k = np.sort(spectrum)[-k:].sum()
    return float(top_k / total)


# ── Analysis 2: Attention asymmetry ──────────────────────────────────────────
@torch.no_grad()
def per_head_from_b_attention(model: HookedTransformer, loader: DataLoader) -> np.ndarray:
    """Returns (n_heads, n_ctx) mean attention from pos_b to each src position."""
    sum_attn = np.zeros((N_HEADS, 3), dtype=np.float64)
    n_total = 0
    for inputs, _ in loader:
        inputs = inputs.to(DEVICE)
        _, cache = model.run_with_cache(inputs)
        pat = cache["blocks.0.attn.hook_pattern"]   # (B, n_heads, n_ctx, n_ctx)
        from_b = pat[:, :, 2, :]                     # (B, n_heads, n_ctx)
        sum_attn += from_b.sum(dim=0).detach().cpu().numpy().astype(np.float64)
        n_total += inputs.size(0)
    return sum_attn / n_total


def asymmetry_per_head(from_b_attn: np.ndarray) -> np.ndarray:
    """attn_to_a - attn_to_b for each head."""
    return from_b_attn[:, 0] - from_b_attn[:, 2]


# ── Analysis 3: Causal scrubbing ─────────────────────────────────────────────
@torch.no_grad()
def evaluate_with_pattern_hook(
    model: HookedTransformer, loader: DataLoader, hook_fn
) -> float:
    """Returns argmax accuracy under the given hook on hook_pattern."""
    correct, total = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        logits = model.run_with_hooks(
            inputs,
            fwd_hooks=[("blocks.0.attn.hook_pattern", hook_fn)],
        )[:, 2, :]
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += inputs.size(0)
    return correct / total


@torch.no_grad()
def evaluate_baseline(model: HookedTransformer, loader: DataLoader) -> float:
    correct, total = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        logits = model(inputs)[:, 2, :]
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += inputs.size(0)
    return correct / total


def make_force_pattern_hook(replacement_row: torch.Tensor):
    """
    replacement_row: shape (3,) — values for [src=0, src=1, src=2] at head 2, dest 2.
    """
    rep = replacement_row.to(DEVICE).float()
    def hook_fn(pattern, hook):
        # pattern: (B, n_heads, n_ctx, n_ctx)
        new_pat = pattern.clone()
        new_pat[:, 2, 2, :] = rep.unsqueeze(0).expand(pattern.size(0), -1)
        return new_pat
    return hook_fn


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    # Models we need across analyses
    all_specs = (
        [("addition", s) for s in ADDITION_GROKKED_SEEDS]
        + [("subtraction", 5)]
        + [("subtraction", s) for s in NEAR_MISS_SUBTRACTION_SEEDS]
    )

    # ── Analysis 1: Fourier ──────────────────────────────────────────────────
    print("=== Analysis 1: Fourier Feature Analysis ===")
    fourier_per_model = {}   # mkey -> {top5_frequencies, top5_powers, concentration_ratio}
    spectra = {}             # mkey -> spectrum np.ndarray

    for task, seed in all_specs:
        mkey = f"{task}_seed{seed}"
        model = load_model(task, seed)
        spec = fourier_power_spectrum(model)
        spectra[mkey] = spec
        top_idx, top_pow = top_k_freqs(spec, TOP_K_FREQ)
        conc = concentration_ratio(spec, k=3)
        fourier_per_model[mkey] = {
            "top5_frequencies": [int(i) for i in top_idx],
            "top5_powers": top_pow,
            "concentration_ratio": conc,
        }
        print(f"Processing {mkey}... top-5 frequencies: {top_idx}")
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    addition_mean_spectrum = np.mean(
        [spectra[f"addition_seed{s}"] for s in ADDITION_GROKKED_SEEDS], axis=0
    )
    add_mean_top5_idx, _ = top_k_freqs(addition_mean_spectrum, TOP_K_FREQ)
    sub5_top5_idx = fourier_per_model[SUB5_KEY]["top5_frequencies"]
    shared_freqs = sorted(set(add_mean_top5_idx) & set(sub5_top5_idx))
    freq_jaccard = jaccard(set(add_mean_top5_idx), set(sub5_top5_idx))

    print(f"\nAddition mean top-5 frequencies: {add_mean_top5_idx}")
    print(f"Subtraction seed5 top-5 frequencies: {sub5_top5_idx}")
    print(f"Frequency overlap Jaccard: {freq_jaccard:.3f}")
    print(f"Shared frequencies: {shared_freqs}\n")

    fourier_out = {
        **{mk: v for mk, v in fourier_per_model.items()},
        "addition_mean_spectrum": addition_mean_spectrum.tolist(),
        "subtraction_seed5_spectrum": spectra[SUB5_KEY].tolist(),
        "frequency_overlap_jaccard": freq_jaccard,
        "shared_top5_frequencies": [int(k) for k in shared_freqs],
    }
    with open(os.path.join(METRICS_DIR, "fourier_analysis.json"), "w") as f:
        json.dump(fourier_out, f, indent=2)

    # ── Plot 1: Fourier spectra ──────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8))
    x = np.arange(N_FREQ)

    for s in ADDITION_GROKKED_SEEDS:
        ax_top.plot(x, spectra[f"addition_seed{s}"], color="lightsteelblue",
                    alpha=0.7, linewidth=1)
    ax_top.plot(x, addition_mean_spectrum, color="darkblue", linewidth=2.5,
                label="Addition mean")
    for k in add_mean_top5_idx[:3]:
        ax_top.axvline(k, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax_top.annotate(f"k={k}", xy=(k, addition_mean_spectrum[k]),
                        xytext=(k + 0.5, addition_mean_spectrum[k]),
                        fontsize=9, color="red")
    ax_top.set_yscale("log")
    ax_top.set_ylabel("Power (log scale)")
    ax_top.set_title("Fourier Power Spectrum - Embedding Matrix (Addition)")
    ax_top.legend(fontsize=9)

    ax_bot.plot(x, addition_mean_spectrum, color="darkblue", linewidth=2.5,
                label="Addition mean")
    ax_bot.plot(x, spectra[SUB5_KEY], color="green", linewidth=2.5,
                label="Subtraction seed5 (grokked)")
    ax_bot.plot(x, spectra["subtraction_seed2"], color="darkorange",
                linestyle="dashed", linewidth=1.8, label="Subtraction seed2 (near-miss)")
    ax_bot.plot(x, spectra["subtraction_seed4"], color="orange", alpha=0.6,
                linestyle="dashed", linewidth=1.8, label="Subtraction seed4 (near-miss)")
    for k in shared_freqs[:3]:
        ax_bot.axvline(k, color="purple", linestyle=":", linewidth=1.5, alpha=0.7)
        ax_bot.annotate(f"shared k={k}", xy=(k, max(addition_mean_spectrum[k], spectra[SUB5_KEY][k])),
                        fontsize=8, color="purple")
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel("Frequency index k")
    ax_bot.set_ylabel("Power (log scale)")
    ax_bot.set_title("Fourier Power Spectrum - Addition vs Subtraction")
    ax_bot.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fourier_spectra_comparison.png"), dpi=150)
    plt.close(fig)

    # ── Analysis 2: Attention asymmetry ──────────────────────────────────────
    print("=== Analysis 2: Attention Asymmetry ===")
    per_model_attn = {}        # mkey -> from_b_attn np.ndarray (4, 3)
    asymmetry_scores = {}      # mkey -> np.ndarray (4,)

    sub_loader = load_test_loader("subtraction")
    add_loader = load_test_loader("addition")

    for task, seed in all_specs:
        mkey = f"{task}_seed{seed}"
        model = load_model(task, seed)
        loader = add_loader if task == "addition" else sub_loader
        from_b = per_head_from_b_attention(model, loader)
        per_model_attn[mkey] = from_b
        asym = asymmetry_per_head(from_b)
        asymmetry_scores[mkey] = asym
        print(f"Processing {mkey}... Head 2 asymmetry: {asym[2]:.3f}")
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate addition stats per head
    add_asym_stack = np.stack([asymmetry_scores[f"addition_seed{s}"] for s in ADDITION_GROKKED_SEEDS], axis=0)
    add_means = add_asym_stack.mean(axis=0)
    add_stds  = add_asym_stack.std(axis=0)

    # Z-scores for Head 2
    h2_add_mean = float(add_means[2])
    h2_add_std  = float(add_stds[2])
    def z_for(mkey: str) -> float:
        if h2_add_std < 1e-9:
            return 0.0
        return float((asymmetry_scores[mkey][2] - h2_add_mean) / h2_add_std)

    z_sub5  = z_for(SUB5_KEY)
    z_sub2  = z_for("subtraction_seed2")
    z_sub4  = z_for("subtraction_seed4")

    print(f"\nAddition Head 2 asymmetry: {h2_add_mean:.3f} +/- {h2_add_std:.3f}")
    print(f"Subtraction seed5 Head 2 asymmetry: {asymmetry_scores[SUB5_KEY][2]:.3f} (z-score: {z_sub5:.2f})")
    print(f"Subtraction seed2 Head 2 asymmetry: {asymmetry_scores['subtraction_seed2'][2]:.3f} (z-score: {z_sub2:.2f})")
    print(f"Subtraction seed4 Head 2 asymmetry: {asymmetry_scores['subtraction_seed4'][2]:.3f} (z-score: {z_sub4:.2f})\n")

    asym_out = {
        "per_model": {
            mk: {f"head{h}": float(asymmetry_scores[mk][h]) for h in range(N_HEADS)}
            for mk in asymmetry_scores
        },
        "addition_summary": {
            f"head{h}": {"mean": float(add_means[h]), "std": float(add_stds[h])}
            for h in range(N_HEADS)
        },
        SUB5_KEY: {f"head{h}": float(asymmetry_scores[SUB5_KEY][h]) for h in range(N_HEADS)},
        "subtraction_seed2": {f"head{h}": float(asymmetry_scores['subtraction_seed2'][h]) for h in range(N_HEADS)},
        "subtraction_seed4": {f"head{h}": float(asymmetry_scores['subtraction_seed4'][h]) for h in range(N_HEADS)},
        "head2_z_scores": {
            SUB5_KEY: z_sub5,
            "subtraction_seed2": z_sub2,
            "subtraction_seed4": z_sub4,
        },
    }
    with open(os.path.join(METRICS_DIR, "attention_asymmetry.json"), "w") as f:
        json.dump(asym_out, f, indent=2)

    # ── Plot 2: Attention asymmetry ──────────────────────────────────────────
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
    x_heads = np.arange(N_HEADS)

    # Left panel: per-head asymmetry
    for s in ADDITION_GROKKED_SEEDS:
        ax_l.scatter(x_heads, asymmetry_scores[f"addition_seed{s}"],
                     color="lightsteelblue", alpha=0.6, s=30, zorder=2)
    ax_l.errorbar(x_heads, add_means, yerr=add_stds, fmt="o", color="darkblue",
                  markersize=8, capsize=5, label="Addition mean ± std", zorder=3)
    ax_l.scatter(x_heads, asymmetry_scores[SUB5_KEY], color="green", s=80, marker="D",
                 label="Subtraction seed5 (grokked)", zorder=4)
    ax_l.scatter(x_heads, asymmetry_scores["subtraction_seed2"], color="darkorange",
                 s=70, marker="s", label="Subtraction seed2 (near-miss)", zorder=4)
    ax_l.scatter(x_heads, asymmetry_scores["subtraction_seed4"], color="orange",
                 alpha=0.7, s=70, marker="s", label="Subtraction seed4 (near-miss)", zorder=4)
    ax_l.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax_l.set_xticks(x_heads)
    ax_l.set_xticklabels([f"Head {h}" for h in range(N_HEADS)])
    ax_l.set_ylabel("Asymmetry (attn_to_a - attn_to_b at from_b)")
    ax_l.set_title("Attention Asymmetry (from pos_b: to_a - to_b)")
    ax_l.legend(fontsize=8, loc="best")

    # Right panel: Head 2 distribution
    h2_add_vals = add_asym_stack[:, 2]
    ax_r.hist(h2_add_vals, bins=8, color="steelblue", alpha=0.7,
              edgecolor="black", label="Addition seeds (n=9)")
    ax_r.axvline(asymmetry_scores[SUB5_KEY][2], color="green", linewidth=2.5,
                 label=f"Sub seed5 (z={z_sub5:.2f})")
    ax_r.axvline(asymmetry_scores["subtraction_seed2"][2], color="darkorange",
                 linewidth=2, linestyle="--", label=f"Sub seed2 (z={z_sub2:.2f})")
    ax_r.axvline(asymmetry_scores["subtraction_seed4"][2], color="orange",
                 alpha=0.7, linewidth=2, linestyle="--", label=f"Sub seed4 (z={z_sub4:.2f})")
    ax_r.set_xlabel("Head 2 asymmetry")
    ax_r.set_ylabel("Count")
    ax_r.set_title("Head 2 Asymmetry Distribution")
    ax_r.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "attention_asymmetry.png"), dpi=150)
    plt.close(fig)

    # ── Analysis 3: Causal scrubbing ─────────────────────────────────────────
    print("=== Analysis 3: Causal Scrubbing ===")

    # Subtraction seed 5 baseline & forced symmetric
    sub5_model = load_model("subtraction", 5)
    sub5_baseline = evaluate_baseline(sub5_model, sub_loader)
    print(f"Subtraction seed5 baseline acc: {sub5_baseline:.4f}")

    sym_hook = make_force_pattern_hook(torch.tensor([0.5, 0.0, 0.5]))
    sub5_sym = evaluate_with_pattern_hook(sub5_model, sub_loader, sym_hook)
    print(f"Subtraction seed5 symmetric attention acc: {sub5_sym:.4f}")
    sub5_drop = sub5_baseline - sub5_sym
    print(f"Accuracy drop: {sub5_drop:.4f}")

    # Null intervention: replace with the head's own mean from-b pattern
    sub5_mean_from_b = torch.tensor(per_model_attn[SUB5_KEY][2])  # (3,)
    null_hook = make_force_pattern_hook(sub5_mean_from_b)
    sub5_null = evaluate_with_pattern_hook(sub5_model, sub_loader, null_hook)
    sub5_null_drop = sub5_baseline - sub5_null
    del sub5_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Addition control
    addition_control = {}
    drops = []
    for s in ADDITION_GROKKED_SEEDS:
        model = load_model("addition", s)
        baseline = evaluate_baseline(model, add_loader)
        sym_acc = evaluate_with_pattern_hook(model, add_loader, sym_hook)
        drop = baseline - sym_acc
        addition_control[f"seed{s}"] = {
            "baseline_acc": baseline,
            "symmetric_attention_acc": sym_acc,
            "accuracy_drop": drop,
        }
        drops.append(drop)
        print(f"  addition_seed{s}: baseline={baseline:.4f} sym={sym_acc:.4f} drop={drop:.4f}")
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    add_mean_drop = float(np.mean(drops))
    add_std_drop  = float(np.std(drops))
    print(f"\nAddition mean accuracy drop: {add_mean_drop:.4f} +/- {add_std_drop:.4f}")

    if sub5_drop > 3 * add_mean_drop:
        interp = "asymmetry_causally_necessary"
    elif sub5_drop > add_mean_drop:
        interp = "asymmetry_partially_causal"
    else:
        interp = "asymmetry_not_causally_necessary"
    print(f"Interpretation: {interp}")

    scrubbing_out = {
        SUB5_KEY: {
            "baseline_acc": sub5_baseline,
            "symmetric_attention_acc": sub5_sym,
            "accuracy_drop": sub5_drop,
            "null_intervention_acc": sub5_null,
            "null_intervention_drop": sub5_null_drop,
        },
        "addition_control": addition_control,
        "addition_control_mean_drop": add_mean_drop,
        "addition_control_std_drop": add_std_drop,
        "interpretation": interp,
    }
    with open(os.path.join(METRICS_DIR, "causal_scrubbing.json"), "w") as f:
        json.dump(scrubbing_out, f, indent=2)

    # ── Plot 3: Causal scrubbing ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    bars = ax.bar(x, [sub5_drop, add_mean_drop],
                  yerr=[0, add_std_drop], capsize=8,
                  color=["green", "steelblue"],
                  edgecolor="black")
    ax.axhline(add_mean_drop, color="steelblue", linestyle="--", alpha=0.6,
               label=f"Addition mean drop ({add_mean_drop:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(["Subtraction (seed5)", "Addition mean (9 seeds)"])
    ax.set_ylabel("Accuracy drop from forced symmetric attention")
    ax.set_title("Accuracy Drop from Forced Symmetric Head 2 Attention")
    ax.annotate(interp.replace("_", " "),
                xy=(0, sub5_drop), xytext=(0, sub5_drop + 0.02),
                ha="center", fontsize=9, color="darkgreen", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "causal_scrubbing.png"), dpi=150)
    plt.close(fig)

    print("\nSaved 3 JSON files and 3 plots.")


if __name__ == "__main__":
    main()
