import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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

# ── Architecture constants ────────────────────────────────────────────────────
ARCH = dict(
    n_layers=1, d_model=128, n_heads=4, d_head=32,
    d_mlp=None, act_fn=None, normalization_type=None,
    d_vocab=115, d_vocab_out=115, n_ctx=3, attn_only=True,
)
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_CHANCE = 1 / 113

# ── Checkpoints to load ───────────────────────────────────────────────────────
CHECKPOINTS = [
    # Grokked addition models
    *[(f"addition_seed{s}_final.pt", "addition", s, "grokked")
      for s in [0, 1, 2, 3, 4, 6, 7, 9, 10]],
    # Grokked subtraction
    ("subtraction_seed5_final.pt",  "subtraction", 5,  "grokked"),
    # Near-miss subtraction
    ("subtraction_seed2_final.pt",  "subtraction", 2,  "near-miss"),
    ("subtraction_seed4_final.pt",  "subtraction", 4,  "near-miss"),
    # Failed/low subtraction
    ("subtraction_seed0_final.pt",  "subtraction", 0,  "failed"),
    ("subtraction_seed9_final.pt",  "subtraction", 9,  "partial"),
]

STATE_KEYS = ["embed_pos0", "embed_pos1", "embed_pos2",
              "resid_pos0", "resid_pos1", "resid_pos2"]
STATE_LABELS = ["embed_a", "embed_=", "embed_b",
                "resid_a", "resid_=", "resid_b"]


def build_model(seed: int) -> HookedTransformer:
    cfg = HookedTransformerConfig(**ARCH, seed=seed)
    return HookedTransformer(cfg)


def load_test_data(task: str) -> DataLoader:
    d = torch.load(os.path.join(DATA_DIR, f"{task}_test.pt"), weights_only=True)
    a, b, label = d["a"], d["b"], d["label"]
    sep = torch.full_like(a, 113)
    inputs = torch.stack([a, sep, b], dim=1)
    return DataLoader(TensorDataset(inputs, label), batch_size=BATCH_SIZE, shuffle=False)


@torch.no_grad()
def run_logit_lens(model: HookedTransformer, loader: DataLoader) -> dict:
    """
    Returns per-state lens_acc and mean_rank across the full test set.
    States: embed and resid (embed+attn_out) at each of the 3 positions.
    """
    W_U = model.W_U  # (d_model, d_vocab)
    b_U = model.b_U  # (d_vocab,)

    # Accumulators: correct-answer prob, argmax hits, and rank per state
    sum_prob    = {k: 0.0 for k in STATE_KEYS}
    sum_argmax  = {k: 0.0 for k in STATE_KEYS}
    sum_rank    = {k: 0.0 for k in STATE_KEYS}
    n_total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        _, cache = model.run_with_cache(inputs)

        embed    = cache["hook_embed"]
        if "hook_pos_embed" in cache:
            embed = embed + cache["hook_pos_embed"]               # (B, 3, d)
        attn_out = cache["blocks.0.hook_attn_out"]                # (B, 3, d)
        resid    = embed + attn_out                               # (B, 3, d)

        states = {
            "embed_pos0": embed[:, 0, :],
            "embed_pos1": embed[:, 1, :],
            "embed_pos2": embed[:, 2, :],
            "resid_pos0": resid[:, 0, :],
            "resid_pos1": resid[:, 1, :],
            "resid_pos2": resid[:, 2, :],
        }

        B = inputs.size(0)
        for key, r in states.items():
            logits = r @ W_U + b_U            # (B, d_vocab)
            probs  = F.softmax(logits, dim=-1)

            # Mean probability assigned to the correct answer
            correct_prob = probs[torch.arange(B), labels]        # (B,)
            sum_prob[key] += correct_prob.sum().item()

            # Argmax accuracy (used only for sanity check at resid_pos2)
            sum_argmax[key] += (logits.argmax(dim=-1) == labels).sum().item()

            # Rank of correct answer (rank 1 = highest probability)
            ranks = (probs > correct_prob.unsqueeze(1)).sum(dim=1) + 1  # (B,)
            sum_rank[key] += ranks.float().sum().item()

        n_total += B

    return {
        key: {
            "lens_acc":   sum_prob[key]   / n_total,
            "argmax_acc": sum_argmax[key] / n_total,
            "mean_rank":  sum_rank[key]   / n_total,
        }
        for key in STATE_KEYS
    }


def model_key(task: str, seed: int) -> str:
    return f"{task}_seed{seed}"


def main():
    print(f"Device: {DEVICE}")
    results = {}
    addition_resid_b_profiles = []  # collect lens_acc at resid_pos2 for grokked addition

    for fname, task, seed, role in CHECKPOINTS:
        ckpt_path = os.path.join(CKPT_DIR, fname)
        if not os.path.exists(ckpt_path):
            print(f"MISSING: {fname} — skipping")
            continue

        ckpt = torch.load(ckpt_path, weights_only=False)
        reported_test_acc = ckpt.get("test_acc", float("nan"))
        mkey = model_key(task, seed)
        print(f"\nProcessing {mkey} (role={role}, reported_test_acc={reported_test_acc:.4f})")

        model = build_model(seed)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE).eval()

        loader = load_test_data(task)
        lens = run_logit_lens(model, loader)

        # Sanity check: resid_pos2 argmax accuracy should match reported test accuracy
        resid_b_argmax = lens["resid_pos2"]["argmax_acc"]
        delta = abs(resid_b_argmax - reported_test_acc)
        if delta > 0.01:
            print(
                f"  WARNING: [{mkey}] resid_b argmax_acc={resid_b_argmax:.4f} differs from "
                f"reported test_acc={reported_test_acc:.4f} by {delta:.4f}"
            )

        # Embedding-before-attention flags
        for pos_idx, pos_label in [(0, "pos_a"), (1, "pos_="), (2, "pos_b")]:
            key = f"embed_pos{pos_idx}"
            if lens[key]["lens_acc"] > 0.05:
                print(
                    f"  WARNING: [{mkey}] answer encoded before attention at {pos_label} "
                    f"(lens_acc={lens[key]['lens_acc']:.4f}) — unexpected, may indicate embedding leakage"
                )

        # Separator aggregation flag
        if lens["resid_pos1"]["lens_acc"] > 0.30:
            print(
                f"  NOTE: [{mkey}] separator position carries significant answer signal "
                f"(lens_acc={lens['resid_pos1']['lens_acc']:.3f}) — separator may be aggregating"
            )

        # Strip argmax_acc from saved output — only lens_acc and mean_rank are needed
        entry = {
            k: {"lens_acc": v["lens_acc"], "mean_rank": v["mean_rank"]}
            for k, v in lens.items()
        }
        entry.update({"task": task, "seed": seed,
                      "status": ckpt.get("status", role),
                      "final_test_acc": reported_test_acc})
        results[mkey] = entry

        if task == "addition" and role == "grokked":
            addition_resid_b_profiles.append(
                [lens[k]["lens_acc"] for k in STATE_KEYS]
            )

    # ── Addition vs subtraction_seed5 comparison ──────────────────────────────
    if addition_resid_b_profiles and "subtraction_seed5" in results:
        add_mean = np.mean(addition_resid_b_profiles, axis=0)
        sub5_profile = [results["subtraction_seed5"][k]["lens_acc"] for k in STATE_KEYS]
        # Compare at resid_b (last state)
        resid_indices = [STATE_KEYS.index(k) for k in
                         ["resid_pos0", "resid_pos1", "resid_pos2"]]
        diff = np.mean([abs(sub5_profile[i] - add_mean[i])
                        for i in resid_indices])
        if diff < 0.05:
            print(
                "\nMATCH: grokked subtraction resid profile closely matches addition mean "
                "— consistent with shared circuit"
            )
        elif diff > 0.15:
            print(
                "\nDIVERGE: grokked subtraction resid profile diverges from addition mean "
                "— consistent with different circuit"
            )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = os.path.join(METRICS_DIR, "logit_lens_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # ── Console summary table ─────────────────────────────────────────────────
    print("\n=== Logit Lens Summary ===")
    header = f"{'Model':<24}" + "".join(f"{lbl:>9}" for lbl in STATE_LABELS)
    print(header)
    print("-" * (24 + 9 * 6))
    for mkey, entry in sorted(results.items()):
        row = f"{mkey:<24}"
        for k in STATE_KEYS:
            row += f"{entry[k]['lens_acc']:>9.3f}"
        print(row)

    # ── Plot 1: Addition grokked seeds ────────────────────────────────────────
    add_entries = [
        (mkey, entry) for mkey, entry in results.items()
        if entry["task"] == "addition" and entry["status"] == "grokked"
    ]
    if add_entries:
        fig, ax = plt.subplots(figsize=(8, 5))
        profiles = []
        for mkey, entry in add_entries:
            profile = [entry[k]["lens_acc"] for k in STATE_KEYS]
            profiles.append(profile)
            ax.plot(STATE_LABELS, profile, color="lightsteelblue", linewidth=1, alpha=0.7)

        mean_profile = np.mean(profiles, axis=0)
        ax.plot(STATE_LABELS, mean_profile, color="darkblue", linewidth=2.5, label="Mean")
        ax.axhline(RANDOM_CHANCE, color="red", linestyle="--", linewidth=1,
                   label=f"Random chance ({RANDOM_CHANCE:.3f})")
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("Lens accuracy")
        ax.set_title(f"Logit Lens — Modular Addition ({len(add_entries)} grokked seeds)")
        ax.legend(fontsize=9)
        plt.xticks(rotation=30)
        plt.tight_layout()
        p1 = os.path.join(PLOTS_DIR, "logit_lens_addition.png")
        plt.savefig(p1, dpi=150)
        plt.close(fig)
        print(f"Saved {p1}")
    else:
        mean_profile = None

    # ── Plot 2: Subtraction comparison ───────────────────────────────────────
    sub_plot_specs = [
        ("subtraction_seed5",  "grokked (seed 5)",          "green",       "solid",  1.0),
        ("subtraction_seed2",  "near-miss seed 2 (0.9507)", "darkorange",  "dashed", 1.0),
        ("subtraction_seed4",  "near-miss seed 4 (0.9247)", "orange",      "dashed", 0.6),
        ("subtraction_seed0",  "failed seed 0 (0.3613)",    "red",         "dashed", 1.0),
        ("subtraction_seed9",  "partial seed 9 (0.8398)",   "lightcoral",  "dashed", 0.7),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    if mean_profile is not None:
        ax.plot(STATE_LABELS, mean_profile, color="darkblue", linewidth=2.5,
                linestyle="solid", label="Addition mean (grokked)")

    for mkey, label, color, ls, alpha in sub_plot_specs:
        if mkey not in results:
            continue
        profile = [results[mkey][k]["lens_acc"] for k in STATE_KEYS]
        ax.plot(STATE_LABELS, profile, color=color, linewidth=1.8,
                linestyle=ls, alpha=alpha, label=label)

    ax.axhline(RANDOM_CHANCE, color="grey", linestyle="--", linewidth=1,
               label=f"Random chance ({RANDOM_CHANCE:.3f})")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Lens accuracy")
    ax.set_title("Logit Lens — Subtraction vs Addition Mean Profile")
    ax.legend(fontsize=8, loc="upper left")
    plt.xticks(rotation=30)
    plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, "logit_lens_subtraction_comparison.png")
    plt.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved {p2}")


if __name__ == "__main__":
    main()
