import csv
import os
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

# ── Hyperparameters ───────────────────────────────────────────────────────────
P = 113
VOCAB_SIZE = 115
SEQ_LEN = 3
D_MODEL = 128
N_HEADS = 4
D_HEAD = 32
BATCH_SIZE = 512
MAX_EPOCHS = 5000
LR = 1e-3
WEIGHT_DECAY = 0.5
BETAS = (0.9, 0.98)
GROK_THRESHOLD = 0.95
GROK_SUSTAIN = 50       # consecutive epochs test_acc must stay >= GROK_THRESHOLD
INIT_CHECK_EPOCH = 500  # early-failure cutoff
INIT_TRAIN_THRESHOLD = 0.50
PLATEAU_WINDOW = 500    # epochs per window for plateau detection
PLATEAU_MIN_DELTA = 0.02  # minimum train_acc improvement between windows to not plateau
VALID_RUNS_TARGET = 10
TASKS = ["addition", "subtraction"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Status constants
STATUS_GROKKED = "grokked"
STATUS_PLATEAUED = "learned-but-plateaued"
STATUS_FAILED = "failed-to-learn"
STATUS_INIT_FAIL = "initialization-failure"


def set_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_model(seed: int):
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_mlp=None,
        act_fn=None,
        normalization_type=None,
        d_vocab=VOCAB_SIZE,
        d_vocab_out=VOCAB_SIZE,
        n_ctx=SEQ_LEN,
        attn_only=True,
        seed=seed,
    )
    return HookedTransformer(cfg)


def load_task_data(task: str):
    def _load(split: str):
        d = torch.load(os.path.join(DATA_DIR, f"{task}_{split}.pt"), weights_only=True)
        a, b, label = d["a"], d["b"], d["label"]
        sep = torch.full_like(a, 113)
        inputs = torch.stack([a, sep, b], dim=1)  # (N, 3)
        return inputs, label

    train_inputs, train_labels = _load("train")
    test_inputs, test_labels = _load("test")
    return (
        DataLoader(TensorDataset(train_inputs, train_labels), batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(TensorDataset(test_inputs, test_labels), batch_size=BATCH_SIZE, shuffle=False),
    )


@torch.no_grad()
def evaluate(model: HookedTransformer, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        logits = model(inputs)[:, 2, :]
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, total_correct / total


def train_run(task: str, seed: int, log_rows: list, results_dir: str) -> dict | None:
    """
    Returns a result dict for valid runs, or None for initialization failures.
    Appends per-epoch dicts to log_rows in-place.
    """
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    set_seeds(seed)
    model = build_model(seed).to(DEVICE)
    train_loader, test_loader = load_task_data(task)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS
    )

    # Grokking detection state
    grokking_epoch = None
    candidate_epoch = None  # first epoch test_acc crossed GROK_THRESHOLD
    consec_above = 0        # consecutive epochs at or above GROK_THRESHOLD
    grokked_ckpt_saved = False

    # Pre-grokking checkpoint tracking — last epoch (and weights) where test_acc < 0.5
    last_below_half_epoch = None
    last_below_half_state = None
    epoch1_state = None
    epoch1_train_acc = None
    epoch1_test_acc = None

    train_acc_history = []
    test_acc_history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(inputs)[:, 2, :]
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels)
            epoch_correct += (logits.argmax(dim=-1) == labels).sum().item()
            epoch_total += len(labels)

        train_acc = epoch_correct / epoch_total
        _, test_acc = evaluate(model, test_loader)

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        # Epoch-1 fallback snapshot for fast-grokking seeds
        if epoch == 1:
            state = {n: p.detach().cpu().clone()
                     for n, p in model.named_parameters()
                     if p.requires_grad}
            has_nan = any(torch.isnan(v).any() for v in state.values())
            epoch1_state = None if has_nan else state
            if has_nan:
                print(f"  WARNING: NaN in epoch 1 weights for "
                      f"[{task} | seed {seed}] — epoch1 fallback unavailable")
            epoch1_train_acc = train_acc
            epoch1_test_acc = test_acc

        # Rolling pre-grokking snapshot — last memorisation-phase state
        if test_acc < 0.5:
            state = {n: p.detach().cpu().clone()
                     for n, p in model.named_parameters()
                     if p.requires_grad}
            has_nan = any(torch.isnan(v).any() for v in state.values())
            if not has_nan:
                last_below_half_epoch = epoch
                last_below_half_state = state

        # Initialization failure check
        if epoch == INIT_CHECK_EPOCH and train_acc < INIT_TRAIN_THRESHOLD:
            print(
                f"[{task} | seed {seed}] INITIALIZATION FAILURE at epoch {INIT_CHECK_EPOCH} | "
                f"train_acc={train_acc:.4f} — skipping"
            )
            log_rows.append({
                "task": task, "seed": seed, "epoch": epoch,
                "train_acc": train_acc, "test_acc": test_acc,
                "status": STATUS_INIT_FAIL,
            })
            return None

        # Two-part grokking detection
        if grokking_epoch is None:
            if test_acc >= GROK_THRESHOLD:
                if candidate_epoch is None:
                    candidate_epoch = epoch
                consec_above += 1
                if consec_above >= GROK_SUSTAIN:
                    grokking_epoch = candidate_epoch
                    if not grokked_ckpt_saved:
                        actual_grokked_epoch = candidate_epoch + GROK_SUSTAIN - 1

                        # Resolve pre-grokking checkpoint: prefer last sub-0.5 state,
                        # fall back to epoch 1 for fast-grokking seeds.
                        pre_state_to_save = last_below_half_state
                        pre_epoch_to_save = last_below_half_epoch
                        pre_train_acc = (train_acc_history[last_below_half_epoch - 1]
                                         if last_below_half_epoch is not None else None)
                        pre_test_acc = (test_acc_history[last_below_half_epoch - 1]
                                        if last_below_half_epoch is not None else None)
                        if pre_state_to_save is None and epoch1_state is not None:
                            pre_state_to_save = epoch1_state
                            pre_epoch_to_save = 1
                            pre_train_acc = epoch1_train_acc
                            pre_test_acc = epoch1_test_acc
                            print(
                                f"  NOTE: [{task} | seed {seed}] no sub-0.5 epoch found "
                                f"— using epoch 1 state as pre_grokking fallback"
                            )

                        # Compute weight L2 diff safely
                        if pre_state_to_save is not None:
                            try:
                                # Get names of trainable parameters only
                                trained_names = {n for n, p in model.named_parameters()
                                                 if p.requires_grad}

                                pre_params = torch.cat([
                                    v.float().cpu().flatten()
                                    for k, v in pre_state_to_save.items()
                                    if k in trained_names
                                ])
                                grokked_params = torch.cat([
                                    p.detach().float().cpu().flatten()
                                    for n, p in model.named_parameters()
                                    if n in trained_names and p.requires_grad
                                ])
                                pre_has_nan = torch.isnan(pre_params).any().item()
                                grokked_has_nan = torch.isnan(grokked_params).any().item()
                                if pre_has_nan or grokked_has_nan:
                                    print(f"  WARNING: NaN in weight tensors for "
                                          f"[{task} | seed {seed}] — "
                                          f"pre_nan={pre_has_nan} grokked_nan={grokked_has_nan}")
                                    weight_diff = -1.0
                                else:
                                    weight_diff = (pre_params - grokked_params).norm().item()
                                if weight_diff != -1.0 and torch.isnan(torch.tensor(weight_diff)):
                                    print(f"  WARNING: L2 diff is NaN for "
                                          f"[{task} | seed {seed}] — skipping")
                                    weight_diff = -1.0
                                else:
                                    print(f"  Weight L2 diff (pre_grokking -> grokked): "
                                          f"{weight_diff:.2f}")
                            except Exception as e:
                                print(f"  WARNING: L2 diff computation failed: {e}")
                                weight_diff = -1.0
                        else:
                            print(f"  WARNING: [{task} | seed {seed}] no pre-grokking "
                                  f"state found — test_acc never below 0.5 before grokking")
                            weight_diff = -1.0

                        # Save grokked checkpoint (includes weight_l2_diff_pre_grokked)
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "task": task,
                                "seed": seed,
                                "epoch": actual_grokked_epoch,
                                "train_acc": train_acc_history[actual_grokked_epoch - 1],
                                "test_acc": test_acc_history[actual_grokked_epoch - 1],
                                "grokking_epoch": grokking_epoch,
                                "status": STATUS_GROKKED,
                                "weight_l2_diff_pre_grokked": weight_diff,
                            },
                            os.path.join(ckpt_dir, f"{task}_seed{seed}_grokked.pt"),
                        )
                        grokked_ckpt_saved = True

                        # Save pre-grokking checkpoint (with fallback already resolved).
                        # NOTE: model_state_dict here only contains TRAINABLE parameters
                        # (non-trainable buffers omitted). When loading in Phase 6, use
                        # model.load_state_dict(ckpt["model_state_dict"], strict=False).
                        if pre_state_to_save is not None:
                            torch.save(
                                {
                                    "model_state_dict": pre_state_to_save,
                                    "task": task,
                                    "seed": seed,
                                    "epoch": pre_epoch_to_save,
                                    "train_acc": pre_train_acc,
                                    "test_acc": pre_test_acc,
                                    "grokking_epoch": grokking_epoch,
                                    "status": "pre_grokking",
                                },
                                os.path.join(ckpt_dir, f"{task}_seed{seed}_pre_grokking.pt"),
                            )
                            if 0 <= weight_diff < 1.0:
                                print(
                                    f"  WARNING: [{task} | seed {seed}] pre_grokking and grokked "
                                    f"checkpoints are suspiciously similar (L2={weight_diff:.4f})"
                                )
            else:
                candidate_epoch = None
                consec_above = 0

        # Plateau early stopping
        if grokking_epoch is None and epoch >= PLATEAU_WINDOW * 2:
            window_recent = max(train_acc_history[-PLATEAU_WINDOW:])
            window_prev = max(train_acc_history[-PLATEAU_WINDOW * 2:-PLATEAU_WINDOW])
            if window_recent - window_prev < PLATEAU_MIN_DELTA:
                print(
                    f"[{task} | seed {seed}] PLATEAU STOP at epoch {epoch} | "
                    f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
                )
                break

        # Determine current status for CSV (tentative until run ends)
        if grokking_epoch is not None:
            row_status = STATUS_GROKKED
        elif test_acc >= 0.5:
            row_status = STATUS_PLATEAUED
        else:
            row_status = STATUS_FAILED

        log_rows.append({
            "task": task, "seed": seed, "epoch": epoch,
            "train_acc": train_acc, "test_acc": test_acc,
            "status": row_status,
        })

        if epoch % 100 == 0:
            print(
                f"[{task} | seed {seed} | epoch {epoch:04d}] "
                f"train_acc={train_acc:.3f} test_acc={test_acc:.3f}"
            )

        # Stop once grokking is confirmed (candidate confirmed + buffer elapsed)
        if grokking_epoch is not None and consec_above >= GROK_SUSTAIN:
            break

    final_train_acc = train_acc_history[-1]
    final_test_acc = test_acc_history[-1]

    if grokking_epoch is not None:
        status = STATUS_GROKKED
    elif final_test_acc >= 0.5:
        status = STATUS_PLATEAUED
    else:
        status = STATUS_FAILED

    if grokking_epoch is not None:
        print(
            f"[{task} | seed {seed}] GROKKED at epoch {grokking_epoch} | "
            f"pre_grokking_epoch={last_below_half_epoch} | "
            f"final test_acc={final_test_acc:.4f}"
        )
    else:
        print(
            f"[{task} | seed {seed}] {status.upper()} at epoch {epoch} | "
            f"final test_acc={final_test_acc:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "task": task,
            "seed": seed,
            "epoch": epoch,
            "train_acc": final_train_acc,
            "test_acc": final_test_acc,
            "grokking_epoch": grokking_epoch,
            "status": status,
        },
        os.path.join(ckpt_dir, f"{task}_seed{seed}_final.pt"),
    )

    return {
        "task": task,
        "seed": seed,
        "status": status,
        "grokking_epoch": grokking_epoch,
        "final_test_acc": final_test_acc,
        "final_epoch": epoch,
    }


def print_summary(valid_results: list[dict], init_failures: dict[str, int]):
    print("\n=== Training Summary ===")
    print(f"{'Task':<14} {'Seed':<6} {'Status':<26} {'Grokking Epoch':<16} {'Final Test Acc'}")
    print(f"{'-'*13} {'-'*5} {'-'*25} {'-'*15} {'-'*14}")
    for r in sorted(valid_results, key=lambda x: (x["task"], x["seed"])):
        ge = str(r["grokking_epoch"]) if r["grokking_epoch"] is not None else "—"
        print(f"{r['task']:<14} {r['seed']:<6} {r['status']:<26} {ge:<16} {r['final_test_acc']:.4f}")

    print()
    for task in TASKS:
        runs = [r for r in valid_results if r["task"] == task]
        n_failures = init_failures.get(task, 0)
        grokked = [r for r in runs if r["status"] == STATUS_GROKKED]
        grok_epochs = [r["grokking_epoch"] for r in grokked]
        final_accs = [r["final_test_acc"] for r in runs]

        print(f"[{task}]")
        print(f"  Initialization failures : {n_failures}")
        print(f"  Grokking rate           : {len(grokked)}/{len(runs)} valid seeds")
        if grok_epochs:
            print(f"  Grokking epoch          : {np.mean(grok_epochs):.1f} ± {np.std(grok_epochs):.1f}")
        else:
            print(f"  Grokking epoch          : N/A")
        if final_accs:
            print(f"  Final test acc          : {np.mean(final_accs):.4f} ± {np.std(final_accs):.4f}")
        print(f"  Weight decay used       : {WEIGHT_DECAY}")


def save_csvs(log_rows: list[dict], valid_results: list[dict]):
    logs_path = os.path.join(RESULTS_DIR, "training_logs.csv")
    with open(logs_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "seed", "epoch", "train_acc", "test_acc", "status"])
        writer.writeheader()
        writer.writerows(log_rows)

    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "seed", "status", "grokking_epoch", "final_test_acc"])
        writer.writeheader()
        for r in sorted(valid_results, key=lambda x: (x["task"], x["seed"])):
            writer.writerow({
                "task": r["task"],
                "seed": r["seed"],
                "status": r["status"],
                "grokking_epoch": r["grokking_epoch"] if r["grokking_epoch"] is not None else "",
                "final_test_acc": r["final_test_acc"],
            })

    print(f"\nSaved {logs_path}")
    print(f"Saved {summary_path}")


def main():
    # Step 0: wipe and recreate only results/checkpoints/ —
    # preserve metrics/, plots/, training_logs.csv, summary.csv
    ckpt_dir = os.path.join(RESULTS_DIR, "checkpoints")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir)

    print(f"Device: {DEVICE}\n")

    all_valid_results = []
    init_failures = {task: 0 for task in TASKS}
    log_rows = []

    for task in TASKS:
        valid_count = 0
        seed = 0

        while valid_count < VALID_RUNS_TARGET:
            print(f"\n{'='*60}")
            print(f"Starting: {task} | seed {seed}  (valid so far: {valid_count}/{VALID_RUNS_TARGET})")
            print(f"{'='*60}")

            result = train_run(task, seed, log_rows, RESULTS_DIR)

            if result is None:
                init_failures[task] += 1
            else:
                all_valid_results.append(result)
                valid_count += 1

            seed += 1

    save_csvs(log_rows, all_valid_results)
    print_summary(all_valid_results, init_failures)


if __name__ == "__main__":
    main()
