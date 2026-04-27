import json
import os
import numpy as np
import torch

P = 113
TRAIN_FRACTION = 0.5
SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def generate_split(p: int, train_fraction: float, seed: int):
    """Return train and test index arrays into the flattened (a, b) pair grid."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    n_total = len(all_pairs)  # p * p = 12769

    indices = np.arange(n_total)
    rng.shuffle(indices)

    n_train = round(n_total * train_fraction)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    return all_pairs, train_indices, test_indices


def build_tensors(pairs, indices, task: str, p: int):
    a_vals = []
    b_vals = []
    labels = []

    for idx in indices:
        a, b = pairs[idx]
        if task == "addition":
            label = (a + b) % p
        else:
            label = (a - b) % p
        a_vals.append(a)
        b_vals.append(b)
        labels.append(label)

    return {
        "a": torch.tensor(a_vals, dtype=torch.long),
        "b": torch.tensor(b_vals, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.long),
    }


def verify(add_train, add_test, sub_train, sub_test, p: int):
    n_total = p * p

    # 1. Total pairs = 12769
    assert len(add_train["a"]) + len(add_test["a"]) == n_total, \
        f"Addition total != {n_total}"
    assert len(sub_train["a"]) + len(sub_test["a"]) == n_total, \
        f"Subtraction total != {n_total}"

    # 2. Train + test sum to 12769 (already covered above)

    # 3. No (a, b) pair in both train and test
    def pair_set(split):
        return set(zip(split["a"].tolist(), split["b"].tolist()))

    add_train_pairs = pair_set(add_train)
    add_test_pairs = pair_set(add_test)
    overlap = add_train_pairs & add_test_pairs
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping pairs in addition splits"

    # 4–7. Coverage: every value of a and b in {0,...,p-1} appears in each split
    for split_name, split in [("add_train", add_train), ("add_test", add_test),
                               ("sub_train", sub_train), ("sub_test", sub_test)]:
        a_vals = set(split["a"].tolist())
        b_vals = set(split["b"].tolist())
        expected = set(range(p))
        assert a_vals == expected, f"{split_name}: missing a values {expected - a_vals}"
        assert b_vals == expected, f"{split_name}: missing b values {expected - b_vals}"

    # 8. All labels in {0,...,p-1}
    for split_name, split in [("add_train", add_train), ("add_test", add_test),
                               ("sub_train", sub_train), ("sub_test", sub_test)]:
        labels = split["label"]
        assert labels.min().item() >= 0 and labels.max().item() < p, \
            f"{split_name}: label out of range [0, {p-1}]"

    # 9. Addition labels correct
    for split in [add_train, add_test]:
        expected = (split["a"] + split["b"]) % p
        assert torch.all(split["label"] == expected), "Addition label mismatch"

    # 10. Subtraction labels correct
    for split in [sub_train, sub_test]:
        expected = (split["a"] - split["b"]) % p
        assert torch.all(split["label"] == expected), "Subtraction label mismatch"

    # 11. Addition and subtraction train splits use identical (a, b) pairs
    sub_train_pairs = pair_set(sub_train)
    assert add_train_pairs == sub_train_pairs, \
        "Addition and subtraction train splits have different (a, b) pairs"

    print("Split summary:")
    print(f"  Addition    — train: {len(add_train['a'])} pairs, test: {len(add_test['a'])} pairs")
    print(f"  Subtraction — train: {len(sub_train['a'])} pairs, test: {len(sub_test['a'])} pairs")
    print("  Overlap check: PASSED")
    print("  Label range check: PASSED")
    print("  Coverage check: PASSED")

    return len(add_train["a"]), len(add_test["a"])


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    all_pairs, train_indices, test_indices = generate_split(P, TRAIN_FRACTION, SEED)

    add_train = build_tensors(all_pairs, train_indices, "addition", P)
    add_test = build_tensors(all_pairs, test_indices, "addition", P)
    sub_train = build_tensors(all_pairs, train_indices, "subtraction", P)
    sub_test = build_tensors(all_pairs, test_indices, "subtraction", P)

    train_size, test_size = verify(add_train, add_test, sub_train, sub_test, P)

    torch.save(add_train, os.path.join(DATA_DIR, "addition_train.pt"))
    torch.save(add_test, os.path.join(DATA_DIR, "addition_test.pt"))
    torch.save(sub_train, os.path.join(DATA_DIR, "subtraction_train.pt"))
    torch.save(sub_test, os.path.join(DATA_DIR, "subtraction_test.pt"))

    split_info = {
        "p": P,
        "train_size_addition": train_size,
        "test_size_addition": test_size,
        "train_size_subtraction": train_size,
        "test_size_subtraction": test_size,
        "train_fraction": TRAIN_FRACTION,
        "seed": SEED,
        "split_strategy": "pair_level",
        "note": "Every value of a and b appears in both splits. No (a,b) pair appears in both.",
    }
    with open(os.path.join(DATA_DIR, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSaved to {os.path.abspath(DATA_DIR)}/")


if __name__ == "__main__":
    main()
