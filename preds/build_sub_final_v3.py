"""Build a validation-tuned log-probability blend from saved CAIR fold outputs."""

import ast
import csv
import glob
import math
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "preds"
SUB_DIR = ROOT / "submissions"
MODEL_NAMES = ("debertaL", "robertaL", "qwen8b")
# Best 0.05-step weight combination on the shared original-row OOF set.
WEIGHTS = (0.15, 0.45, 0.40)
REFERENCE_WEIGHTS = (0.14, 0.43, 0.43)  # sub_final_v2 blend


def read_npy(path):
    """Read the little-endian float32/int64 NPY files saved by the notebooks."""
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"Not a NumPy array: {path}")
        version = tuple(stream.read(2))
        header_size_bytes = 2 if version == (1, 0) else 4
        header_size = int.from_bytes(stream.read(header_size_bytes), "little")
        header = ast.literal_eval(stream.read(header_size).decode("latin1").strip())
        raw = stream.read()

    formats = {"<f4": ("f", 4), "<i8": ("q", 8)}
    try:
        fmt, item_size = formats[header["descr"]]
    except KeyError as error:
        raise ValueError(f"Unsupported NPY dtype {header['descr']} in {path}") from error
    if len(raw) % item_size:
        raise ValueError(f"Invalid NPY payload size: {path}")
    values = struct.unpack("<" + str(len(raw) // item_size) + fmt, raw)
    return header["shape"], values


def log_probability(value):
    return math.log(max(1e-7, min(1.0, value)))


def accuracy(weights, models, targets, row_ids, class_count):
    correct = 0
    for row_id in row_ids:
        for name in MODEL_NAMES:
            if len(models[name]["oof"][row_id]) != class_count:
                raise ValueError(
                    f"Unexpected {name} OOF class width at row {row_id}: "
                    f"{len(models[name]['oof'][row_id])} vs {class_count}"
                )
        scores = [
            sum(
                weights[m] * log_probability(models[name]["oof"][row_id][c])
                for m, name in enumerate(MODEL_NAMES)
            )
            for c in range(class_count)
        ]
        correct += max(range(class_count), key=scores.__getitem__) == targets[row_id]
    return correct / len(row_ids)


def main():
    train_rows = []
    with (ROOT / "CAIRLAB.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) >= 4 and row[3].strip():
                train_rows.append(row)

    labels = sorted({row[2] for row in train_rows})
    label_to_index = {label: index for index, label in enumerate(labels)}
    y = [label_to_index[row[2]] for row in train_rows]
    original_rows = []
    seen_ids = set()
    for row in train_rows:
        original_rows.append(row[0] not in seen_ids)
        seen_ids.add(row[0])

    test_rows = list(csv.DictReader((ROOT / "test.csv").open(encoding="utf-8-sig", newline="")))
    models = {}
    for name in MODEL_NAMES:
        oof = [None] * len(train_rows)
        test_folds = []
        val_files = sorted(glob.glob(str(PRED_DIR / f"{name}_f*_val.npy")))
        if not val_files:
            raise FileNotFoundError(f"No validation predictions found for {name}")

        for val_file in val_files:
            fold = re.search(r"_f(\d+)_val\.npy$", val_file).group(1)
            val_shape, val_probs = read_npy(Path(val_file))
            index_shape, val_indices = read_npy(PRED_DIR / f"{name}_f{fold}_va_idx.npy")
            test_shape, test_probs = read_npy(PRED_DIR / f"{name}_f{fold}_test.npy")
            if val_shape != (index_shape[0], len(labels)):
                raise ValueError(f"Validation shape/index mismatch for {name} fold {fold}")
            if test_shape != (len(test_rows), len(labels)):
                raise ValueError(f"Test prediction shape mismatch for {name} fold {fold}")

            for position, row_index in enumerate(val_indices):
                start = position * len(labels)
                oof[row_index] = val_probs[start : start + len(labels)]
            test_folds.append([
                test_probs[i * len(labels) : (i + 1) * len(labels)]
                for i in range(len(test_rows))
            ])

        averaged_test = [
            [
                sum(fold[i][c] for fold in test_folds) / len(test_folds)
                for c in range(len(labels))
            ]
            for i in range(len(test_rows))
        ]
        models[name] = {"oof": oof, "test": averaged_test}

    common_originals = [
        index
        for index, is_original in enumerate(original_rows)
        if is_original and all(models[name]["oof"][index] is not None for name in MODEL_NAMES)
    ]
    if not common_originals:
        raise ValueError("No shared original-row OOF predictions found")

    current_score = accuracy(REFERENCE_WEIGHTS, models, y, common_originals, len(labels))
    tuned_score = accuracy(WEIGHTS, models, y, common_originals, len(labels))

    predictions = []
    for row_index, test_row in enumerate(test_rows):
        scores = [
            sum(
                WEIGHTS[m] * log_probability(models[name]["test"][row_index][c])
                for m, name in enumerate(MODEL_NAMES)
            )
            for c in range(len(labels))
        ]
        predictions.append(labels[max(range(len(labels)), key=scores.__getitem__)])

    output = SUB_DIR / "sub_final_v3.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ID", "Sentiment"))
        writer.writerows((row["ID"], prediction) for row, prediction in zip(test_rows, predictions))

    distribution = {label: predictions.count(label) for label in labels}
    print(f"Models: {', '.join(MODEL_NAMES)}")
    print(f"Weights: {dict(zip(MODEL_NAMES, WEIGHTS))}")
    print(f"Shared original-row OOF: {tuned_score:.5f} ({len(common_originals)} rows)")
    print(f"sub_final_v2 weights OOF: {current_score:.5f}")
    print(f"OOF difference: {tuned_score - current_score:+.5f}")
    print(f"Wrote {output.relative_to(ROOT)} ({len(predictions)} rows)")
    print(f"Prediction counts: {distribution}")


if __name__ == "__main__":
    main()
