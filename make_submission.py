"""Instructor-provided submission helper for the ECG midterm competition.

Input prediction CSV: Id,type
Output Kaggle CSV: Id,type,trainable_params

Use the trained PyTorch model that produced the predictions. Model weights
are not embedded in the CSV; only their trainable parameter count is reported.
"""

import csv
from pathlib import Path


SUBMISSION_COLUMNS = ["Id", "type", "trainable_params"]
PREDICTION_COLUMNS = ["Id", "type"]
ALLOWED_LABELS = set(range(5))


def count_trainable_params(model) -> int:
    """Count trainable scalar parameters in a PyTorch model."""
    if not callable(getattr(model, "parameters", None)):
        raise TypeError("model must be a PyTorch model with a parameters() method")

    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count <= 0:
        raise ValueError("model must have at least one trainable parameter")
    return int(count)


def _read_csv(path, expected_columns):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"{path}: expected columns {expected_columns}, "
                f"but found {reader.fieldnames}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: CSV has no data rows")

    for line, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}: malformed CSV row on line {line}")
    return rows


def make_submission(
    model,
    predictions_csv="predictions.csv",
    template_csv="sample_submission.csv",
    output_csv="submission.csv",
) -> str:
    """Add the model's trainable-parameter count to an Id,type CSV.

    Checks that every test Id appears exactly once, that prediction labels are
    integers in [0, 4], and that the output matches the Kaggle submission
    template. The output is ordered by the template's Id column.
    """
    params = count_trainable_params(model)
    template_rows = _read_csv(template_csv, SUBMISSION_COLUMNS)
    prediction_rows = _read_csv(predictions_csv, PREDICTION_COLUMNS)

    template_ids = [row["Id"].strip() for row in template_rows]
    if any(not id_value for id_value in template_ids):
        raise ValueError("sample_submission.csv contains an empty Id")
    if len(template_ids) != len(set(template_ids)):
        raise ValueError("sample_submission.csv contains duplicate Id values")

    predictions = {}
    for line, row in enumerate(prediction_rows, start=2):
        id_value = row["Id"].strip()
        if not id_value:
            raise ValueError(f"predictions.csv: empty Id on line {line}")
        if id_value in predictions:
            raise ValueError(f"predictions.csv: duplicate Id {id_value!r}")
        label_text = row["type"].strip()
        try:
            label = int(label_text)
        except ValueError as exc:
            raise ValueError(
                f"predictions.csv: type for Id {id_value!r} must be an integer 0–4"
            ) from exc
        if label not in ALLOWED_LABELS:
            raise ValueError(f"predictions.csv: invalid class {label} for Id {id_value!r}")
        predictions[id_value] = label

    expected_ids = set(template_ids)
    actual_ids = set(predictions)
    if expected_ids != actual_ids:
        missing = sorted(expected_ids - actual_ids)[:5]
        unexpected = sorted(actual_ids - expected_ids)[:5]
        raise ValueError(
            f"Prediction Ids do not match sample_submission.csv: "
            f"missing={missing}, unexpected={unexpected}"
        )

    output_path = Path(output_csv)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUBMISSION_COLUMNS)
        for id_value in template_ids:
            writer.writerow([id_value, predictions[id_value], params])

    print(f"Created {output_path} ({len(template_ids)} predictions, {params} trainable parameters)")
    return str(output_path)


# Example (execute in the same Python session after training your model):
# from make_submission import make_submission
# make_submission(model, "predictions.csv", "sample_submission.csv", "submission.csv")
