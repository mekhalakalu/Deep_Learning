# CS 5173/6073 Deep Learning Midterm — Kaggle Submission Generator

This folder contains `make_submission.py`, the instructor-provided helper for producing a correctly formatted Kaggle submission for the five-class ECG classification exam.

## What it does

Given a trained **PyTorch model** and a CSV of predictions, `make_submission`:

1. Counts the model's **trainable parameters**.
2. Checks the prediction IDs against `sample_submission.csv`.
3. Validates that every prediction is an integer label from 0 through 4.
4. Writes the final `submission.csv` in the template's row order.

It **does not** train the model, make predictions, or put the actual model weights in the CSV. Submit the trained model and source code separately through Canvas, as required by the exam instructions.

## Required files

Put these files in your working directory:

| File | Purpose |
| --- | --- |
| `make_submission.py` | Instructor-provided submission generator. |
| `sample_submission.csv` | Official Kaggle submission template. |
| `predictions.csv` | Your model's predictions, with columns `Id,type`. |

You also need the trained PyTorch `model` object in your Python session. The full test set contains **21,892** records, so `predictions.csv` must contain one row for every test ID.

## Step 1 — Prepare predictions.csv

Export your model's predictions to a CSV containing **exactly** these column names, in this order:

```csv
Id,type
0,1
1,3
2,0
3,4
```

Each `Id` must match an ID in `sample_submission.csv`; IDs cannot be missing or repeated. The `type` column uses the following exam label encoding:

| Label | Heartbeat type |
| --- | --- |
| 0 | Fusion (F) |
| 1 | Normal (N) |
| 2 | Unknown (Q) |
| 3 | Supraventricular ectopic (SVEB) |
| 4 | Ventricular ectopic (VEB) |

Do **not** include `trainable_params` in `predictions.csv`; the function adds it automatically.

## Step 2 — Generate submission.csv

In the Python session where your trained model exists, run:

```python
from make_submission import make_submission

make_submission(
    model,
    predictions_csv="predictions.csv",
    template_csv="sample_submission.csv",
    output_csv="submission.csv",
)
```

`model` must be the trained PyTorch model that produced the predictions. The function returns the output file path and prints the number of predictions and trainable parameters.

The resulting `submission.csv` contains **exactly** these columns:

```csv
Id,type,trainable_params
0,1,12345
1,3,12345
2,0,12345
3,4,12345
```

The number `12345` above is only an example; your actual count is calculated from your model. The parameter count is identical on every row.

## How trainable parameters are counted

The helper uses this PyTorch expression:

```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

This counts each scalar element of every parameter with `requires_grad=True`. It excludes frozen parameters, optimizer state, estimated FLOPs, and the size of the saved model file. The model must have at least one trainable parameter. Use the trained model object, not a file path, as the function argument.

## File validation

The helper raises a `ValueError` instead of producing a submission when it finds missing/duplicate IDs, invalid class labels, an incorrect CSV header, or an empty input. The final rows are written in the same ID order as `sample_submission.csv`.

Before uploading, verify that `submission.csv` exists and upload **that generated file** to Kaggle. The `trainable_params` field is a *reported count*; it does not allow Kaggle to verify the contents of the model. The instructor may inspect your separately submitted model and code through Canvas.

## Exam rules

Use only the permitted training data, do not use pretrained model weights, complete the exam individually, and follow the course's academic integrity policy and submission deadline.
