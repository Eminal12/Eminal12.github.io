# Importing the libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Loading the excel files produced by the extraction system.
def load_results(excel_path: str) -> pd.DataFrame:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Could not find: {excel_path}")
    return pd.read_excel(excel_path)

# Converting individual cells in the excel files into '0' for missing values and '1'
# for anything else. 
def cell_to_label(x) -> int:
    if pd.isna(x):
        return 0

    s = str(x).strip().lower()

    if s in {"not available", "n/a", ""}:
        return 0

    return 1

# Converting the results table into a single list of values so it can be evaluated.
def to_binary_availability(df: pd.DataFrame, metric_col_hint: str = "Unnamed: 0") -> np.ndarray:

    data = df.copy()

    if metric_col_hint in data.columns:
        data = data.drop(columns=[metric_col_hint])

    bin_df = data.applymap(cell_to_label)

    return bin_df.to_numpy().ravel()

# Creating a confusion matrix to show the true positves and false negatives of the system.
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix (Availability-Based)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# Creating an overall metrics performance chart.
def plot_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}

    fig, ax = plt.subplots()
    ax.bar(list(metrics.keys()), list(metrics.values()))
    ax.set_ylim(0, 1)
    ax.set_title("Overall Performance Metrics (Availability-Based)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return metrics

# Creating a correct vs incorrect chart.
def plot_correct_incorrect(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> tuple[int, int]:
    correct = int(np.sum(y_pred == y_true))
    incorrect = int(np.sum(y_pred != y_true))

    fig, ax = plt.subplots()
    ax.bar(["Correct", "Incorrect"], [correct, incorrect])
    ax.set_title("Correct vs Incorrect (Availability Labels)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return correct, incorrect

# Here, I am running the whole evaluation from end to end by loading excel, converting outputs to availability
# labels, assuming all existing metrics are correct (1) and saves all the graphs created.
def main() -> None:
    excel_path = "results.xlsx"

    df = load_results(excel_path)

    y_pred = to_binary_availability(df, metric_col_hint="Unnamed: 0")

    y_true = np.ones_like(y_pred)

    plot_confusion_matrix(y_true, y_pred, "confusion_matrix.png")
    metrics = plot_overall_metrics(y_true, y_pred, "overall_metrics.png")
    correct, incorrect = plot_correct_incorrect(y_true, y_pred, "correct_vs_incorrect.png")

    print("Availability-Based Evaluation Summary")
    print(f"Total outputs evaluated: {len(y_pred)}")
    print(f"Correct (matches expected availability): {correct}")
    print(f"Incorrect (does not match expected availability): {incorrect}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Saved figures:")
    print("confusion_matrix.png")
    print("overall_metrics.png")
    print("correct_vs_incorrect.png")

if __name__ == "__main__":
    main()