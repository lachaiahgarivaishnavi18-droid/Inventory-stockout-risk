import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def print_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Risk", "Risk"], yticklabels=["No Risk", "Risk"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Stockout Risk Confusion Matrix")
    plt.tight_layout()
    plt.show()


def model_comparison_table(results_df):
    return results_df[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].sort_values("F1", ascending=False)
