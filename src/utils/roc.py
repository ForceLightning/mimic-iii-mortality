# Scientific Libraries
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# PyTorch
import torch
from torch import Tensor


def auc_charts(labels: Tensor, probas: Tensor, model_name: str):
    labels = labels.to(torch.long)
    sns.set_theme("paper", "whitegrid")
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11.69, 5.15), dpi=100)
    draw_chance = True

    fpr, tpr, _ = roc_curve(labels, probas)
    auroc = roc_auc_score(labels, probas)
    prec, rec, _ = precision_recall_curve(labels, probas)
    auprc = average_precision_score(labels, probas)

    if draw_chance:
        # ROC
        ax[0].plot(
            (0, 1),
            (0, 1),
            linestyle="--",
            color="red",
            label="Chance level (AUROC = 0.5)",
        )
        # PRC
        chance_level = labels.sum() / len(labels)
        ax[1].plot(
            (0, 1),
            (chance_level, chance_level),
            linestyle="--",
            color="red",
            label=f"Chance level (AUPRC = {chance_level:.4f})",
        )

    ax[0].plot(
        fpr, tpr, label=f"AUROC: {auroc:.4f}", lw=2, alpha=0.8, drawstyle="steps-post"
    )
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title("Receiver Operating Characteristics")
    ax[0].set_ylim(-0.1, 1.1)
    ax[0].legend(loc="lower right")

    ax[1].plot(
        rec, prec, label=f"AUPRC: {auprc:.4f}", lw=2, alpha=0.8, drawstyle="steps-post"
    )
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision Recall Curve")
    ax[1].set_ylim(-0.1, 1.1)
    ax[1].legend(loc="lower right")

    fig.suptitle(f"{model_name} ROC and PRC")

    return fig
