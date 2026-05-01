import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def visualize_residual_plot(y_pred: np.ndarray, y_true: np.ndarray) -> Figure:
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(x=y_pred, y=y_true - y_pred, c='steelblue', marker='s', edgecolors='white')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.title("Residuals vs Predicted Values", fontsize=18)
    plt.xlabel("Predicted Values", fontsize=16)
    plt.ylabel("Residuals", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.close(fig)
    return fig
