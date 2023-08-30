import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any
import numpy as np

def plot_confusion_matrix(confusion_matrix : pd.DataFrame, title : str, save_file : str = None, format : str = ""):
    plt.figure()
    min_val = 0
    max_val = confusion_matrix.to_numpy().max()
    ax = sns.heatmap(confusion_matrix, vmin=min_val, vmax=max_val, annot=True, fmt=format, linewidth=1, cmap=sns.cubehelix_palette(as_cmap=True))
    ax.set(xlabel="Predicción", ylabel="Real", title=title)
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight', dpi=1200)
    else:
        plt.show()

def plot_roc_from_positive_rates(positive_rates : list[dict[str, Any]], title : str, save_file : str = None, annotate : bool = False, font_size : int = 5):
    plt.figure()
    FPRs, TPRs = np.array(list(map(lambda pr : pr["FPR"], positive_rates))), np.array(list(map(lambda pr : pr["TPR"], positive_rates)))
    plt.xlim((-0.025, 1.025))
    plt.ylim((-0.025, 1.025))
    plt.ylabel("Tasa de verdaderos positivos")
    plt.xlabel("Tasa de falsos positivos")
    plt.title(title)
    plt.plot(np.array([0, 1]),np.array([0, 1]), linestyle="dashed", color="grey")
    plt.plot(FPRs, TPRs, color="purple")
    plt.plot(FPRs, TPRs, marker='o', color="purple")
    plt.vlines(x = 0, ymin=0, ymax=TPRs.min(), colors="purple")
    if annotate:
        for i, j in zip(FPRs, TPRs):
            plt.annotate('(%.2g, %.2g)' % (i, j), xy=(i, j), textcoords='offset points', xytext=(0,10), ha='center', fontsize=font_size, color="purple")
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight', dpi=1200)
    else:
        plt.show()