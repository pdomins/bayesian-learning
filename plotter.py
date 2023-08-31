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

def __add_roc_plot_template__(title : str):
    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    plt.ylabel("Tasa de verdaderos positivos")
    plt.xlabel("Tasa de falsos positivos")
    plt.title(title)
    plt.plot(np.array([0, 1]),np.array([0, 1]), linestyle="dashed", color="grey")

def add_roc_plot_from_positive_rates(positive_rates : list[dict[str, Any]], annotate : bool = False, font_size : int = 5, color : str = "purple", label : str = None):
    FPRs, TPRs = np.array(list(map(lambda pr : pr["FPR"], positive_rates))), np.array(list(map(lambda pr : pr["TPR"], positive_rates)))
    if label is not None:
        plt.plot(FPRs, TPRs, color=color, label=label)
    else:
        plt.plot(FPRs, TPRs, color=color)
    plt.plot(FPRs, TPRs, marker='o', color=color)
    plt.plot([0, FPRs.min()], [0, TPRs.min()], color=color)
    if annotate:
        for i, j in zip(FPRs, TPRs):
            plt.annotate('(%.2g, %.2g)' % (i, j), xy=(i, j), textcoords='offset points', xytext=(0,7), ha='center', fontsize=font_size, color=color)

def plot_roc_from_positive_rates(positive_rates : list[dict[str, Any]], title : str, save_file : str = None, annotate : bool = False, font_size : int = 5, label : str = None, color : str = "purple"):
    plt.figure()
    __add_roc_plot_template__(title)
    add_roc_plot_from_positive_rates(positive_rates, annotate, font_size, label=label, color=color)
    plt.legend(loc="lower right")
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight', dpi=1200)
    else:
        plt.show()

def plot_multiple_roc_fromm_positive_rates(positive_rates_list : list[list[dict[str, Any]]], colors : list[str], title : str, save_file : str = None, annotate : bool = False, font_size : int = 5, labels : list[str] = None):
    plt.figure()
    __add_roc_plot_template__(title)
    for i in range(len(positive_rates_list)):
        add_roc_plot_from_positive_rates(positive_rates_list[i], annotate, font_size, color=colors[i], label=labels[i])
    plt.legend(loc="lower right")
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight', dpi=1200)
    else:
        plt.show()