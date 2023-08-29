import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_confusion_matrix(confusion_matrix : pd.DataFrame, title : str, save_file : str = None, format : str = ""):
    plt.figure()
    min_val = 0
    max_val = confusion_matrix.to_numpy().max()
    ax = sns.heatmap(confusion_matrix, vmin=min_val, vmax=max_val, annot=True, fmt=format, linewidth=.5, cmap=sns.cubehelix_palette(as_cmap=True))
    ax.set(xlabel="Predicción", ylabel="Real", title=title)
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight', dpi=1200)
    else:
        plt.show()