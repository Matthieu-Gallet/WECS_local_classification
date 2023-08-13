from utils import *
import numpy as np
import matplotlib as mpl

mpl.use("pgf")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[utf8x]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{cmbright}",
            ]
        ),
    }
)


def plot_variation_alpha(path):
    palpha = open_pkl(path)
    palpha[1] = palpha[-1]
    del palpha[-1]
    data = palpha

    fig, ax = plt.subplots(figsize=(5, 2.5))

    sorted_keys = sorted(data.keys())

    # Tri des valeurs associées aux clés
    sorted_values = [100 * np.array(data[key]) for key in sorted_keys]

    # Création du graphe boxplot avec échelle logarithmique sur l'axe x
    bplot2 = ax.boxplot(
        sorted_values,
        labels=1 - np.array(sorted_keys),
        patch_artist=True,
        showfliers=False,
        notch=True,
    )

    colors_ = (len(sorted_keys) - 1) * ["tab:blue"] + ["tab:orange"]
    for _, line_list in bplot2.items():
        for line in line_list:
            line.set_color("grey")
            line.set_linewidth(0.75)
    for patch, color in zip(bplot2["boxes"], colors_):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.legend(
        [bplot2["boxes"][0], bplot2["boxes"][-1]], ["L-WECS", "WECS"], loc="upper right"
    )
    ax.set_xlabel("Local weights ($\\alpha$)", fontsize=12)
    ax.set_ylabel("F1-score (%)", fontsize=12)
    # ax.set_title('Variation of F1-score with local weights ($\\alpha$)')
    ax.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    ax.set_ylim(62, 82)
    plt.tight_layout()
    makedirs("figures", exist_ok=True)
    plt.savefig("figures/variation_alpha.pdf", backend="pgf")


def plot_variation_weights(path):
    palpha = open_pkl(path)
    data = palpha

    fig, ax = plt.subplots(figsize=(5, 2.5))

    sorted_keys = sorted(data.keys())

    # Tri des valeurs associées aux clés
    sorted_values = [100 * np.array(data[key]) for key in sorted_keys]

    # Création du graphe boxplot avec échelle logarithmique sur l'axe x
    bplot2 = ax.boxplot(
        sorted_values,
        labels=np.array(sorted_keys).astype(int),
        patch_artist=True,
        showfliers=False,
        notch=True,
    )

    colors = "tab:green"
    for _, line_list in bplot2.items():
        for line in line_list:
            line.set_color("grey")
            line.set_linewidth(0.75)
    for patch in bplot2["boxes"]:
        patch.set_facecolor(colors)
        patch.set_alpha(0.75)

    ax.set_xlabel("Window size $k$", fontsize=12)
    ax.set_ylabel("F1-score (%)", fontsize=12)
    # ax.set_title('Variation of F1-score with local weights ($\\alpha$)')
    ax.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    ax.set_ylim(62, 82)
    plt.tight_layout()
    makedirs("figures", exist_ok=True)
    plt.savefig("figures/variation_weights.pdf", backend="pgf")


if __name__ == "__main__":
    palpha = "./results/BFOLD_knn_f1_1691934871.pkl"
    plot_variation_alpha(palpha)

    pwsize = "results/BFOLD_knn_f1_1691940950.pkl"
    plot_variation_weights(pwsize)

    import pandas as pd

    mf7 = "results/multi_f1__7_1691956910.pkl"
    mfg = "results/multi_f1_-1_1691956910.pkl"
    mf63 = "results/multi_f1_63_1691956910.pkl"

    mf7, le = open_pkl(mf7)
    mfg, le = open_pkl(mfg)
    mf63, le = open_pkl(mf63)

    print(pd.DataFrame(mf7.mean(axis=0), index=le, columns=le).round(2))
    print(pd.DataFrame(mfg.mean(axis=0), index=le, columns=le).round(2))
    print(pd.DataFrame(mf63.mean(axis=0), index=le, columns=le).round(2))

    print(pd.DataFrame(mf7.std(axis=0), index=le, columns=le).round(2))
    print(pd.DataFrame(mfg.std(axis=0), index=le, columns=le).round(2))
    print(pd.DataFrame(mf63.std(axis=0), index=le, columns=le).round(2))
