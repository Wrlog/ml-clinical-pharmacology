"""
Render every dashboard figure with matplotlib/seaborn.

Reads the CSVs in results/ (and, for the raw-data figures of analysis 1, the
input CSV in data/ when present) and writes SVGs to dashboard/figures/.
All combinations offered by the dashboard selectors are pre-rendered.

Run from the repository root after the R analyses:
    python python/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, linkage

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "dashboard" / "figures"

# ---------------------------------------------------------------- style ----
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"  # matches the dashboard card background
DIVERGING = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", BLUE])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#86b6ef", "#104281"])

MODELS = ["rf", "svm_linear", "svm_rbf", "gbm"]
MODEL_LABEL = {"rf": "Random forest", "svm_linear": "SVM (linear)", "svm_rbf": "SVM (RBF)", "gbm": "Gradient boosting"}
BIOLOGICAL = {"plasmaSerotonin", "AHR", "TSPAN5", "CYP2C19", "DEFB1", "SLC6A4", "COMT", "CYP2D6", "ERICH3"}
ANALYSES = {
    "ssri": {"dir": "02_ssri_response", "title": "SSRI remission",
             "fs": {"clinical": "Clinical", "clinical_biomarkers": "Clinical + biomarkers"}},
    "ssri_symptoms": {"dir": "03_ssri_response_symptoms", "title": "+ Symptom items",
                      "fs": {"clinical": "Clinical + symptoms", "clinical_biomarkers": "Clinical + symptoms + biomarkers"}},
}
METRICS = {
    "roc_auc": ("Test ROC AUC", 0.5, "Chance"),
    "cv_roc_auc": ("Cross-validated ROC AUC", 0.5, "Chance"),
    "accuracy": ("Test accuracy", "nir", "No-information rate"),
    "bal_accuracy": ("Test balanced accuracy", 0.5, "Chance"),
    "sens": ("Sensitivity (remission)", None, None),
    "spec": ("Specificity", None, None),
    "ppv": ("Positive predictive value", None, None),
    "f_meas": ("F1 score", None, None),
}
DIAG_ORDER = ["Healthy Controls", "Mild", "Moderate", "Treatment Resistant"]
DIAG_MARKER = {"Healthy Controls": "o", "Mild": "s", "Moderate": "D", "Treatment Resistant": "^"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "svg.fonttype": "path",
})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def value_labels(ax, bars, fmt="{:.2f}", pad=0.012):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(b.get_x() + b.get_width() / 2, h + pad, fmt.format(h), ha="center", va="bottom",
                    fontsize=8, color=INK2)


def cluster_label(key):
    if key.startswith("kmeans_"):
        return f"k-means, k = {key.split('_')[1]}"
    if key.startswith("hclust_"):
        _, link, k = key.split("_")
        return f"Hierarchical ({link}), k = {k}"
    if key == "gmm_auto":
        return "GMM, BIC-selected"
    return f"GMM, G = {key.split('_')[1]}"


# ------------------------------------------------------ 1. unsupervised ----
def unsupervised():
    d = RESULTS / "01_unsupervised"
    out = OUT / "unsupervised"
    ari = pd.read_csv(d / "cluster_diagnosis_agreement.csv").sort_values("adjusted_rand_index")
    assign = pd.read_csv(d / "cluster_assignments.csv")
    pv = pd.read_csv(d / "pca_variance.csv")
    load = pd.read_csv(d / "pca_loadings.csv")
    elbow = pd.read_csv(d / "kmeans_elbow.csv")

    # Agreement with diagnosis
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = [cluster_label(k) for k in ari["clustering"]]
    colors = [BLUE if v > 0.999 else "#86b6ef" for v in ari["adjusted_rand_index"]]
    bars = ax.barh(labels, ari["adjusted_rand_index"], color=colors, height=0.66)
    for b in bars:
        ax.text(b.get_width() + 0.015, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}",
                va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Adjusted Rand index vs. clinical diagnosis")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=BLUE, label="Perfect agreement (ARI = 1)"), Patch(color="#86b6ef", label="Partial agreement")],
              loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2)
    save(fig, out / "ari.svg")

    # PCA scatter, one per clustering
    pc1, pc2 = pv.loc[0, "variance.percent"], pv.loc[1, "variance.percent"]
    diags = [x for x in DIAG_ORDER if x in set(assign["Diagnosis"])] or sorted(assign["Diagnosis"].unique())
    for key in ari["clustering"]:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        clusters = sorted(assign[key].unique())
        for i, c in enumerate(clusters):
            for dg in diags:
                sub = assign[(assign[key] == c) & (assign["Diagnosis"] == dg)]
                if len(sub):
                    ax.scatter(sub["PC1"], sub["PC2"], s=70, marker=DIAG_MARKER.get(dg, "o"), color=SERIES[i % 5],
                               edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.set_xlabel(f"PC1 ({pc1:.0f}% of variance)")
        ax.set_ylabel(f"PC2 ({pc2:.0f}% of variance)")
        h1 = [Line2D([], [], marker="o", ls="", color=SERIES[i % 5], markersize=8, label=f"Cluster {c}") for i, c in enumerate(clusters)]
        h2 = [Line2D([], [], marker=DIAG_MARKER.get(dg, "o"), ls="", markerfacecolor="none", markeredgecolor=INK2,
                     markersize=8, label=dg) for dg in diags]
        l1 = ax.legend(handles=h1, title="Cluster", loc="upper left", bbox_to_anchor=(1.01, 1), title_fontsize=9, alignment="left")
        ax.add_artist(l1)
        ax.legend(handles=h2, title="Diagnosis", loc="upper left", bbox_to_anchor=(1.01, 0.52), title_fontsize=9, alignment="left")
        save(fig, out / f"pca_{key}.svg")

    # Elbow curve
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(elbow["k"], elbow["within_ss"] / 1000, color=BLUE, lw=2, marker="o", ms=5, mec=SURFACE, mew=1, zorder=3)
    k4 = elbow.loc[elbow["k"] == 4, "within_ss"].iloc[0] / 1000
    ax.annotate("elbow at k = 4", xy=(4, k4), xytext=(8, k4 + 30), color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Within-cluster SS (×1,000)")
    ax.set_xticks(range(2, 21, 2))
    save(fig, out / "elbow.svg")

    # Scree with cumulative variance (both in %, one axis)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    comps = [c.replace("Dim.", "PC") for c in pv["component"]]
    ax.bar(comps, pv["variance.percent"], color=BLUE, width=0.66, label="Per component")
    ax.plot(comps, pv["cumulative.variance.percent"], color=ORANGE, lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1, label="Cumulative")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Variance explained (%)")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center right")
    save(fig, out / "scree.svg")

    # Loadings heatmap
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    m = load.set_index("gene")[["PC1", "PC2", "PC3"]]
    lim = np.abs(m.values).max()
    sns.heatmap(m, ax=ax, cmap=DIVERGING, vmin=-lim, vmax=lim, annot=True, fmt=".2f", annot_kws={"fontsize": 8},
                linewidths=1.5, linecolor=SURFACE, cbar_kws={"shrink": 0.8, "label": "Loading"})
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.tick_params(axis="y", rotation=0)
    save(fig, out / "loadings.svg")

    # Raw-data figures (need the input data)
    raw = ROOT / "data" / "gene_expression.csv"
    if not raw.exists():
        print("data/gene_expression.csv not found - skipping raw-data figures")
        return
    df = pd.read_csv(raw)
    genes = [c for c in df.columns if c not in ("ID", "Diagnosis")]
    dcol = {dg: SERIES[i] for i, dg in enumerate(diags)}
    long = df.melt(id_vars=["ID", "Diagnosis"], value_vars=genes, var_name="gene", value_name="expression")

    g = sns.FacetGrid(long, col="gene", col_wrap=5, height=1.9, aspect=1.15, sharey=False, col_order=genes)
    g.map_dataframe(sns.boxplot, x="Diagnosis", y="expression", hue="Diagnosis", order=diags, hue_order=diags,
                    palette=dcol, width=0.62, linewidth=0.8, fliersize=2, legend=False)
    g.map_dataframe(sns.stripplot, x="Diagnosis", y="expression", order=diags, color=INK, size=2.2, alpha=0.45, jitter=0.18)
    g.set_titles("{col_name}", size=10, weight="semibold")
    g.set_axis_labels("", "Expression")
    for ax in g.axes.flat:
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
    g.figure.legend(handles=[Patch(color=dcol[dg], label=dg) for dg in diags], loc="lower center", ncol=len(diags),
                    bbox_to_anchor=(0.5, -0.04))
    save(g.figure, out / "expression_boxplots.svg")

    g = sns.FacetGrid(long, col="gene", col_wrap=5, height=1.9, aspect=1.15, sharey=False, col_order=genes)
    g.map_dataframe(sns.kdeplot, x="expression", hue="Diagnosis", hue_order=diags, palette=dcol, bw_adjust=1.4,
                    fill=True, alpha=0.18, linewidth=1.4, common_norm=False, warn_singular=False)
    g.set_titles("{col_name}", size=10, weight="semibold")
    g.set_axis_labels("Expression", "Density")
    for ax in g.axes.flat:
        ax.set_yticks([])
    g.figure.legend(handles=[Patch(color=dcol[dg], label=dg) for dg in diags], loc="lower center", ncol=len(diags),
                    bbox_to_anchor=(0.5, -0.04))
    save(g.figure, out / "expression_density.svg")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
    X = df[genes].to_numpy()
    for ax, method in zip(axes, ["complete", "average", "single"]):
        Z = linkage(X, method=method, metric="euclidean")
        dn = dendrogram(Z, ax=ax, labels=df["ID"].astype(str).tolist(), color_threshold=0, above_threshold_color=INK2,
                        leaf_font_size=7)
        for lbl in ax.get_xticklabels():
            dg = df.loc[df["ID"].astype(str) == lbl.get_text(), "Diagnosis"].iloc[0]
            lbl.set_color(dcol[dg])
            lbl.set_fontweight("bold")
        ax.set_title(f"{method.title()} linkage")
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_ylabel("Euclidean distance" if method == "complete" else "")
    fig.legend(handles=[Patch(color=dcol[dg], label=dg) for dg in diags], loc="lower center", ncol=len(diags),
               bbox_to_anchor=(0.5, -0.16), title="Leaf label colour = diagnosis", title_fontsize=9)
    fig.tight_layout()
    save(fig, out / "dendrograms.svg")


# ---------------------------------------------------- 2/3. supervised ----
def supervised(key, cfg):
    d = RESULTS / cfg["dir"]
    out = OUT / key
    fs_labels = cfg["fs"]
    m = pd.read_csv(d / "metrics.csv")
    roc = pd.read_csv(d / "roc_curves.csv")
    imp = pd.read_csv(d / "variable_importance.csv")
    cm = pd.read_csv(d / "confusion_matrices.csv")
    scores = pd.read_csv(d / "pca_scores.csv")
    pv = pd.read_csv(d / "pca_variance.csv")
    x = np.arange(len(MODELS))
    w = 0.38

    # Model comparison, one figure per metric
    for met, (label, ref, ref_label) in METRICS.items():
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for j, fs in enumerate(fs_labels):
            rows = m.set_index(["model", "feature_set"]).loc[[(mo, fs) for mo in MODELS]]
            err = rows["cv_roc_auc_se"].to_numpy() if met == "cv_roc_auc" else None
            bars = ax.bar(x + (j - 0.5) * w, rows[met].to_numpy(dtype=float), w * 0.94, color=SERIES[j], label=fs_labels[fs],
                          yerr=err, error_kw=dict(ecolor=INK2, elinewidth=1, capsize=3))
            value_labels(ax, bars, pad=0.02 if err is not None else 0.012)
        refv = m["nir"].iloc[0] if ref == "nir" else ref
        if refv is not None:
            ax.axhline(refv, color=MUTED, ls=(0, (4, 3)), lw=1, zorder=1, label=f"{ref_label} ({refv:.2f})")
        ax.set_xticks(x, [MODEL_LABEL[mo] for mo in MODELS])
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
        save(fig, out / f"compare_{met}.svg")

    for fs, fs_label in fs_labels.items():
        # ROC curves
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.plot([0, 1], [0, 1], color=MUTED, ls=(0, (4, 3)), lw=1)
        for i, mo in enumerate(MODELS):
            r = roc[(roc["model"] == mo) & (roc["feature_set"] == fs)].copy()
            r["fpr"] = 1 - r["specificity"]
            r = r.sort_values(["fpr", "sensitivity"])
            auc = m[(m["model"] == mo) & (m["feature_set"] == fs)]["roc_auc"].iloc[0]
            ax.step(r["fpr"], r["sensitivity"], where="post", color=SERIES[i], lw=2, label=f"{MODEL_LABEL[mo]}  (AUC {auc:.2f})")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("False positive rate (1 − specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.legend(loc="lower right", frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=1)
        save(fig, out / f"roc_{fs}.svg")

        # PCA of training cohort
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        s = scores[scores["feature_set"] == fs]
        p = pv[pv["feature_set"] == fs].reset_index(drop=True)
        for resp, name, col in [(0, "No remission", ORANGE), (1, "Remission", BLUE)]:
            sub = s[s["response"] == resp]
            ax.scatter(sub["PC1"], sub["PC2"], s=18, color=col, alpha=0.75, edgecolor=SURFACE, linewidth=0.5, label=name)
        ax.set_xlabel(f"PC1 ({p.loc[0, 'variance.percent']:.0f}% of variance)")
        ax.set_ylabel(f"PC2 ({p.loc[1, 'variance.percent']:.0f}%)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, handletextpad=0.2, markerscale=1.6)
        save(fig, out / f"pca_{fs}.svg")

        for mo in MODELS:
            # Variable importance
            r = imp[(imp["model"] == mo) & (imp["feature_set"] == fs)].sort_values("Importance")
            fig, ax = plt.subplots(figsize=(5.6, max(2.6, 0.24 * len(r) + 0.9)))
            if r["Importance"].fillna(0).abs().sum() == 0:
                ax.axis("off")
                ax.text(0.5, 0.5, "All importances are 0:\nthis model predicts the same class for every patient,\n"
                        "so shuffling any variable changes nothing.", ha="center", va="center", color=INK2, fontsize=10,
                        transform=ax.transAxes)
            else:
                cols = [ORANGE if v in BIOLOGICAL else BLUE for v in r["Variable"]]
                ax.barh(r["Variable"], r["Importance"], color=cols, height=0.7)
                ax.set_xlim(0, 105)
                ax.set_xlabel("Scaled importance (0–100)")
                ax.grid(axis="y", visible=False)
                ax.tick_params(axis="y", labelsize=8.5)
                ax.legend(handles=[Patch(color=BLUE, label="Clinical / symptom"), Patch(color=ORANGE, label="Biological")],
                          loc="lower right")
            save(fig, out / f"importance_{mo}_{fs}.svg")

            # Confusion matrix
            c = cm[(cm["model"] == mo) & (cm["feature_set"] == fs)]
            mat = np.array([[c[(c["Prediction"] == p_) & (c["Truth"] == t)]["n"].sum() for t in (1, 0)] for p_ in (1, 0)])
            col_tot = mat.sum(axis=0, keepdims=True)
            pct = np.divide(mat, col_tot, out=np.zeros_like(mat, dtype=float), where=col_tot > 0)
            annot = np.array([[f"{mat[i, j]}\n{pct[i, j]:.0%} of actual" for j in range(2)] for i in range(2)])
            fig, ax = plt.subplots(figsize=(4.6, 3.2))
            sns.heatmap(pct, ax=ax, cmap=SEQUENTIAL, vmin=0, vmax=1, annot=annot, fmt="", cbar=False, linewidths=2,
                        linecolor=SURFACE, annot_kws={"fontsize": 10.5},
                        xticklabels=["Remission", "No remission"], yticklabels=["Remission", "No remission"])
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
            ax.tick_params(axis="y", rotation=0)
            ax.grid(False)
            save(fig, out / f"confusion_{mo}_{fs}.svg")


# ------------------------------------------------------------ overview ----
def overview():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    x = np.arange(len(MODELS))
    w = 0.38
    for ax, (i, (key, cfg)) in zip(axes, enumerate(ANALYSES.items())):
        m = pd.read_csv(RESULTS / cfg["dir"] / "metrics.csv").set_index(["model", "feature_set"])
        for j, fs in enumerate(cfg["fs"]):
            vals = [m.loc[(mo, fs), "roc_auc"] for mo in MODELS]
            bars = ax.bar(x + (j - 0.5) * w, vals, w * 0.94, color=SERIES[j],
                          label="Clinical features" if j == 0 else "+ Biomarkers")
            value_labels(ax, bars)
        ax.axhline(0.5, color=MUTED, ls=(0, (4, 3)), lw=1, zorder=1, label="Chance (0.50)")
        ax.set_title(f"{i + 2} · {cfg['title']}")
        ax.set_xticks(x, [MODEL_LABEL[mo] for mo in MODELS])
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test ROC AUC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, OUT / "overview_auc.svg")


if __name__ == "__main__":
    unsupervised()
    for k, c in ANALYSES.items():
        supervised(k, c)
    overview()
    n = len(list(OUT.rglob("*.svg")))
    kb = sum(f.stat().st_size for f in OUT.rglob("*.svg")) / 1024
    print(f"Wrote {n} figures to {OUT.relative_to(ROOT)} ({kb:.0f} KB)")
