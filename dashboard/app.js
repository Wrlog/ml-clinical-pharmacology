/* Dashboard for results/ - reads window.RESULTS (dashboard/results.js). */
(function () {
  "use strict";

  const R = window.RESULTS;
  const MODELS = ["rf", "svm_linear", "svm_rbf", "gbm"];
  const MODEL_LABEL = { rf: "Random forest", svm_linear: "SVM (linear)", svm_rbf: "SVM (RBF)", gbm: "Gradient boosting" };
  const BIOLOGICAL = ["plasmaSerotonin", "AHR", "TSPAN5", "CYP2C19", "DEFB1", "SLC6A4", "COMT", "CYP2D6", "ERICH3"];
  const ANALYSES = {
    ssri: {
      title: "SSRI remission",
      featureLabel: { clinical: "Clinical", clinical_biomarkers: "Clinical + biomarkers" },
      selectedBy: "accuracy"
    },
    ssri_symptoms: {
      title: "+ Symptom items",
      featureLabel: { clinical: "Clinical + symptoms", clinical_biomarkers: "Clinical + symptoms + biomarkers" },
      selectedBy: "ROC AUC"
    }
  };
  const METRICS = [
    { key: "roc_auc", label: "Test ROC AUC", ref: 0.5, refLabel: "chance" },
    { key: "cv_roc_auc", label: "Cross-validated ROC AUC", err: "cv_roc_auc_se", ref: 0.5, refLabel: "chance" },
    { key: "accuracy", label: "Test accuracy", ref: "nir", refLabel: "no-information rate" },
    { key: "bal_accuracy", label: "Test balanced accuracy", ref: 0.5, refLabel: "chance" },
    { key: "sens", label: "Sensitivity (remission)" },
    { key: "spec", label: "Specificity" },
    { key: "ppv", label: "Positive predictive value" },
    { key: "f_meas", label: "F1 score" }
  ];
  const CLUSTER_LABEL = (k) => {
    let m;
    if ((m = k.match(/^kmeans_(\d+)$/))) return `k-means, k = ${m[1]}`;
    if ((m = k.match(/^hclust_(\w+)_(\d+)$/))) return `Hierarchical (${m[1]}), k = ${m[2]}`;
    if (k === "gmm_auto") return "GMM, BIC-selected";
    if ((m = k.match(/^gmm_(\d+)$/))) return `GMM, G = ${m[1]}`;
    return k;
  };
  const DIAG_SYMBOL = ["circle", "square", "diamond", "triangle-up"];

  const fmt = (v, d = 3) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
  const fmtP = (p) => (p == null ? "–" : p < 0.001 ? "<0.001" : Number(p).toFixed(3));
  const el = (tag, attrs = {}, text) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    return n;
  };

  /* ---------- theme ---------- */
  let C = {};
  function readTheme() {
    const s = getComputedStyle(document.documentElement);
    const v = (n) => s.getPropertyValue(n).trim();
    C = {
      surface: v("--surface"), text: v("--text-primary"), text2: v("--text-secondary"), muted: v("--text-muted"),
      grid: v("--grid"), axis: v("--axis"), series: [v("--s1"), v("--s2"), v("--s3"), v("--s4"), v("--s5")],
      seqLo: v("--seq-lo"), seqHi: v("--seq-hi"), divNeg: v("--div-neg"), divMid: v("--div-mid"), divPos: v("--div-pos")
    };
  }
  function baseLayout(extra = {}) {
    const axis = { gridcolor: C.grid, linecolor: C.axis, zerolinecolor: C.axis, tickfont: { color: C.muted, size: 11 },
      title: { font: { color: C.text2, size: 12 } }, automargin: true };
    return Object.assign({
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: C.text, size: 12 },
      margin: { l: 8, r: 12, t: 8, b: 8 },
      xaxis: Object.assign({}, axis), yaxis: Object.assign({}, axis),
      legend: { orientation: "h", y: -0.18, x: 0, font: { color: C.text2, size: 12 } },
      hoverlabel: { bgcolor: C.surface, bordercolor: C.axis, font: { color: C.text, size: 12 } },
      bargap: 0.3, bargroupgap: 0.08
    }, extra);
  }
  function mergeAxis(layout, key, props) { layout[key] = Object.assign({}, layout[key], props); return layout; }
  const CONFIG = { displayModeBar: false, responsive: true };
  const plot = (id, data, layout) => Plotly.react(typeof id === "string" ? document.getElementById(id) : id, data, layout, CONFIG);

  /* ---------- tiles & tables ---------- */
  function tiles(container, items) {
    container.replaceChildren(...items.map((t) => {
      const d = el("div", { class: "tile" });
      d.append(el("div", { class: "label" }, t.label), el("div", { class: "value" }, t.value));
      if (t.detail) d.append(el("div", { class: "detail" }, t.detail));
      return d;
    }));
  }

  /* ---------- overview ---------- */
  function renderOverview() {
    const u = R.unsupervised;
    const ari = [...u.cluster_diagnosis_agreement].sort((a, b) => b.adjusted_rand_index - a.adjusted_rand_index);
    const perfect = ari.filter((r) => r.adjusted_rand_index > 0.999).length;
    const items = [{ label: "Clustering vs diagnosis", value: `${perfect} / ${ari.length}`,
      detail: "clusterings recover diagnosis perfectly (ARI = 1)" }];
    for (const [key, cfg] of Object.entries(ANALYSES)) {
      const m = R[key] && R[key].metrics;
      if (!m) continue;
      const best = [...m].sort((a, b) => b.roc_auc - a.roc_auc)[0];
      items.push({ label: `Best test AUC · ${cfg.title}`, value: fmt(best.roc_auc, 2),
        detail: `${MODEL_LABEL[best.model]}, ${cfg.featureLabel[best.feature_set].toLowerCase()}` });
    }
    tiles(document.getElementById("overview-tiles"), items);

    const keys = Object.keys(ANALYSES).filter((k) => R[k] && R[k].metrics);
    const data = [];
    const layout = baseLayout({ barmode: "group", margin: { l: 8, r: 12, t: 28, b: 8 }, annotations: [], shapes: [] });
    keys.forEach((key, i) => {
      const xa = i === 0 ? "x" : "x" + (i + 1), ya = i === 0 ? "y" : "y" + (i + 1);
      const fsKeys = Object.keys(ANALYSES[key].featureLabel);
      fsKeys.forEach((fs, j) => {
        const rows = MODELS.map((m) => R[key].metrics.find((r) => r.model === m && r.feature_set === fs));
        data.push({
          type: "bar", xaxis: xa, yaxis: ya, name: j === 0 ? "Clinical features" : "+ Biomarkers",
          legendgroup: fs, showlegend: i === 0,
          x: MODELS.map((m) => MODEL_LABEL[m]), y: rows.map((r) => r && r.roc_auc),
          marker: { color: C.series[j], cornerradius: 4 },
          customdata: rows.map(() => ANALYSES[key].featureLabel[fs]),
          hovertemplate: "<b>%{y:.3f}</b> test AUC<br>%{x} · %{customdata}<extra></extra>"
        });
      });
      const dom = keys.length === 1 ? [0, 1] : i === 0 ? [0, 0.48] : [0.52, 1];
      layout[xa === "x" ? "xaxis" : "xaxis" + (i + 1)] = Object.assign({}, layout.xaxis, { domain: dom, anchor: ya });
      layout[ya === "y" ? "yaxis" : "yaxis" + (i + 1)] = Object.assign({}, layout.yaxis, { range: [0, 1], anchor: xa,
        title: i === 0 ? { text: "ROC AUC", font: { color: C.text2, size: 12 } } : undefined, showticklabels: i === 0 });
      layout.annotations.push({ text: `<b>${i + 2} · ${ANALYSES[key].title}</b>`, x: (dom[0] + dom[1]) / 2, y: 1.08,
        xref: "paper", yref: "paper", showarrow: false, font: { size: 13, color: C.text } });
      layout.shapes.push({ type: "line", xref: `${xa} domain`, yref: ya, x0: 0, x1: 1, y0: 0.5, y1: 0.5,
        line: { color: C.muted, dash: "dash", width: 1 } });
    });
    plot("overview-auc", data, layout);
  }

  /* ---------- 1. unsupervised ---------- */
  function renderUnsupervised() {
    const u = R.unsupervised;
    const ari = [...u.cluster_diagnosis_agreement].sort((a, b) => a.adjusted_rand_index - b.adjusted_rand_index);
    const auto = u.cluster_assignments.length ? new Set(u.cluster_assignments.map((r) => r.gmm_auto)).size : "–";
    const pv = u.pca_variance;
    tiles(document.getElementById("unsup-tiles"), [
      { label: "Subjects × genes", value: `${u.cluster_assignments.length} × ${u.pca_loadings.length}`, detail: "4 diagnostic groups" },
      { label: "PC1 + PC2 variance", value: `${fmt(pv[1]["cumulative.variance.percent"], 0)}%`, detail: "of total variance on the first two PCs" },
      { label: "GMM (BIC) chooses", value: `${auto} clusters`, detail: "same as the number of diagnoses" },
      { label: "Best ARI", value: fmt(ari[ari.length - 1].adjusted_rand_index, 2), detail: "k-means k = 4, all hierarchical linkages, GMM" }
    ]);

    plot("unsup-ari", [{
      type: "bar", orientation: "h", x: ari.map((r) => r.adjusted_rand_index), y: ari.map((r) => CLUSTER_LABEL(r.clustering)),
      marker: { color: C.series[0], cornerradius: 4 }, text: ari.map((r) => fmt(r.adjusted_rand_index, 2)),
      textposition: "outside", textfont: { color: C.text2 }, cliponaxis: false,
      hovertemplate: "<b>ARI %{x:.3f}</b><br>%{y}<extra></extra>"
    }], mergeAxis(baseLayout({ margin: { l: 8, r: 36, t: 8, b: 8 } }), "xaxis", { range: [0, 1.05], title: { text: "Adjusted Rand index" } }));

    // PCA scatter coloured by chosen clustering
    const sel = document.getElementById("unsup-cluster-select");
    if (!sel.options.length) {
      for (const r of [...u.cluster_diagnosis_agreement]) sel.append(el("option", { value: r.clustering }, CLUSTER_LABEL(r.clustering)));
      sel.value = "gmm_auto";
      sel.addEventListener("change", drawPca);
    }
    function drawPca() {
      const key = sel.value;
      const diags = [...new Set(u.cluster_assignments.map((r) => r.Diagnosis))];
      const clusters = [...new Set(u.cluster_assignments.map((r) => r[key]))].sort((a, b) => a - b);
      const traces = clusters.map((c, i) => {
        const rows = u.cluster_assignments.filter((r) => r[key] === c);
        return {
          type: "scatter", mode: "markers", name: `Cluster ${c}`, legendgroup: `c${c}`, showlegend: false,
          x: rows.map((r) => r.PC1), y: rows.map((r) => r.PC2),
          marker: { size: 11, color: C.series[i % 5], symbol: rows.map((r) => DIAG_SYMBOL[diags.indexOf(r.Diagnosis)]),
            line: { color: C.surface, width: 2 } },
          customdata: rows.map((r) => [r.ID, r.Diagnosis]),
          hovertemplate: `<b>Cluster ${c}</b><br>Subject %{customdata[0]} · %{customdata[1]}<br>PC1 %{x:.2f}, PC2 %{y:.2f}<extra></extra>`
        };
      });
      // legend keys: colour = cluster (circles), shape = diagnosis (outlines)
      clusters.forEach((c, i) => traces.push({ type: "scatter", mode: "markers", x: [null], y: [null], name: `Cluster ${c}`,
        legendgroup: `c${c}`, marker: { size: 10, symbol: "circle", color: C.series[i % 5] }, hoverinfo: "skip" }));
      diags.forEach((d, i) => traces.push({ type: "scatter", mode: "markers", x: [null], y: [null], name: d, legendgroup: "diag",
        marker: { size: 10, symbol: DIAG_SYMBOL[i], color: "rgba(0,0,0,0)", line: { color: C.text2, width: 1.5 } }, hoverinfo: "skip" }));
      const l = baseLayout({ legend: { orientation: "v", x: 1.02, y: 1, font: { color: C.text2, size: 12 } }, margin: { l: 8, r: 8, t: 8, b: 8 } });
      mergeAxis(l, "xaxis", { title: { text: `PC1 (${fmt(pv[0]["variance.percent"], 0)}%)` } });
      mergeAxis(l, "yaxis", { title: { text: `PC2 (${fmt(pv[1]["variance.percent"], 0)}%)` } });
      plot("unsup-pca", traces, l);
    }
    drawPca();

    plot("unsup-elbow", [{
      type: "scatter", mode: "lines+markers", x: u.kmeans_elbow.map((r) => r.k), y: u.kmeans_elbow.map((r) => r.within_ss),
      line: { color: C.series[0], width: 2 }, marker: { size: 8, color: C.series[0], line: { color: C.surface, width: 2 } },
      hovertemplate: "k = %{x}<br><b>%{y:,.0f}</b> within-cluster SS<extra></extra>"
    }], mergeAxis(mergeAxis(baseLayout(), "xaxis", { title: { text: "Number of clusters (k)" }, dtick: 2 }), "yaxis", { title: { text: "Within-cluster SS" } }));

    plot("unsup-scree", [{
      type: "bar", x: pv.map((r) => r.component.replace("Dim.", "PC")), y: pv.map((r) => r["variance.percent"]),
      marker: { color: C.series[0], cornerradius: 4 },
      hovertemplate: "%{x}<br><b>%{y:.1f}%</b> of variance<extra></extra>"
    }], mergeAxis(baseLayout(), "yaxis", { title: { text: "% variance" } }));

    const L = u.pca_loadings, pcs = ["PC1", "PC2", "PC3"];
    const lim = Math.max(...L.flatMap((r) => pcs.map((p) => Math.abs(r[p]))));
    plot("unsup-loadings", [{
      type: "heatmap", x: pcs, y: L.map((r) => r.gene), z: L.map((r) => pcs.map((p) => r[p])), zmin: -lim, zmax: lim,
      colorscale: [[0, C.divNeg], [0.5, C.divMid], [1, C.divPos]], xgap: 2, ygap: 2,
      colorbar: { thickness: 10, outlinewidth: 0, tickfont: { color: C.muted, size: 10 } },
      hovertemplate: "%{y} → %{x}<br><b>%{z:.3f}</b> loading<extra></extra>"
    }], mergeAxis(baseLayout(), "yaxis", { autorange: "reversed" }));

    const g = document.getElementById("unsup-gallery");
    if (!g.children.length) {
      const pick = ["density_by_gene", "density_by_diagnosis", "expression_boxplots", "pca_biplot",
        "hc_complete_linkage", "hc_average_linkage", "hc_single_linkage", "pc_gmm_auto"];
      const caption = { density_by_gene: "Expression density by gene", density_by_diagnosis: "Expression density by diagnosis",
        expression_boxplots: "Expression by diagnosis", pca_biplot: "PCA biplot", hc_complete_linkage: "Dendrogram, complete linkage",
        hc_average_linkage: "Dendrogram, average linkage", hc_single_linkage: "Dendrogram, single linkage", pc_gmm_auto: "GMM clusters on PCs" };
      for (const name of pick) {
        const src = R.figures.unsupervised.find((f) => f.endsWith(`/${name}.png`));
        if (!src) continue;
        const a = el("a", { href: src, target: "_blank", rel: "noopener" });
        a.append(el("img", { src, alt: caption[name], loading: "lazy" }));
        const fig = el("figure");
        fig.append(a, el("figcaption", {}, caption[name]));
        g.append(fig);
      }
    }
  }

  /* ---------- 2 & 3. supervised ---------- */
  const state = {};
  function renderSupervised(key) {
    const res = R[key], cfg = ANALYSES[key];
    const root = document.getElementById(`tab-${key}`);
    if (!res || !res.metrics) { root.replaceChildren(el("p", { class: "note" }, "Results not generated yet.")); return; }
    if (!root.children.length) {
      root.append(document.getElementById("supervised-template").content.cloneNode(true));
      state[key] = { fs: "clinical_biomarkers", model: [...res.metrics].sort((a, b) => b.roc_auc - a.roc_auc)[0].model, metric: "roc_auc" };
      const q = (r) => root.querySelector(`[data-role="${r}"]`);
      const seg = q("feature-set");
      for (const [fs, label] of Object.entries(cfg.featureLabel)) {
        const b = el("button", { role: "radio", "data-fs": fs, "aria-checked": String(fs === state[key].fs) }, label);
        b.addEventListener("click", () => { state[key].fs = fs; seg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b))); draw(key); });
        seg.append(b);
      }
      const ms = q("model");
      MODELS.forEach((m) => ms.append(el("option", { value: m }, MODEL_LABEL[m])));
      ms.value = state[key].model;
      ms.addEventListener("change", () => { state[key].model = ms.value; draw(key); });
      const mt = q("metric");
      METRICS.forEach((m) => mt.append(el("option", { value: m.key }, m.label)));
      mt.addEventListener("change", () => { state[key].metric = mt.value; draw(key); });
    }
    draw(key);
  }

  function draw(key) {
    const res = R[key], cfg = ANALYSES[key], st = state[key];
    const root = document.getElementById(`tab-${key}`);
    const q = (r) => root.querySelector(`[data-role="${r}"]`);
    const M = res.metrics;
    const row = M.find((r) => r.model === st.model && r.feature_set === st.fs);
    const fsLabel = cfg.featureLabel[st.fs];
    const nTest = res.confusion_matrices.filter((r) => r.model === st.model && r.feature_set === st.fs).reduce((s, r) => s + r.n, 0);
    const bestOf = (fs) => M.filter((r) => r.feature_set === fs).sort((a, b) => b.roc_auc - a.roc_auc)[0];
    const bClin = bestOf("clinical"), bBio = bestOf("clinical_biomarkers");
    const gain = bBio.roc_auc - bClin.roc_auc;

    tiles(q("tiles"), [
      { label: `${MODEL_LABEL[st.model]} · test AUC`, value: fmt(row.roc_auc, 2), detail: `${fsLabel} · CV ${fmt(row.cv_roc_auc, 2)} ± ${fmt(row.cv_roc_auc_se, 2)}` },
      { label: "Test accuracy", value: `${fmt(row.accuracy * 100, 0)}%`, detail: `vs ${fmt(row.nir * 100, 0)}% no-information rate · p = ${fmtP(row.p_value_acc_gt_nir)}` },
      { label: "Sensitivity / specificity", value: `${fmt(row.sens, 2)} / ${fmt(row.spec, 2)}`, detail: "remission is the positive class" },
      { label: "AUC gain from biomarkers", value: `${gain >= 0 ? "+" : "−"}${fmt(Math.abs(gain), 2)}`,
        detail: `best model ${fmt(bClin.roc_auc, 2)} → ${fmt(bBio.roc_auc, 2)} · n test = ${nTest}` }
    ]);

    // model comparison
    const met = METRICS.find((m) => m.key === st.metric);
    const refVal = met.ref === "nir" ? M[0].nir : met.ref;
    q("metric-sub").textContent = `${met.label} for each model and feature set. Hyperparameters selected by CV ${cfg.selectedBy}.` +
      (refVal != null ? ` Dashed line = ${met.refLabel} (${fmt(refVal, 2)}).` : "");
    const data = Object.entries(cfg.featureLabel).map(([fs, label], j) => {
      const rows = MODELS.map((m) => M.find((r) => r.model === m && r.feature_set === fs));
      const tr = {
        type: "bar", name: label, x: MODELS.map((m) => MODEL_LABEL[m]), y: rows.map((r) => r[met.key]),
        marker: { color: C.series[j], cornerradius: 4, line: { width: MODELS.map((m) => (m === st.model && fs === st.fs ? 2 : 0)), color: C.text } },
        hovertemplate: `<b>%{y:.3f}</b><br>%{x} · ${label}<extra></extra>`
      };
      if (met.err) tr.error_y = { type: "data", array: rows.map((r) => r[met.err]), color: C.text2, thickness: 1.2, width: 4 };
      return tr;
    });
    const lay = mergeAxis(baseLayout({ barmode: "group" }), "yaxis", { range: [0, 1.02] });
    if (refVal != null) {
      lay.shapes = [{ type: "line", xref: "x domain", x0: 0, x1: 1, y0: refVal, y1: refVal, line: { color: C.muted, dash: "dash", width: 1 } }];
    }
    plot(q("compare"), data, lay);

    // ROC
    q("roc-sub").textContent = `${fsLabel}. Dashed diagonal = chance.`;
    const roc = MODELS.map((m, i) => {
      const pts = res.roc_curves.filter((r) => r.model === m && r.feature_set === st.fs)
        .map((r) => [1 - r.specificity, r.sensitivity]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      const auc = M.find((r) => r.model === m && r.feature_set === st.fs).roc_auc;
      return { type: "scatter", mode: "lines", name: `${MODEL_LABEL[m]} (${fmt(auc, 2)})`, x: pts.map((p) => p[0]), y: pts.map((p) => p[1]),
        line: { color: C.series[i], width: m === st.model ? 3 : 2, shape: "hv" }, opacity: m === st.model ? 1 : 0.75,
        hovertemplate: `${MODEL_LABEL[m]}<br>FPR %{x:.2f} · <b>TPR %{y:.2f}</b><extra></extra>` };
    });
    const rl = baseLayout({ legend: { orientation: "v", x: 0.98, xanchor: "right", y: 0.04, yanchor: "bottom", bgcolor: C.surface, font: { color: C.text2, size: 11 } } });
    rl.shapes = [{ type: "line", x0: 0, y0: 0, x1: 1, y1: 1, line: { color: C.muted, dash: "dash", width: 1 } }];
    mergeAxis(rl, "xaxis", { range: [0, 1], title: { text: "False positive rate (1 − specificity)" }, constrain: "domain" });
    mergeAxis(rl, "yaxis", { range: [0, 1.02], title: { text: "True positive rate (sensitivity)" }, scaleanchor: "x" });
    plot(q("roc"), roc, rl);

    // importance
    const imp = res.variable_importance.filter((r) => r.model === st.model && r.feature_set === st.fs).sort((a, b) => a.Importance - b.Importance);
    const method = st.model === "rf" ? "permutation importance (ranger)" : st.model === "gbm" ? "gain (xgboost)" : "permutation importance, drop in accuracy";
    const impEl = q("importance");
    impEl.style.height = `${Math.max(300, imp.length * 20 + 90)}px`;
    q("imp-sub").textContent = `${MODEL_LABEL[st.model]} · ${fsLabel}. Scaled 0–100; ${method}.` +
      (imp.every((r) => !r.Importance) ? " This model predicts the same class for every training patient, so shuffling any variable changes nothing and all importances are 0." : "");
    const groups = [["Clinical / symptom", (v) => !BIOLOGICAL.includes(v), 0], ["Biological", (v) => BIOLOGICAL.includes(v), 1]];
    plot(impEl, groups.map(([name, f, ci]) => {
      const rows = imp.filter((r) => f(r.Variable));
      return { type: "bar", orientation: "h", name, x: rows.map((r) => r.Importance), y: rows.map((r) => r.Variable),
        marker: { color: C.series[ci], cornerradius: 4 }, hovertemplate: `<b>%{x:.1f}</b><br>%{y} · ${name}<extra></extra>` };
    }).filter((t) => t.x.length), mergeAxis(mergeAxis(baseLayout({ barmode: "overlay" }), "yaxis",
      { categoryorder: "array", categoryarray: imp.map((r) => r.Variable), dtick: 1, tickfont: { color: C.text2, size: 11 } }), "xaxis", { title: { text: "Scaled importance" } }));

    // confusion matrix
    const cm = res.confusion_matrices.filter((r) => r.model === st.model && r.feature_set === st.fs);
    const lab = { 0: "No remission", 1: "Remission" };
    const cell = (p, t) => (cm.find((r) => String(r.Prediction) === p && String(r.Truth) === t) || { n: 0 }).n;
    const z = [["1", "0"].map((t) => cell("1", t)), ["1", "0"].map((t) => cell("0", t))];
    q("cm-sub").textContent = `${MODEL_LABEL[st.model]} · ${fsLabel}. Rows = predicted, columns = actual.`;
    const zmax = Math.max(...z.flat());
    plot(q("confusion"), [{
      type: "heatmap", x: ["Actual: " + lab[1], "Actual: " + lab[0]], y: ["Pred: " + lab[1], "Pred: " + lab[0]], z,
      colorscale: [[0, C.seqLo], [1, C.seqHi]], showscale: false, xgap: 2, ygap: 2,
      hovertemplate: "%{y}<br>%{x}<br><b>%{z}</b> patients<extra></extra>"
    }], Object.assign(mergeAxis(mergeAxis(baseLayout(), "yaxis", { autorange: "reversed", tickfont: { color: C.text2, size: 12 } }),
      "xaxis", { side: "top", tickfont: { color: C.text2, size: 12 } }), {
      annotations: z.flatMap((r, i) => r.map((v, j) => ({ x: j, y: i, text: `<b>${v}</b>`, showarrow: false,
        font: { size: 18, color: v > zmax * 0.55 ? "#ffffff" : C.text } })))
    }));

    // PCA of training cohort
    const pv = res.pca_variance.filter((r) => r.feature_set === st.fs);
    q("pca-sub").textContent = `${fsLabel}; PC1 ${fmt(pv[0]["variance.percent"], 0)}%, PC2 ${fmt(pv[1]["variance.percent"], 0)}% of variance.`;
    const sc = res.pca_scores.filter((r) => r.feature_set === st.fs);
    plot(q("pca"), [[1, "Remission", 0], [0, "No remission", 1]].map(([v, name, ci]) => {
      const rows = sc.filter((r) => Number(r.response) === v);
      return { type: "scatter", mode: "markers", name, x: rows.map((r) => r.PC1), y: rows.map((r) => r.PC2),
        marker: { size: 7, color: C.series[ci], opacity: 0.8, line: { color: C.surface, width: 1 } },
        hovertemplate: `${name}<br>PC1 %{x:.2f}, PC2 %{y:.2f}<extra></extra>` };
    }), mergeAxis(mergeAxis(baseLayout({ legend: { orientation: "h", y: 1.12, x: 0, font: { color: C.text2, size: 11 } } }),
      "xaxis", { title: { text: "PC1" } }), "yaxis", { title: { text: "PC2" } }));

    // table
    const cols = [["model", "Model"], ["feature_set", "Features"], ["cv_roc_auc", "CV AUC"], ["roc_auc", "Test AUC"], ["accuracy", "Accuracy"],
      ["bal_accuracy", "Bal. acc."], ["sens", "Sens."], ["spec", "Spec."], ["ppv", "PPV"], ["npv", "NPV"], ["f_meas", "F1"], ["kap", "Kappa"], ["p_value_acc_gt_nir", "p (acc > NIR)"]];
    const t = q("table");
    const thead = el("thead"), hr = el("tr");
    cols.forEach(([, h]) => hr.append(el("th", { scope: "col" }, h)));
    thead.append(hr);
    const tb = el("tbody");
    const bestAuc = Math.max(...M.map((r) => r.roc_auc));
    [...M].sort((a, b) => a.feature_set.localeCompare(b.feature_set) || MODELS.indexOf(a.model) - MODELS.indexOf(b.model)).forEach((r) => {
      const tr = el("tr", r.roc_auc === bestAuc ? { class: "best" } : {});
      cols.forEach(([k]) => {
        const td = el("td");
        if (k === "model") {
          const dot = el("span", { class: "dot" }); dot.style.background = C.series[MODELS.indexOf(r.model)];
          td.append(dot, document.createTextNode(MODEL_LABEL[r.model]));
        } else if (k === "feature_set") td.textContent = cfg.featureLabel[r.feature_set];
        else if (k === "p_value_acc_gt_nir") td.textContent = fmtP(r[k]);
        else td.textContent = fmt(r[k]);
        tr.append(td);
      });
      tb.append(tr);
    });
    t.replaceChildren(thead, tb);
  }

  /* ---------- tabs & boot ---------- */
  const renderers = { overview: renderOverview, unsupervised: renderUnsupervised, ssri: () => renderSupervised("ssri"), ssri_symptoms: () => renderSupervised("ssri_symptoms") };
  let current = "overview";
  function show(tab) {
    if (!renderers[tab]) tab = "overview";
    current = tab;
    document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
    document.querySelectorAll(".tab-panel").forEach((p) => { p.hidden = p.id !== `tab-${tab}`; });
    renderers[tab]();
    if (location.hash.slice(1) !== tab) history.replaceState(null, "", `#${tab}`);
  }
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
  window.addEventListener("hashchange", () => show(location.hash.slice(1)));
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readTheme(); renderers[current](); });

  if (!R) { document.querySelector("main").textContent = "dashboard/results.js not found - run Rscript R/build_dashboard_data.R"; return; }
  readTheme();
  document.getElementById("generated").textContent = `Results generated ${R.generated}.`;
  show(location.hash.slice(1) || "overview");
})();
