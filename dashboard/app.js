/* Dashboard: figures are pre-rendered SVGs (python/make_figures.py);
   tiles and tables read window.RESULTS (dashboard/results.js). */
(function () {
  "use strict";

  const R = window.RESULTS;
  const FIG = "dashboard/figures";
  const MODELS = ["rf", "svm_linear", "svm_rbf", "gbm"];
  const MODEL_LABEL = { rf: "Random forest", svm_linear: "SVM (linear)", svm_rbf: "SVM (RBF)", gbm: "Gradient boosting" };
  const MODEL_COLOR = { rf: "#2a78d6", svm_linear: "#eb6834", svm_rbf: "#1baf7a", gbm: "#eda100" };
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
    ["roc_auc", "Test ROC AUC"],
    ["cv_roc_auc", "Cross-validated ROC AUC"],
    ["accuracy", "Test accuracy"],
    ["bal_accuracy", "Test balanced accuracy"],
    ["sens", "Sensitivity (remission)"],
    ["spec", "Specificity"],
    ["ppv", "Positive predictive value"],
    ["f_meas", "F1 score"]
  ];
  const IMPORTANCE_METHOD = {
    rf: "permutation importance (ranger)",
    gbm: "gain (xgboost)",
    svm_linear: "permutation importance, drop in accuracy",
    svm_rbf: "permutation importance, drop in accuracy"
  };
  const clusterLabel = (k) => {
    let m;
    if ((m = k.match(/^kmeans_(\d+)$/))) return `k-means, k = ${m[1]}`;
    if ((m = k.match(/^hclust_(\w+)_(\d+)$/))) return `Hierarchical (${m[1]}), k = ${m[2]}`;
    if (k === "gmm_auto") return "GMM, BIC-selected";
    if ((m = k.match(/^gmm_(\d+)$/))) return `GMM, G = ${m[1]}`;
    return k;
  };

  const fmt = (v, d = 3) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
  const fmtP = (p) => (p == null ? "–" : p < 0.001 ? "<0.001" : Number(p).toFixed(3));
  const el = (tag, attrs = {}, text) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    return n;
  };
  function tiles(container, items) {
    container.replaceChildren(...items.map((t) => {
      const d = el("div", { class: "tile" });
      d.append(el("div", { class: "label" }, t.label), el("div", { class: "value" }, t.value));
      if (t.detail) d.append(el("div", { class: "detail" }, t.detail));
      return d;
    }));
  }
  function setImg(img, src, alt) {
    if (img.getAttribute("src") !== src) img.setAttribute("src", src);
    if (alt) img.setAttribute("alt", alt);
  }

  /* ---------- overview ---------- */
  function renderOverview() {
    const ari = R.unsupervised.cluster_diagnosis_agreement;
    const perfect = ari.filter((r) => r.adjusted_rand_index > 0.999).length;
    const items = [{ label: "Clustering vs diagnosis", value: `${perfect} / ${ari.length}`,
      detail: "clusterings recover diagnosis perfectly (ARI = 1)" }];
    for (const [key, cfg] of Object.entries(ANALYSES)) {
      const m = R[key].metrics;
      const clin = [...m].filter((r) => r.feature_set === "clinical").sort((a, b) => b.roc_auc - a.roc_auc)[0];
      const bio = [...m].filter((r) => r.feature_set === "clinical_biomarkers").sort((a, b) => b.roc_auc - a.roc_auc)[0];
      items.push({ label: `Best test AUC · ${cfg.title}`, value: `${fmt(clin.roc_auc, 2)} → ${fmt(bio.roc_auc, 2)}`,
        detail: `${cfg.featureLabel.clinical.toLowerCase()} → with biomarkers` });
    }
    tiles(document.getElementById("overview-tiles"), items);
  }

  /* ---------- 1. unsupervised ---------- */
  function renderUnsupervised() {
    const u = R.unsupervised;
    const ari = [...u.cluster_diagnosis_agreement].sort((a, b) => b.adjusted_rand_index - a.adjusted_rand_index);
    const pv = u.pca_variance;
    const gmmK = new Set(u.cluster_assignments.map((r) => r.gmm_auto)).size;
    tiles(document.getElementById("unsup-tiles"), [
      { label: "Subjects × genes", value: `${u.cluster_assignments.length} × ${u.pca_loadings.length}`, detail: "4 diagnostic groups" },
      { label: "PC1 + PC2 variance", value: `${fmt(pv[1]["cumulative.variance.percent"], 0)}%`, detail: "of total variance on the first two PCs" },
      { label: "GMM (BIC) chooses", value: `${gmmK} clusters`, detail: "same as the number of diagnoses" },
      { label: "Best ARI", value: fmt(ari[0].adjusted_rand_index, 2), detail: "k-means k = 4, all hierarchical linkages, GMM" }
    ]);
    const sel = document.getElementById("unsup-cluster-select");
    const img = document.getElementById("unsup-pca");
    if (!sel.options.length) {
      ari.forEach((r) => sel.append(el("option", { value: r.clustering }, clusterLabel(r.clustering))));
      sel.value = "gmm_auto";
      sel.addEventListener("change", () => setImg(img, `${FIG}/unsupervised/pca_${sel.value}.svg`,
        `PCA scatter coloured by ${clusterLabel(sel.value)}`));
    }
    setImg(img, `${FIG}/unsupervised/pca_${sel.value}.svg`, `PCA scatter coloured by ${clusterLabel(sel.value)}`);
  }

  /* ---------- 2 & 3. supervised ---------- */
  const state = {};
  function renderSupervised(key) {
    const res = R[key], cfg = ANALYSES[key];
    const root = document.getElementById(`tab-${key}`);
    if (!root.children.length) {
      root.append(document.getElementById("supervised-template").content.cloneNode(true));
      state[key] = { fs: "clinical_biomarkers", model: [...res.metrics].sort((a, b) => b.roc_auc - a.roc_auc)[0].model, metric: "roc_auc" };
      const q = (r) => root.querySelector(`[data-role="${r}"]`);
      const seg = q("feature-set");
      for (const [fs, label] of Object.entries(cfg.featureLabel)) {
        const b = el("button", { role: "radio", "aria-checked": String(fs === state[key].fs) }, label);
        b.addEventListener("click", () => {
          state[key].fs = fs;
          seg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
          draw(key);
        });
        seg.append(b);
      }
      const ms = q("model");
      MODELS.forEach((m) => ms.append(el("option", { value: m }, MODEL_LABEL[m])));
      ms.value = state[key].model;
      ms.addEventListener("change", () => { state[key].model = ms.value; draw(key); });
      const mt = q("metric");
      METRICS.forEach(([k, label]) => mt.append(el("option", { value: k }, label)));
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
    const base = `${FIG}/${key}`;
    const bestOf = (fs) => M.filter((r) => r.feature_set === fs).sort((a, b) => b.roc_auc - a.roc_auc)[0];
    const bClin = bestOf("clinical"), bBio = bestOf("clinical_biomarkers");
    const gain = bBio.roc_auc - bClin.roc_auc;
    const nTest = res.confusion_matrices.filter((r) => r.model === st.model && r.feature_set === st.fs).reduce((s, r) => s + r.n, 0);

    tiles(q("tiles"), [
      { label: `${MODEL_LABEL[st.model]} · test AUC`, value: fmt(row.roc_auc, 2), detail: `${fsLabel} · CV ${fmt(row.cv_roc_auc, 2)} ± ${fmt(row.cv_roc_auc_se, 2)}` },
      { label: "Test accuracy", value: `${fmt(row.accuracy * 100, 0)}%`, detail: `vs ${fmt(row.nir * 100, 0)}% no-information rate · p = ${fmtP(row.p_value_acc_gt_nir)}` },
      { label: "Sensitivity / specificity", value: `${fmt(row.sens, 2)} / ${fmt(row.spec, 2)}`, detail: "remission is the positive class" },
      { label: "AUC gain from biomarkers", value: `${gain >= 0 ? "+" : "−"}${fmt(Math.abs(gain), 2)}`,
        detail: `best model ${fmt(bClin.roc_auc, 2)} → ${fmt(bBio.roc_auc, 2)} · n test = ${nTest}` }
    ]);

    const metLabel = METRICS.find(([k]) => k === st.metric)[1];
    q("metric-sub").textContent = `${metLabel} for each model and feature set. Hyperparameters selected by CV ${cfg.selectedBy}.`;
    setImg(q("compare"), `${base}/compare_${st.metric}.svg`, `${metLabel} by model and feature set`);

    q("roc-sub").textContent = `${fsLabel}. Dashed diagonal = chance.`;
    setImg(q("roc"), `${base}/roc_${st.fs}.svg`, `ROC curves, ${fsLabel}`);

    q("imp-sub").textContent = `${MODEL_LABEL[st.model]} · ${fsLabel}. Scaled 0–100; ${IMPORTANCE_METHOD[st.model]}.`;
    setImg(q("importance"), `${base}/importance_${st.model}_${st.fs}.svg`, `Variable importance, ${MODEL_LABEL[st.model]}, ${fsLabel}`);

    q("cm-sub").textContent = `${MODEL_LABEL[st.model]} · ${fsLabel}. Percentages are of each actual class.`;
    setImg(q("confusion"), `${base}/confusion_${st.model}_${st.fs}.svg`, `Confusion matrix, ${MODEL_LABEL[st.model]}, ${fsLabel}`);

    q("pca-sub").textContent = `${fsLabel}. Coloured by observed outcome.`;
    setImg(q("pca"), `${base}/pca_${st.fs}.svg`, `PCA of training cohort, ${fsLabel}`);

    // metrics table
    const cols = [["model", "Model"], ["feature_set", "Features"], ["cv_roc_auc", "CV AUC"], ["roc_auc", "Test AUC"], ["accuracy", "Accuracy"],
      ["bal_accuracy", "Bal. acc."], ["sens", "Sens."], ["spec", "Spec."], ["ppv", "PPV"], ["npv", "NPV"], ["f_meas", "F1"],
      ["kap", "Kappa"], ["p_value_acc_gt_nir", "p (acc > NIR)"]];
    const t = q("table");
    const thead = el("thead"), hr = el("tr");
    cols.forEach(([, h]) => hr.append(el("th", { scope: "col" }, h)));
    thead.append(hr);
    const tb = el("tbody");
    [...M].sort((a, b) => a.feature_set.localeCompare(b.feature_set) || MODELS.indexOf(a.model) - MODELS.indexOf(b.model)).forEach((r) => {
      const selected = r.model === st.model && r.feature_set === st.fs;
      const tr = el("tr", selected ? { class: "selected" } : {});
      cols.forEach(([k]) => {
        const td = el("td");
        if (k === "model") {
          const dot = el("span", { class: "dot" });
          dot.style.background = MODEL_COLOR[r.model];
          td.append(dot, document.createTextNode(MODEL_LABEL[r.model]));
        } else if (k === "feature_set") td.textContent = cfg.featureLabel[r.feature_set];
        else if (k === "cv_roc_auc") td.textContent = `${fmt(r.cv_roc_auc)} ± ${fmt(r.cv_roc_auc_se)}`;
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
  function show(tab) {
    if (!renderers[tab]) tab = "overview";
    document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
    document.querySelectorAll(".tab-panel").forEach((p) => { p.hidden = p.id !== `tab-${tab}`; });
    renderers[tab]();
    if (location.hash.slice(1) !== tab) history.replaceState(null, "", `#${tab}`);
  }
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
  window.addEventListener("hashchange", () => show(location.hash.slice(1)));

  if (!R) { document.querySelector("main").textContent = "dashboard/results.js not found - run Rscript R/build_dashboard_data.R"; return; }
  document.getElementById("generated").textContent = `Results generated ${R.generated}.`;
  show(location.hash.slice(1) || "overview");
})();
