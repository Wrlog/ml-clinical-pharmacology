/* Interactive figures (plotly.js). Shares the design tokens in style.css and the
   palette used by python/make_figures.py, so the interactive and static versions
   of a figure look the same.

   Every builder degrades gracefully: if plotly.js did not load (offline, blocked
   CDN), charts.ready is false and app.js falls back to the pre-rendered SVG. */
(function () {
  "use strict";

  /* ---------- tokens (mirror :root in style.css / make_figures.py) ---------- */
  const SURFACE = "#fcfcfb";
  const INK = "#0b0b0b", INK2 = "#52514e", MUTED = "#898781";
  const GRID = "#e1e0d9", AXIS = "#c3c2b7";
  const BLUE = "#2a78d6", ORANGE = "#eb6834", AQUA = "#1baf7a", YELLOW = "#eda100", MAGENTA = "#e87ba4";
  const SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA];
  const FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif';

  // Diagnosis is encoded by shape as well as position, so cluster colour is never
  // the only thing telling two groups apart.
  const DIAG_ORDER = ["Healthy Controls", "Mild", "Moderate", "Treatment Resistant"];
  const DIAG_SYMBOL = { "Healthy Controls": "circle", Mild: "square", Moderate: "diamond", "Treatment Resistant": "triangle-up" };

  const ready = typeof window.Plotly !== "undefined";

  /* ---------- shared layout / config ---------- */
  const axis = (title) => ({
    title: { text: title, font: { size: 11.5, color: INK2 }, standoff: 8 },
    gridcolor: GRID, gridwidth: 1,
    zeroline: false,
    showline: true, linecolor: AXIS, linewidth: 1, mirror: false,
    ticks: "", tickfont: { size: 11, color: INK2 },
    automargin: true
  });

  const baseLayout = (xTitle, yTitle) => ({
    paper_bgcolor: SURFACE,
    plot_bgcolor: SURFACE,
    font: { family: FONT, size: 11, color: INK },
    xaxis: axis(xTitle),
    yaxis: axis(yTitle),
    margin: { l: 8, r: 8, t: 8, b: 8 },
    hoverlabel: {
      bgcolor: SURFACE, bordercolor: GRID, align: "left",
      font: { family: FONT, size: 12, color: INK }
    },
    hovermode: "closest",
    showlegend: true,
    dragmode: "pan"
  });

  const config = (filename) => ({
    displaylogo: false,
    responsive: true,
    scrollZoom: false,
    displayModeBar: "hover",
    modeBarButtonsToRemove: ["lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "toggleSpikelines"],
    toImageButtonOptions: { format: "png", filename, scale: 2 }
  });

  /* Draw into `node`, reusing the existing plot when the shape of the figure is
     unchanged so that a selector change animates instead of flashing. */
  function render(node, traces, layout, filename) {
    const fn = node.dataset.plotted ? window.Plotly.react : window.Plotly.newPlot;
    node.dataset.plotted = "1";
    return fn(node, traces, layout, config(filename));
  }

  /* ---------- 1. unsupervised: PCA scatter, colour = cluster ---------- */
  function unsupervisedPCA(node, assignments, variance, clusterKey, clusterLabel) {
    const pc1 = variance[0]["variance.percent"], pc2 = variance[1]["variance.percent"];
    const present = new Set(assignments.map((r) => r.Diagnosis));
    const diags = DIAG_ORDER.filter((d) => present.has(d)).concat(
      [...present].filter((d) => !DIAG_ORDER.includes(d)).sort()
    );
    const clusters = [...new Set(assignments.map((r) => r[clusterKey]))].sort((a, b) => a - b);

    const traces = clusters.map((c, i) => {
      const rows = assignments.filter((r) => r[clusterKey] === c);
      return {
        type: "scatter",
        mode: "markers",
        name: `Cluster ${c}`,
        x: rows.map((r) => r.PC1),
        y: rows.map((r) => r.PC2),
        customdata: rows.map((r) => [r.ID, r.Diagnosis]),
        marker: {
          color: SERIES[i % SERIES.length],
          size: 11,
          symbol: rows.map((r) => DIAG_SYMBOL[r.Diagnosis] || "circle"),
          // 2px surface ring keeps overlapping subjects readable
          line: { color: SURFACE, width: 2 }
        },
        hovertemplate:
          "<b>Subject %{customdata[0]}</b><br>" +
          `Cluster ${c}<br>` +
          "Diagnosis: %{customdata[1]}<br>" +
          "PC1 %{x:.2f} · PC2 %{y:.2f}<extra></extra>"
      };
    });

    const layout = baseLayout(`PC1 (${pc1.toFixed(0)}% of variance)`, `PC2 (${pc2.toFixed(0)}% of variance)`);
    layout.margin = { l: 8, r: 8, t: 8, b: 8 };
    layout.legend = {
      title: { text: "Cluster", font: { size: 11, color: INK2 } },
      x: 1.02, xanchor: "left", y: 1, yanchor: "top",
      bgcolor: "rgba(0,0,0,0)", borderwidth: 0,
      font: { size: 11.5, color: INK2 },
      itemsizing: "constant", itemclick: "toggle", itemdoubleclick: "toggleothers"
    };
    return render(node, traces, layout, `pca_${clusterKey}`).then(() => diags);
  }

  /* ---------- 2 & 3. supervised: PCA of the training cohort, colour = outcome ---------- */
  function supervisedPCA(node, scores, variance, featureSet, key) {
    const s = scores.filter((r) => r.feature_set === featureSet);
    const v = variance.filter((r) => r.feature_set === featureSet);
    const pc1 = v[0]["variance.percent"], pc2 = v[1]["variance.percent"];

    const traces = [
      { resp: 0, name: "No remission", color: ORANGE },
      { resp: 1, name: "Remission", color: BLUE }
    ].map(({ resp, name, color }) => {
      const rows = s.filter((r) => r.response === resp);
      return {
        type: "scatter",
        mode: "markers",
        name: `${name} (n = ${rows.length})`,
        x: rows.map((r) => r.PC1),
        y: rows.map((r) => r.PC2),
        marker: { color, size: 7, opacity: 0.8, line: { color: SURFACE, width: 1 } },
        hovertemplate: `<b>${name}</b><br>PC1 %{x:.2f} · PC2 %{y:.2f}<extra></extra>`
      };
    });

    const layout = baseLayout(`PC1 (${pc1.toFixed(0)}% of variance)`, `PC2 (${pc2.toFixed(0)}%)`);
    layout.legend = {
      orientation: "h", x: 0.5, xanchor: "center", y: 1.14, yanchor: "top",
      bgcolor: "rgba(0,0,0,0)", borderwidth: 0,
      font: { size: 11.5, color: INK2 }, itemsizing: "constant"
    };
    layout.margin = { l: 8, r: 8, t: 26, b: 8 };
    return render(node, traces, layout, `${key}_pca_${featureSet}`);
  }

  /* ---------- shape key (rendered as HTML so it can sit under the plot) ---------- */
  function diagnosisKey(container, diags) {
    const ns = "http://www.w3.org/2000/svg";
    const glyph = {
      circle: "<circle cx='7' cy='7' r='5'/>",
      square: "<rect x='2.2' y='2.2' width='9.6' height='9.6'/>",
      diamond: "<path d='M7 1.4 12.6 7 7 12.6 1.4 7Z'/>",
      "triangle-up": "<path d='M7 1.8 12.8 12.2H1.2Z'/>"
    };
    container.replaceChildren(...diags.map((d) => {
      const item = document.createElement("span");
      item.className = "key-item";
      const svg = document.createElementNS(ns, "svg");
      svg.setAttribute("viewBox", "0 0 14 14");
      svg.setAttribute("width", "13");
      svg.setAttribute("height", "13");
      svg.setAttribute("aria-hidden", "true");
      svg.innerHTML = glyph[DIAG_SYMBOL[d] || "circle"];
      item.append(svg, document.createTextNode(d));
      return item;
    }));
  }

  window.CHARTS = { ready, unsupervisedPCA, supervisedPCA, diagnosisKey, SERIES, DIAG_ORDER, DIAG_SYMBOL };
})();
