/**
 * Main Application Orchestrator
 */

import { API } from "./api.js";
import { initTumorAnalyzer } from "./tumor_analyzer.js";
import { initStrokePrognosis } from "./stroke_prognosis.js";
import { initQuantumLab } from "./quantum_lab.js";
import { renderConfusionMatrix } from "./charts.js";

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initFontSwitcher();
  initTumorAnalyzer();
  initStrokePrognosis();
  initQuantumLab();
  loadBenchmarks();
});

function initFontSwitcher() {
  const fontSelect = document.getElementById("fontThemeSelect");
  if (!fontSelect) return;

  const fontMap = {
    "manrope": "'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "dm-sans": "'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "figtree": "'Figtree', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "inter": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "system": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
  };

  const savedFont = localStorage.getItem("qai_font_theme") || "manrope";
  fontSelect.value = savedFont;
  applyFont(savedFont);

  fontSelect.addEventListener("change", (e) => {
    const val = e.target.value;
    localStorage.setItem("qai_font_theme", val);
    applyFont(val);
  });

  function applyFont(key) {
    const fontStr = fontMap[key] || fontMap["manrope"];
    document.documentElement.style.setProperty("--font-heading", fontStr);
    document.documentElement.style.setProperty("--font-body", fontStr);
  }
}

function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  const views = document.querySelectorAll(".view-section");

  navItems.forEach(item => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      const targetView = item.dataset.view;

      navItems.forEach(n => n.classList.remove("active"));
      item.classList.add("active");

      views.forEach(v => {
        if (v.id === targetView) {
          v.classList.add("active");
        } else {
          v.classList.remove("active");
        }
      });
    });
  });
}

async function loadBenchmarks() {
  try {
    const data = await API.getBenchmarkMetrics();
    
    // 1. Populate Tumor Benchmark Table
    const tumorLive = data.live_trained_tumor_metrics;
    const tumorBody = document.getElementById("tumorBenchmarkBody");
    if (tumorBody && tumorLive) {
      tumorBody.innerHTML = "";
      // Find top model by accuracy
      let topTumorModel = "";
      let topTumorAcc = -1;

      Object.entries(tumorLive).forEach(([name, m]) => {
        if (m.accuracy > topTumorAcc) {
          topTumorAcc = m.accuracy;
          topTumorModel = name;
        }
      });

      function formatModelName(name) {
        if (name.includes("CNN") || name.includes("Convolutional")) return "CNN";
        if (name.includes("HQNN") || name.includes("Quantum")) return "HQNN";
        if (name.includes("Random Forest")) return "Random Forest";
        if (name.includes("Decision Tree")) return "Decision Tree";
        if (name.includes("Logistic")) return "Logistic Regression";
        if (name.includes("Nearest Neighbors") || name.includes("KNN")) return "KNN";
        if (name.includes("Support Vector") || name.includes("SVM")) return "SVM";
        if (name.includes("Naive Bayes") || name.includes("GNB")) return "Naive Bayes";
        if (name.includes("QSVC")) return "QSVC";
        return name;
      }

      // Update overview tile
      const ovTumorAcc = document.getElementById("overviewTumorAcc");
      const ovTumorLabel = document.getElementById("overviewTumorLabel");
      if (ovTumorAcc) ovTumorAcc.textContent = `${topTumorAcc.toFixed(2)}%`;
      if (ovTumorLabel) ovTumorLabel.textContent = `Brain Tumor (${formatModelName(topTumorModel)})`;

      const tumorBadge = document.getElementById("tumorTopBadge");
      if (tumorBadge) tumorBadge.textContent = `Top Model: ${topTumorModel} (${topTumorAcc.toFixed(2)}%)`;

      Object.entries(tumorLive).forEach(([name, m]) => {
        const tr = document.createElement("tr");
        const isTop = name === topTumorModel;
        if (isTop) tr.className = "highlight-top";

        tr.innerHTML = `
          <td><strong>${name}</strong> ${isTop ? '<span class="badge-top-algo" style="margin-left: 6px;">Top Performer</span>' : ''}</td>
          <td><strong>${m.accuracy.toFixed(2)}%</strong></td>
          <td>${m.precision.toFixed(2)}%</td>
          <td>${m.recall.toFixed(2)}%</td>
          <td>${m.f1_score.toFixed(2)}%</td>
          <td>${m.roc_auc ? m.roc_auc.toFixed(4) : '--'}</td>
        `;
        tumorBody.appendChild(tr);
      });
    }

    // 2. Populate Stroke Benchmark Table
    const strokeLive = data.live_trained_stroke_metrics;
    const strokeBody = document.getElementById("strokeBenchmarkBody");
    if (strokeBody && strokeLive) {
      strokeBody.innerHTML = "";
      let topStrokeModel = "";
      let topStrokeAcc = -1;

      Object.entries(strokeLive).forEach(([name, m]) => {
        if (m.accuracy > topStrokeAcc) {
          topStrokeAcc = m.accuracy;
          topStrokeModel = name;
        }
      });

      function formatStrokeName(name) {
        if (name.includes("Logistic")) return "Logistic Regression";
        if (name.includes("Nearest Neighbors") || name.includes("KNN")) return "KNN";
        if (name.includes("Support Vector") || name.includes("SVM")) return "SVM";
        if (name.includes("Random Forest")) return "Random Forest";
        if (name.includes("Decision Tree")) return "Decision Tree";
        if (name.includes("Naive Bayes") || name.includes("GNB")) return "Naive Bayes";
        if (name.includes("QSVC")) return "QSVC";
        return name;
      }

      // Update overview tile
      const ovStrokeAcc = document.getElementById("overviewStrokeAcc");
      const ovStrokeLabel = document.getElementById("overviewStrokeLabel");
      if (ovStrokeAcc) ovStrokeAcc.textContent = `${topStrokeAcc.toFixed(2)}%`;
      if (ovStrokeLabel) ovStrokeLabel.textContent = `Brain Stroke (${formatStrokeName(topStrokeModel)})`;

      const strokeBadge = document.getElementById("strokeTopBadge");
      if (strokeBadge) strokeBadge.textContent = `Top Model: ${topStrokeModel} (${topStrokeAcc.toFixed(2)}%)`;

      Object.entries(strokeLive).forEach(([name, m]) => {
        const tr = document.createElement("tr");
        const isTop = name === topStrokeModel;
        if (isTop) tr.className = "highlight-top";

        tr.innerHTML = `
          <td><strong>${name}</strong> ${isTop ? '<span class="badge-top-algo" style="margin-left: 6px;">Top Performer</span>' : ''}</td>
          <td><strong>${m.accuracy.toFixed(2)}%</strong></td>
          <td>${m.precision.toFixed(2)}%</td>
          <td>${m.recall.toFixed(2)}%</td>
          <td>${m.f1_score.toFixed(2)}%</td>
          <td>${m.roc_auc ? m.roc_auc.toFixed(4) : '--'}</td>
        `;
        strokeBody.appendChild(tr);
      });
    }

    // 3. Render Confusion Matrices from live test splits
    if (tumorLive && tumorLive["Convolutional Neural Network (CNN)"]) {
      renderConfusionMatrix("tumorCnnCmCanvas", tumorLive["Convolutional Neural Network (CNN)"].confusion_matrix);
    }

    if (strokeLive && strokeLive["Logistic Regression"]) {
      renderConfusionMatrix("strokeLrCmCanvas", strokeLive["Logistic Regression"].confusion_matrix);
    }

  } catch (err) {
    console.error("Failed to load benchmarks:", err);
  }
}
