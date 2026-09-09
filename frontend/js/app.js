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
    
    // Populate dataset summary counts
    if (data.dataset_summary) {
      const mriCountEl = document.getElementById("overviewMriCount");
      const ptCountEl = document.getElementById("overviewPatientCount");
      if (mriCountEl && data.dataset_summary.brain_tumor_mri_total) {
        mriCountEl.textContent = data.dataset_summary.brain_tumor_mri_total.toLocaleString();
      }
      if (ptCountEl && data.dataset_summary.stroke_records_total) {
        ptCountEl.textContent = data.dataset_summary.stroke_records_total.toLocaleString();
      }
    }

    // 1. Populate Tumor Benchmark Table (Project Empirical vs Base1.pdf Paper Table 9)
    const tumorLive = data.live_trained_tumor_metrics;
    const tumorPaper = data.table_9_brain_tumor ? data.table_9_brain_tumor.models : {};
    const tumorBody = document.getElementById("tumorBenchmarkBody");

    if (tumorBody && tumorLive) {
      tumorBody.innerHTML = "";
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
        return name;
      }

      // Update overview tile
      const ovTumorAcc = document.getElementById("overviewTumorAcc");
      const ovTumorLabel = document.getElementById("overviewTumorLabel");
      if (ovTumorAcc) ovTumorAcc.textContent = `${topTumorAcc.toFixed(2)}%`;
      if (ovTumorLabel) ovTumorLabel.textContent = `Brain Tumor (${formatModelName(topTumorModel)})`;

      const tumorBadge = document.getElementById("tumorTopBadge");
      if (tumorBadge) tumorBadge.textContent = `Top Performer: ${topTumorModel} (${topTumorAcc.toFixed(2)}%)`;

      Object.entries(tumorLive).forEach(([name, m]) => {
        const tr = document.createElement("tr");
        const isTop = name === topTumorModel;
        if (isTop) tr.className = "highlight-top";

        // Find paper target matching entry
        const paperEntry = tumorPaper[name] || {};
        const paperAcc = paperEntry.accuracy ? paperEntry.accuracy.toFixed(2) + "%" : "--";
        const paperF1 = paperEntry.f1_score ? paperEntry.f1_score.toFixed(2) + "%" : "--";
        const variance = paperEntry.accuracy ? (m.accuracy - paperEntry.accuracy).toFixed(2) : "--";
        const varColor = parseFloat(variance) < 0 ? "#EF4444" : "#10B981";

        tr.innerHTML = `
          <td><strong>${name}</strong> ${isTop ? '<span class="badge-top-algo" style="margin-left: 6px;">Top Performer</span>' : ''}</td>
          <td><strong style="color: var(--accent-cyan);">${m.accuracy.toFixed(2)}%</strong></td>
          <td><strong style="color: #64748B;">${paperAcc}</strong></td>
          <td>${m.f1_score.toFixed(2)}%</td>
          <td>${paperF1}</td>
          <td>${m.roc_auc ? m.roc_auc.toFixed(4) : '--'}</td>
          <td><span style="font-weight: 700; color: ${varColor};">${variance > 0 ? '+' : ''}${variance}%</span></td>
        `;
        tumorBody.appendChild(tr);
      });
    }

    // 2. Populate Stroke Benchmark Table (Project Empirical vs Base1.pdf Paper Table 10)
    const strokeLive = data.live_trained_stroke_metrics;
    const strokePaper = data.table_10_brain_stroke ? data.table_10_brain_stroke.models : {};
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
      if (strokeBadge) strokeBadge.textContent = `Top Performer: ${topStrokeModel} (${topStrokeAcc.toFixed(2)}%)`;

      Object.entries(strokeLive).forEach(([name, m]) => {
        const tr = document.createElement("tr");
        const isTop = name === topStrokeModel;
        if (isTop) tr.className = "highlight-top";

        const paperEntry = strokePaper[name] || {};
        const paperAcc = paperEntry.accuracy ? paperEntry.accuracy.toFixed(2) + "%" : "--";
        const paperF1 = paperEntry.f1_score ? paperEntry.f1_score.toFixed(2) + "%" : "--";
        const variance = paperEntry.accuracy ? (m.accuracy - paperEntry.accuracy).toFixed(2) : "--";
        const varColor = parseFloat(variance) < 0 ? "#EF4444" : "#10B981";

        tr.innerHTML = `
          <td><strong>${name}</strong> ${isTop ? '<span class="badge-top-algo" style="margin-left: 6px;">Top Performer</span>' : ''}</td>
          <td><strong style="color: var(--accent-purple);">${m.accuracy.toFixed(2)}%</strong></td>
          <td><strong style="color: #64748B;">${paperAcc}</strong></td>
          <td>${m.f1_score.toFixed(2)}%</td>
          <td>${paperF1}</td>
          <td>${m.roc_auc ? m.roc_auc.toFixed(4) : '--'}</td>
          <td><span style="font-weight: 700; color: ${varColor};">${variance > 0 ? '+' : ''}${variance}%</span></td>
        `;
        strokeBody.appendChild(tr);
      });
    }

    // 3. Render Confusion Matrices from live test splits
    if (tumorLive) {
      const topModel = tumorLive["Random Forest"] || tumorLive["Convolutional Neural Network (CNN)"] || Object.values(tumorLive)[0];
      if (topModel && topModel.confusion_matrix) {
        renderConfusionMatrix("tumorCnnCmCanvas", topModel.confusion_matrix, ["Healthy", "Tumor"]);
      }
    }

    if (strokeLive) {
      const topStroke = strokeLive["K Nearest Neighbors"] || strokeLive["Random Forest"] || strokeLive["Logistic Regression"] || Object.values(strokeLive)[0];
      if (topStroke && topStroke.confusion_matrix) {
        renderConfusionMatrix("strokeLrCmCanvas", topStroke.confusion_matrix, ["No Stroke", "Stroke"]);
      }
    }

  } catch (err) {
    console.error("Failed to load benchmarks:", err);
  }
}
