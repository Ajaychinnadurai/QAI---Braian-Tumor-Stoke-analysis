/**
 * Brain Tumor MRI Diagnostic Center Logic
 * Supports real Kaggle MRI scans, Quantum filter previewing, and CNN/HQNN inference.
 */

import { API } from "./api.js";

let currentFilteredImages = null;
let currentActiveFilter = "original";

export function initTumorAnalyzer() {
  const dropzone = document.getElementById("mriDropzone");
  const fileInput = document.getElementById("mriFileInput");
  const analyzeBtn = document.getElementById("btnAnalyzeMri");
  const filterTabs = document.querySelectorAll(".filter-tab-btn");

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelection(e.target.files[0]);
      }
    });
  }

  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", runTumorAnalysis);
  }

  filterTabs.forEach(tab => {
    tab.addEventListener("click", (e) => {
      filterTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      currentActiveFilter = tab.dataset.filter;
      updateFilterPreview();
    });
  });

  loadSampleMriPresets();
}

function handleFileSelection(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const base64Str = e.target.result;
    document.getElementById("previewImage").src = base64Str;
    document.getElementById("previewImage").dataset.base64 = base64Str;
    document.getElementById("previewImage").dataset.sampleName = "";
    document.getElementById("selectedMriName").textContent = file.name;
    document.getElementById("mriPlaceholderText").style.display = "none";
    document.getElementById("previewImage").style.display = "block";
    resetFilterTabs();
  };
  reader.readAsDataURL(file);
}

export function selectSampleMri(filename, type = "healthy") {
  const imgUrl = `/data/brain_tumor/${type}/${filename}`;
  const previewImg = document.getElementById("previewImage");
  previewImg.src = imgUrl;
  previewImg.dataset.sampleName = filename;
  previewImg.dataset.sampleType = type;
  previewImg.dataset.base64 = "";
  
  const prettyType = type.charAt(0).toUpperCase() + type.slice(1);
  document.getElementById("selectedMriName").textContent = `Real MRI: ${filename} (${prettyType})`;
  document.getElementById("mriPlaceholderText").style.display = "none";
  previewImg.style.display = "block";
  resetFilterTabs();
  runTumorAnalysis();
}

async function loadSampleMriPresets() {
  try {
    const samples = await API.getDatasetSamples();
    const config = [
      { id: "mriGbmPresets", key: "glioblastoma_scans", type: "glioblastoma" },
      { id: "mriMeningiomaPresets", key: "meningioma_scans", type: "meningioma" },
      { id: "mriPituitaryPresets", key: "pituitary_scans", type: "pituitary" },
      { id: "mriAstrocytomaPresets", key: "astrocytoma_scans", type: "astrocytoma" },
      { id: "mriHealthyPresets", key: "healthy_scans", type: "healthy" }
    ];

    config.forEach(({ id, key, type }) => {
      const container = document.getElementById(id);
      if (container && samples.mri_samples && samples.mri_samples[key]) {
        container.innerHTML = "";
        samples.mri_samples[key].slice(0, 4).forEach(file => {
          const chip = document.createElement("button");
          chip.className = "preset-chip";
          chip.textContent = file;
          chip.onclick = () => selectSampleMri(file, type);
          container.appendChild(chip);
        });
      }
    });
  } catch (err) {
    console.error("Failed loading MRI samples:", err);
  }
}

function resetFilterTabs() {
  currentFilteredImages = null;
  currentActiveFilter = "original";
  document.querySelectorAll(".filter-tab-btn").forEach(t => t.classList.remove("active"));
  const origTab = document.querySelector('.filter-tab-btn[data-filter="original"]');
  if (origTab) origTab.classList.add("active");
}

function updateFilterPreview() {
  if (!currentFilteredImages) return;
  const previewImg = document.getElementById("previewImage");
  if (currentActiveFilter === "original") {
    previewImg.src = currentFilteredImages.original;
  } else if (currentActiveFilter === "qmft") {
    previewImg.src = currentFilteredImages.qmft_denoised;
  } else if (currentActiveFilter === "qelbp") {
    previewImg.src = currentFilteredImages.qe_lbp_edges;
  } else if (currentActiveFilter === "clahe") {
    previewImg.src = currentFilteredImages.clahe_enhanced;
  }
}

async function runTumorAnalysis() {
  const previewImg = document.getElementById("previewImage");
  const btn = document.getElementById("btnAnalyzeMri");
  const base64 = previewImg.dataset.base64;
  const sampleName = previewImg.dataset.sampleName;
  const sampleType = previewImg.dataset.sampleType || "yes";

  if (!base64 && !sampleName) {
    alert("Please upload or select an authentic MRI scan first.");
    return;
  }

  try {
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = `<span class="pulse-dot"></span> Processing Quantum Pipeline...`;
    }

    const payload = base64 ? { image_base64: base64 } : { sample_name: sampleName, sample_type: sampleType };
    const res = await API.predictTumor(payload);

    currentFilteredImages = res.filtered_previews;
    updateFilterPreview();

    // Render subtype banner & badges
    const banner = document.getElementById("tumorDiagnosisBanner");
    const subtypeBadge = document.getElementById("tumorSubtypeBadge");
    const gradeBadge = document.getElementById("tumorGradeBadge");
    const titleEl = document.getElementById("tumorDiagnosisTitle");
    const descEl = document.getElementById("tumorDiagnosisDesc");
    const probEl = document.getElementById("tumorProbValue");

    if (banner) banner.className = `diagnosis-banner ${res.tumor_detected ? 'positive' : 'negative'}`;
    if (subtypeBadge) {
      subtypeBadge.textContent = res.tumor_short_name || (res.tumor_detected ? "Tumor Detected" : "Healthy");
      subtypeBadge.style.background = res.badge_color ? `${res.badge_color}20` : '#E2E8F0';
      subtypeBadge.style.color = res.badge_color || '#1E293B';
      subtypeBadge.style.border = `1px solid ${res.badge_color || '#CBD5E1'}`;
    }
    if (gradeBadge) {
      gradeBadge.textContent = res.tumor_grade || "WHO Grade: --";
    }
    if (titleEl) titleEl.textContent = res.prediction || "Diagnosis Complete";
    if (descEl) descEl.textContent = res.severity_assessment || "";
    if (probEl) probEl.textContent = `${res.confidence_score}%`;

    // Render clinical protocol
    const protocolCard = document.getElementById("tumorProtocolCard");
    const protocolText = document.getElementById("tumorProtocolText");
    if (protocolCard && protocolText) {
      if (res.recommended_protocol) {
        protocolCard.style.display = "block";
        protocolText.textContent = res.recommended_protocol;
      } else {
        protocolCard.style.display = "none";
      }
    }

    // Render 5-class probability distribution bars
    const distContainer = document.getElementById("subtypeDistributionBars");
    if (distContainer && res.class_probabilities) {
      distContainer.innerHTML = "";
      const colors = {
        "Healthy (No Tumor)": "#10B981",
        "Glioblastoma": "#EF4444",
        "Meningioma": "#F59E0B",
        "Pituitary Adenoma": "#3B82F6",
        "Astrocytoma": "#8B5CF6"
      };

      for (const [subtype, pct] of Object.entries(res.class_probabilities)) {
        const item = document.createElement("div");
        const color = colors[subtype] || "#0284C7";
        item.innerHTML = `
          <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px;">
            <span><strong>${subtype}</strong></span>
            <span style="font-weight: 700; color: ${color};">${pct}%</span>
          </div>
          <div class="state-bar-track">
            <div class="state-bar-fill" style="width: ${pct}%; background: ${color};"></div>
          </div>
        `;
        distContainer.appendChild(item);
      }
    }

    // Render Model Architecture Consensus
    if (res.individual_models) {
      const cnnEl = document.getElementById("cnnSubtypeText");
      const hqnnEl = document.getElementById("hqnnSubtypeText");
      const rfEl = document.getElementById("rfSubtypeText");
      const dtEl = document.getElementById("dtSubtypeText");

      if (cnnEl) cnnEl.textContent = res.individual_models["Deep 2D CNN"] || "--";
      if (hqnnEl) hqnnEl.textContent = res.individual_models["HQNN (Quantum)"] || "--";
      if (rfEl) rfEl.textContent = res.individual_models["Random Forest"] || "--";
      if (dtEl) dtEl.textContent = res.individual_models["Decision Tree"] || "--";
    }

  } catch (err) {
    alert(`Inference failed: ${err.message}`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg> Run AI & Quantum Diagnostic Engine`;
    }
  }
}
