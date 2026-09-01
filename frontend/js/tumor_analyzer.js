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

export function selectSampleMri(filename, type = "yes") {
  const imgUrl = `/data/brain_tumor/${type}/${filename}`;
  const previewImg = document.getElementById("previewImage");
  previewImg.src = imgUrl;
  previewImg.dataset.sampleName = filename;
  previewImg.dataset.sampleType = type;
  previewImg.dataset.base64 = "";
  
  document.getElementById("selectedMriName").textContent = `Real Kaggle MRI: ${filename} (${type.toUpperCase()})`;
  document.getElementById("mriPlaceholderText").style.display = "none";
  previewImg.style.display = "block";
  resetFilterTabs();
  runTumorAnalysis();
}

async function loadSampleMriPresets() {
  try {
    const samples = await API.getDatasetSamples();
    const yesContainer = document.getElementById("mriYesPresets");
    const noContainer = document.getElementById("mriNoPresets");

    if (yesContainer && samples.mri_samples.yes_scans) {
      yesContainer.innerHTML = "";
      samples.mri_samples.yes_scans.slice(0, 4).forEach(file => {
        const chip = document.createElement("button");
        chip.className = "preset-chip";
        chip.textContent = file;
        chip.onclick = () => selectSampleMri(file, "yes");
        yesContainer.appendChild(chip);
      });
    }

    if (noContainer && samples.mri_samples.no_scans) {
      noContainer.innerHTML = "";
      samples.mri_samples.no_scans.slice(0, 4).forEach(file => {
        const chip = document.createElement("button");
        chip.className = "preset-chip";
        chip.textContent = file;
        chip.onclick = () => selectSampleMri(file, "no");
        noContainer.appendChild(chip);
      });
    }
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

    // Render results
    const banner = document.getElementById("tumorDiagnosisBanner");
    const titleEl = document.getElementById("tumorDiagnosisTitle");
    const descEl = document.getElementById("tumorDiagnosisDesc");
    const probEl = document.getElementById("tumorProbValue");
    const confEl = document.getElementById("tumorConfValue");

    if (banner) banner.className = `diagnosis-banner ${res.tumor_detected ? 'positive' : 'negative'}`;
    if (titleEl) titleEl.textContent = res.prediction || "Diagnosis Complete";
    if (descEl) descEl.textContent = res.severity_assessment || "";
    if (probEl) probEl.textContent = `${res.probability_percentage}%`;
    if (confEl) confEl.textContent = `${res.confidence_score}%`;

    // Model breakdown
    const cnnBar = document.getElementById("cnnProbBar");
    const cnnText = document.getElementById("cnnProbText");
    if (cnnBar) cnnBar.style.width = `${res.individual_models.CNN}%`;
    if (cnnText) cnnText.textContent = `${res.individual_models.CNN}%`;

    const hqnnBar = document.getElementById("hqnnProbBar");
    const hqnnText = document.getElementById("hqnnProbText");
    if (hqnnBar) hqnnBar.style.width = `${res.individual_models["HQNN (Quantum)"]}%`;
    if (hqnnText) hqnnText.textContent = `${res.individual_models["HQNN (Quantum)"]}%`;

    const rfBar = document.getElementById("rfProbBar");
    const rfText = document.getElementById("rfProbText");
    if (rfBar) rfBar.style.width = `${res.individual_models["Random Forest"]}%`;
    if (rfText) rfText.textContent = `${res.individual_models["Random Forest"]}%`;

    const dtBar = document.getElementById("dtProbBar");
    const dtText = document.getElementById("dtProbText");
    if (dtBar) dtBar.style.width = `${res.individual_models["Decision Tree"]}%`;
    if (dtText) dtText.textContent = `${res.individual_models["Decision Tree"]}%`;

  } catch (err) {
    alert(`Inference failed: ${err.message}`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg> Run AI & Quantum Diagnostic Engine`;
    }
  }
}
