/**
 * Brain Stroke Prognosis Engine Logic
 * Supports real Kaggle EHR patients, multi-model evaluation (LR, SVM, KNN, RF, DT, GNB, QSVC),
 * factor radar chart, and personalized preventive care protocols.
 */

import { API } from "./api.js";
import { renderRadarChart } from "./charts.js";

let realPatientSamples = [];

export function initStrokePrognosis() {
  const form = document.getElementById("strokeForm");
  const evaluateBtn = document.getElementById("btnEvaluateStroke");

  if (evaluateBtn) {
    evaluateBtn.addEventListener("click", runStrokePrognosis);
  }

  loadPatientPresets();
}

async function loadPatientPresets() {
  try {
    const samples = await API.getDatasetSamples();
    realPatientSamples = samples.patient_samples || [];
    const container = document.getElementById("patientPresetsContainer");
    if (!container) return;

    container.innerHTML = "";
    realPatientSamples.forEach((p, idx) => {
      const chip = document.createElement("button");
      chip.className = "preset-chip";
      const isStroke = p.stroke === 1;
      chip.innerHTML = `<strong>Case #${idx + 1}</strong>: ${p.age}y ${p.gender} (Glucose: ${Math.round(p.avg_glucose_level)}, ${isStroke ? 'Stroke Case' : 'Healthy'})`;
      chip.onclick = () => populatePatientForm(p);
      container.appendChild(chip);
    });

    // Populate first sample by default
    if (realPatientSamples.length > 0) {
      populatePatientForm(realPatientSamples[0]);
    }
  } catch (err) {
    console.error("Failed loading patient presets:", err);
  }
}

export function populatePatientForm(patient) {
  document.getElementById("strokeAge").value = patient.age || 50;
  document.getElementById("strokeGender").value = patient.gender || "Male";
  document.getElementById("strokeHyp").value = patient.hypertension !== undefined ? patient.hypertension : 0;
  document.getElementById("strokeHeart").value = patient.heart_disease !== undefined ? patient.heart_disease : 0;
  document.getElementById("strokeMarried").value = patient.ever_married || "Yes";
  document.getElementById("strokeWork").value = patient.work_type || "Private";
  document.getElementById("strokeResidence").value = patient.Residence_type || "Urban";
  document.getElementById("strokeGlucose").value = patient.avg_glucose_level || 105.0;
  document.getElementById("strokeBmi").value = patient.bmi ? parseFloat(patient.bmi).toFixed(1) : 28.8;
  document.getElementById("strokeSmoking").value = patient.smoking_status || "never smoked";
  document.getElementById("strokeAlcohol").value = patient.alcohol_intake !== undefined ? patient.alcohol_intake : 2;

  runStrokePrognosis();
}

function getFormData() {
  return {
    age: parseFloat(document.getElementById("strokeAge").value),
    gender: document.getElementById("strokeGender").value,
    hypertension: parseInt(document.getElementById("strokeHyp").value),
    heart_disease: parseInt(document.getElementById("strokeHeart").value),
    ever_married: document.getElementById("strokeMarried").value,
    work_type: document.getElementById("strokeWork").value,
    Residence_type: document.getElementById("strokeResidence").value,
    avg_glucose_level: parseFloat(document.getElementById("strokeGlucose").value),
    bmi: parseFloat(document.getElementById("strokeBmi").value),
    smoking_status: document.getElementById("strokeSmoking").value,
    alcohol_intake: parseInt(document.getElementById("strokeAlcohol").value)
  };
}

async function runStrokePrognosis() {
  const patientData = getFormData();
  const btn = document.getElementById("btnEvaluateStroke");

  try {
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = `<span class="pulse-dot"></span> Evaluating Multi-Model Suite...`;
    }

    const res = await API.predictStroke(patientData);

    // Update risk badge & scores
    const banner = document.getElementById("strokeDiagnosisBanner");
    const titleEl = document.getElementById("strokeDiagnosisTitle");
    const descEl = document.getElementById("strokeDiagnosisDesc");
    const riskEl = document.getElementById("strokeRiskScore");

    const risk = res.overall_risk_percentage;
    if (riskEl) riskEl.textContent = `${risk}%`;

    if (banner) {
      if (risk >= 70) {
        banner.className = "diagnosis-banner positive";
      } else if (risk >= 40) {
        banner.className = "diagnosis-banner elevated";
      } else {
        banner.className = "diagnosis-banner negative";
      }
    }

    if (titleEl) titleEl.textContent = res.stroke_detected ? "Elevated Stroke Risk Detected" : "Low Risk Profile";
    if (descEl) descEl.textContent = res.severity_level || "";

    // Model Shootout Probabilities
    const probs = res.probabilities;
    setBarProgress("lrProb", probs["Logistic Regression"]);
    setBarProgress("svmProb", probs["Support Vector Machine"]);
    setBarProgress("knnProb", probs["K Nearest Neighbors"]);
    setBarProgress("rfStrokeProb", probs["Random Forest"]);
    setBarProgress("dtStrokeProb", probs["Decision Tree"]);
    setBarProgress("gnbProb", probs["Gaussian Naive Bayes"]);
    setBarProgress("qsvcProb", probs["QSVC (Quantum SVM)"]);

    // Radar factor chart
    const factors = {
      "Age": Math.min(patientData.age / 90.0, 1.0),
      "Glucose": Math.min(patientData.avg_glucose_level / 250.0, 1.0),
      "BMI": Math.min(patientData.bmi / 45.0, 1.0),
      "Hypertension": patientData.hypertension ? 0.9 : 0.1,
      "Heart Disease": patientData.heart_disease ? 0.9 : 0.1,
      "Smoking": patientData.smoking_status === "smokes" ? 0.85 : (patientData.smoking_status === "formerly smoked" ? 0.5 : 0.15)
    };
    renderRadarChart("strokeRadarChart", factors);

    // Preventive Recommendations
    const recsContainer = document.getElementById("preventiveRecommendations");
    if (recsContainer && res.recommendations) {
      recsContainer.innerHTML = "";
      res.recommendations.forEach(r => {
        const item = document.createElement("div");
        item.className = "rec-card";
        item.innerHTML = `
          <div>
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
              <span class="rec-badge ${r.urgency}">${r.urgency}</span>
              <strong style="font-size: 0.86rem; color: #F8FAFC;">${r.category}</strong>
            </div>
            <p style="font-size: 0.8rem; color: #94A3B8;">${r.advice}</p>
          </div>
        `;
        recsContainer.appendChild(item);
      });
    }

  } catch (err) {
    alert(`Stroke evaluation failed: ${err.message}`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg> Run Clinical Risk Prognosis`;
    }
  }
}

function setBarProgress(idPrefix, val) {
  const bar = document.getElementById(`${idPrefix}Bar`);
  const text = document.getElementById(`${idPrefix}Text`);
  if (bar && text && val !== undefined) {
    bar.style.width = `${val}%`;
    text.textContent = `${val}%`;
  }
}
