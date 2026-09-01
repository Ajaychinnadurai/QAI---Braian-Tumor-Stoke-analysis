/**
 * Quantum Circuit Laboratory Logic
 * Renders interactive circuit diagram for Figure 8 and Table 5 Bell states.
 */

import { API } from "./api.js";

export function initQuantumLab() {
  const theta1 = document.getElementById("qParamTheta1");
  const phi1 = document.getElementById("qParamPhi1");
  const lam1 = document.getElementById("qParamLam1");
  const theta2 = document.getElementById("qParamTheta2");
  const phi2 = document.getElementById("qParamPhi2");
  const lam2 = document.getElementById("qParamLam2");

  const sliders = [theta1, phi1, lam1, theta2, phi2, lam2];
  sliders.forEach(s => {
    if (s) {
      s.addEventListener("input", (e) => {
        const valEl = document.getElementById(`${s.id}Val`);
        if (valEl) valEl.textContent = parseFloat(s.value).toFixed(2);
        triggerSimulation();
      });
    }
  });

  const btnSim = document.getElementById("btnRunQuantumSim");
  if (btnSim) {
    btnSim.addEventListener("click", triggerSimulation);
  }

  loadBellStates();
  triggerSimulation();
}

async function triggerSimulation() {
  const t1 = parseFloat(document.getElementById("qParamTheta1")?.value || (Math.PI/4));
  const p1 = parseFloat(document.getElementById("qParamPhi1")?.value || (Math.PI/2));
  const l1 = parseFloat(document.getElementById("qParamLam1")?.value || 0.0);
  const t2 = parseFloat(document.getElementById("qParamTheta2")?.value || (Math.PI/4));
  const p2 = parseFloat(document.getElementById("qParamPhi2")?.value || (Math.PI/5));
  const l2 = parseFloat(document.getElementById("qParamLam2")?.value || (Math.PI/2));

  try {
    const res = await API.simulateQuantumCircuit({
      theta1: t1, phi1: p1, lam1: l1,
      theta2: t2, phi2: p2, lam2: l2
    });

    // Render Basis state probabilities
    const container = document.getElementById("quantumStateBars");
    if (container && res.basis_states) {
      container.innerHTML = "";
      res.basis_states.forEach(state => {
        const row = document.createElement("div");
        row.className = "state-bar-row";
        const pct = (state.probability * 100).toFixed(2);
        row.innerHTML = `
          <div class="state-basis-label">${state.basis}</div>
          <div class="state-bar-track">
            <div class="state-bar-fill" style="width: ${pct}%;"></div>
          </div>
          <div class="state-val-label">${pct}%</div>
        `;
        container.appendChild(row);
      });
    }

    // Entanglement entropy
    const entropyEl = document.getElementById("qEntanglementEntropy");
    if (entropyEl) entropyEl.textContent = res.entanglement_entropy;

    // Bloch Spheres
    updateBlochSphere("q0Bloch", res.bloch_vectors.q0);
    updateBlochSphere("q1Bloch", res.bloch_vectors.q1);
    updateBlochSphere("q2Bloch", res.bloch_vectors.q2);

  } catch (err) {
    console.error("Quantum simulation error:", err);
  }
}

function updateBlochSphere(elementId, coords) {
  const container = document.getElementById(elementId);
  if (!container) return;

  const dot = container.querySelector(".bloch-vector-pointer");
  const coordText = container.querySelector(".bloch-coords-text");

  if (dot) {
    // Project x, z onto 2D sphere circle
    const posX = 45 + coords.x * 35;
    const posY = 45 - coords.z * 35;
    dot.style.left = `${posX - 5}px`;
    dot.style.top = `${posY - 5}px`;
  }

  if (coordText) {
    coordText.textContent = `[x:${coords.x}, y:${coords.y}, z:${coords.z}]`;
  }
}

async function loadBellStates() {
  try {
    const res = await API.getBellStates();
    const container = document.getElementById("bellStatesGrid");
    if (!container || !res.bell_states) return;

    container.innerHTML = "";
    Object.entries(res.bell_states).forEach(([key, bs]) => {
      const card = document.createElement("div");
      card.className = "card";
      card.style.padding = "16px";
      card.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <strong style="color: var(--accent-cyan); font-family: var(--font-mono);">${key}</strong>
          <span style="font-size: 0.72rem; color: var(--text-muted);">${bs.name}</span>
        </div>
        <div style="font-family: var(--font-mono); font-size: 0.95rem; color: var(--text-primary); margin-bottom: 10px; background: #F1F5F9; border: 1px solid var(--border-color); padding: 6px 10px; border-radius: 6px;">
          ${bs.formula}
        </div>
        <div style="font-size: 0.75rem; color: var(--text-secondary);">
          State: [${bs.state_vector.join(", ")}]
        </div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error("Failed loading Bell states:", err);
  }
}
