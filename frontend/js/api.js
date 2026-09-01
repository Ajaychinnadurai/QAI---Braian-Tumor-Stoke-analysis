/**
 * API Client Module for Healthcare Analytics System
 */

const API_BASE = "";

export const API = {
  async getBenchmarkMetrics() {
    const res = await fetch(`${API_BASE}/api/benchmark/metrics`);
    if (!res.ok) throw new Error("Failed to fetch benchmark metrics");
    return await res.json();
  },

  async getDatasetSamples() {
    const res = await fetch(`${API_BASE}/api/dataset/samples`);
    if (!res.ok) throw new Error("Failed to fetch dataset samples");
    return await res.json();
  },

  async getBellStates() {
    const res = await fetch(`${API_BASE}/api/quantum/bell-states`);
    if (!res.ok) throw new Error("Failed to fetch Bell states");
    return await res.json();
  },

  async simulateQuantumCircuit(params) {
    const res = await fetch(`${API_BASE}/api/quantum/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params || {})
    });
    if (!res.ok) throw new Error("Quantum simulation failed");
    return await res.json();
  },

  async predictTumor(data) {
    const res = await fetch(`${API_BASE}/api/predict/tumor`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error("Tumor prediction failed");
    return await res.json();
  },

  async predictStroke(patientData) {
    const res = await fetch(`${API_BASE}/api/predict/stroke`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patientData)
    });
    if (!res.ok) throw new Error("Stroke prediction failed");
    return await res.json();
  }
};
