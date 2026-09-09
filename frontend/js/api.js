/**
 * API Client Module for Healthcare Analytics System
 */

const getApiBase = () => {
  if (typeof window === "undefined") return "http://localhost:8080";
  const { protocol, hostname, port } = window.location;
  if (protocol === "file:" || (hostname === "localhost" || hostname === "127.0.0.1") && port !== "8080") {
    return "http://localhost:8080";
  }
  return "";
};

const API_BASE = getApiBase();

async function safeFetchJson(res) {
  const text = await res.text();
  const cleanText = text.replace(/:\s*NaN\b/g, ": null").replace(/:\s*Infinity\b/g, ": null");
  return JSON.parse(cleanText);
}

export const API = {
  async getBenchmarkMetrics() {
    const res = await fetch(`${API_BASE}/api/benchmark/metrics`);
    if (!res.ok) throw new Error("Failed to fetch benchmark metrics");
    return await safeFetchJson(res);
  },

  async getDatasetSamples() {
    const res = await fetch(`${API_BASE}/api/dataset/samples`);
    if (!res.ok) throw new Error("Failed to fetch dataset samples");
    return await safeFetchJson(res);
  },

  async getBellStates() {
    const res = await fetch(`${API_BASE}/api/quantum/bell-states`);
    if (!res.ok) throw new Error("Failed to fetch Bell states");
    return await safeFetchJson(res);
  },

  async simulateQuantumCircuit(params) {
    const res = await fetch(`${API_BASE}/api/quantum/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params || {})
    });
    if (!res.ok) throw new Error("Quantum simulation failed");
    return await safeFetchJson(res);
  },

  async predictTumor(data) {
    const res = await fetch(`${API_BASE}/api/predict/tumor`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error("Tumor prediction failed");
    return await safeFetchJson(res);
  },

  async predictStroke(patientData) {
    const res = await fetch(`${API_BASE}/api/predict/stroke`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patientData)
    });
    if (!res.ok) throw new Error("Stroke prediction failed");
    return await safeFetchJson(res);
  }
};
