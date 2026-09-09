"""
Production-Grade API & Application Server
Serves the Interactive Healthcare Analytics Dashboard and provides
REST endpoints for Brain Tumor MRI Detection, Stroke Prognosis,
Quantum Circuit Simulations, and Research Benchmark Analytics.
"""

import os
import io
import json
import base64
import urllib.parse
import warnings
warnings.filterwarnings("ignore")
import sys
from typing import Any, Dict, List, Tuple
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
import pandas as pd
from PIL import Image
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.data_loader import (
    StrokeDataLoader, TumorDataLoader, 
    apply_qmft_denoising, apply_qe_lbp_filter, apply_clahe_enhancement
)
from backend.stroke_models import StrokeModelSuite
from backend.tumor_models import BrainTumorModelSuite
from backend.quantum_engine import (
    simulate_figure_8_circuit, generate_bell_states, QuantumSimulator, unitary_gate, H, X, Z
)

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load models and loaders
stroke_loader: StrokeDataLoader = joblib.load(os.path.join(MODELS_DIR, "stroke_scaler.joblib"))
stroke_suite = StrokeModelSuite()
stroke_suite.load_models(os.path.join(MODELS_DIR, "stroke_models.joblib"))

tumor_loader = TumorDataLoader()
tumor_suite = BrainTumorModelSuite()
tumor_suite.load_models(os.path.join(MODELS_DIR, "tumor_models.joblib"))

with open(os.path.join(MODELS_DIR, "benchmark_metrics.json"), "r", encoding="utf-8") as f:
    BENCHMARK_METRICS = json.load(f)

# Load real dataset samples for quick dashboard exploration
REAL_STROKE_DF = pd.read_csv(os.path.join(DATA_DIR, "stroke", "healthcare-dataset-stroke-data.csv"))
REAL_STROKE_DF["bmi"] = REAL_STROKE_DF["bmi"].fillna(28.89)
REAL_STROKE_DF = REAL_STROKE_DF.fillna(0)

def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_preventive_recommendations(risk_score: float, patient_data: dict) -> list:
    recs = []
    age = float(patient_data.get("age", 50))
    glucose = float(patient_data.get("avg_glucose_level", 100))
    bmi = float(patient_data.get("bmi", 25))
    hyp = int(patient_data.get("hypertension", 0))
    hd = int(patient_data.get("heart_disease", 0))
    smoking = str(patient_data.get("smoking_status", ""))

    if hyp == 1 or risk_score > 60:
        recs.append({
            "category": "Cardiovascular & Blood Pressure",
            "icon": "heart",
            "advice": "Daily blood pressure monitoring required. Maintain target systolic BP < 120 mmHg under physician supervision.",
            "urgency": "High"
        })
    if glucose > 140:
        recs.append({
            "category": "Metabolic & Glycemic Control",
            "icon": "activity",
            "advice": f"Elevated blood glucose ({glucose} mg/dL). Fasting blood sugar and HbA1c screening recommended to mitigate ischemic risk.",
            "urgency": "High" if glucose > 180 else "Moderate"
        })
    if bmi > 28:
        recs.append({
            "category": "Weight & Lifestyle Management",
            "icon": "shield",
            "advice": f"BMI is {round(bmi,1)}. Adopt a Mediterranean or DASH dietary pattern paired with 150 min/week moderate aerobic activity.",
            "urgency": "Moderate"
        })
    if "smoke" in smoking.lower() and "never" not in smoking.lower():
        recs.append({
            "category": "Smoking Cessation Protocol",
            "icon": "alert-circle",
            "advice": "Tobacco use exponentially accelerates arterial plaque buildup. Immediate smoking cessation therapy strongly advised.",
            "urgency": "High"
        })
    if hd == 1:
        recs.append({
            "category": "Cardiac Care Plan",
            "icon": "crosshair",
            "advice": "Active heart condition detected. Coordinate cardiology review for antiplatelet / anticoagulant management.",
            "urgency": "Critical"
        })
    if len(recs) == 0 or risk_score < 30:
        recs.append({
            "category": "Routine Preventative Care",
            "icon": "check-circle",
            "advice": "Biomarkers indicate low current stroke vulnerability. Continue regular annual checkups and balanced physical routine.",
            "urgency": "Low"
        })
    return recs


def sanitize_nan(obj: Any) -> Any:
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    elif isinstance(obj, np.ndarray):
        return [sanitize_nan(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_nan(v) for v in obj]
    return obj


class HealthcareRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: Any, status: int = 200):
        clean_data = sanitize_nan(data)
        body = json.dumps(clean_data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, filepath: str, content_type: str):
        if not os.path.exists(filepath):
            self.send_error(404, f"File not found: {filepath}")
            return
        with open(filepath, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        url_path = urllib.parse.urlparse(self.path).path

        # API Endpoints
        if url_path == "/api/benchmark/metrics":
            with open(os.path.join(MODELS_DIR, "benchmark_metrics.json"), "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            self._send_json(metrics_data)
            return

        elif url_path == "/api/quantum/bell-states":
            bell_states = generate_bell_states()
            self._send_json({"bell_states": bell_states})
            return

        elif url_path == "/api/dataset/samples":
            # Return real sample MRI scans separated by subtype and real patient records
            subtypes = ["healthy", "glioblastoma", "meningioma", "pituitary", "astrocytoma"]
            mri_samples = {}
            for st in subtypes:
                st_dir = os.path.join(DATA_DIR, "brain_tumor", st)
                if os.path.exists(st_dir):
                    mri_samples[f"{st}_scans"] = [os.path.basename(p) for p in sorted(os.listdir(st_dir))[:6]]
                else:
                    mri_samples[f"{st}_scans"] = []

            # Backward compatibility
            mri_samples["yes_scans"] = mri_samples.get("glioblastoma_scans", []) + mri_samples.get("meningioma_scans", [])
            mri_samples["no_scans"] = mri_samples.get("healthy_scans", [])
            
            # 6 sample real stroke cases from healthcare-dataset-stroke-data.csv
            stroke_cases = REAL_STROKE_DF[REAL_STROKE_DF["stroke"] == 1].head(3).to_dict(orient="records")
            healthy_cases = REAL_STROKE_DF[REAL_STROKE_DF["stroke"] == 0].head(3).to_dict(orient="records")
            sample_patients = stroke_cases + healthy_cases

            self._send_json({
                "mri_samples": mri_samples,
                "patient_samples": sample_patients
            })
            return

        elif url_path.startswith("/data/brain_tumor/"):
            # Serve real MRI images directly
            rel_path = url_path.replace("/data/brain_tumor/", "")
            full_path = os.path.join(DATA_DIR, "brain_tumor", rel_path)
            ext = os.path.splitext(full_path)[1].lower()
            mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
            self._send_file(full_path, mime)
            return

        # Static Frontend Files
        if url_path == "/" or url_path == "/index.html":
            self._send_file(os.path.join(FRONTEND_DIR, "index.html"), "text/html")
        elif url_path.startswith("/css/"):
            rel = url_path.replace("/css/", "")
            self._send_file(os.path.join(FRONTEND_DIR, "css", rel), "text/css")
        elif url_path.startswith("/js/"):
            rel = url_path.replace("/js/", "")
            self._send_file(os.path.join(FRONTEND_DIR, "js", rel), "application/javascript")
        else:
            local_file = os.path.join(FRONTEND_DIR, url_path.lstrip("/"))
            if os.path.exists(local_file):
                ext = os.path.splitext(local_file)[1].lower()
                mimes = {".html": "text/html", ".css": "text/css", ".js": "application/javascript", ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml"}
                self._send_file(local_file, mimes.get(ext, "application/octet-stream"))
            else:
                self.send_error(404, "Not Found")

    def do_POST(self):
        try:
            url_path = urllib.parse.urlparse(self.path).path
            content_len = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_len) if content_len > 0 else b""
            try:
                payload = json.loads(post_data.decode("utf-8")) if post_data else {}
            except Exception:
                payload = {}

            if url_path == "/api/predict/stroke":
                # Real stroke prediction
                x_scaled = stroke_loader.transform_single_patient(payload)
                result = stroke_suite.predict_single_patient(x_scaled)
                recommendations = generate_preventive_recommendations(result["overall_risk_percentage"], payload)
                result["recommendations"] = recommendations
                self._send_json(result)

            elif url_path == "/api/predict/tumor":
                # Real brain tumor MRI prediction
                image_raw = None
                if "image_base64" in payload and payload["image_base64"]:
                    b64_data = payload["image_base64"]
                    if "," in b64_data:
                        b64_data = b64_data.split(",")[1]
                    img_bytes = base64.b64decode(b64_data)
                    image_raw = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                elif "sample_name" in payload and payload["sample_name"]:
                    # Load from real dataset
                    sname = payload["sample_name"]
                    stype = payload.get("sample_type", "healthy")
                    p = os.path.join(DATA_DIR, "brain_tumor", stype, sname)
                    if not os.path.exists(p):
                        for sub in ["healthy", "glioblastoma", "meningioma", "pituitary", "astrocytoma"]:
                            alt_p = os.path.join(DATA_DIR, "brain_tumor", sub, sname)
                            if os.path.exists(alt_p):
                                p = alt_p
                                break
                    if os.path.exists(p):
                        image_raw = Image.open(p).convert("RGB")

                if image_raw is None:
                    self._send_json({"error": "No valid MRI image provided"}, status=400)
                    return

                # Apply Preprocessing (Contour cropping, CLAHE enhancement, normalization)
                arr, img_resized = tumor_loader.preprocess_image(image_raw)
                img_qmft = apply_qmft_denoising(img_resized)
                img_qelbp = apply_qe_lbp_filter(img_resized)
                img_clahe = apply_clahe_enhancement(img_resized)

                # Extract tabular features for comparative models
                feats = tumor_loader.extract_tabular_features_from_images(np.array([arr]))

                # Run inference on Deep 2D PyTorch CNN and comparative models
                result = tumor_suite.predict_single_mri(arr, feats)
                result["filtered_previews"] = {
                    "original": f"data:image/png;base64,{image_to_base64(img_resized)}",
                    "qmft_denoised": f"data:image/png;base64,{image_to_base64(img_qmft)}",
                    "qe_lbp_edges": f"data:image/png;base64,{image_to_base64(img_qelbp)}",
                    "clahe_enhanced": f"data:image/png;base64,{image_to_base64(img_clahe)}"
                }
                self._send_json(result)

            elif url_path == "/api/quantum/simulate":
                # Real quantum circuit simulation
                t1 = float(payload.get("theta1", np.pi/4))
                p1 = float(payload.get("phi1", np.pi/2))
                l1 = float(payload.get("lam1", 0.0))
                t2 = float(payload.get("theta2", np.pi/4))
                p2 = float(payload.get("phi2", np.pi/5))
                l2 = float(payload.get("lam2", np.pi/2))

                sim_res = simulate_figure_8_circuit(t1, p1, l1, t2, p2, l2)
                self._send_json(sim_res)

            else:
                self.send_error(404, "Endpoint not found")

        except Exception as err:
            import traceback
            traceback.print_exc()
            self._send_json({"error": f"Inference processing exception: {str(err)}"}, status=500)

def start_server(port: int = 8080):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, HealthcareRequestHandler)
    print("\n============================================================")
    print("AI-Driven QML Healthcare Analytics Server running at:")
    print(f"   http://localhost:{port}")
    print("============================================================\n")
    sys.stdout.flush()
    httpd.serve_forever()

if __name__ == "__main__":
    start_server()
