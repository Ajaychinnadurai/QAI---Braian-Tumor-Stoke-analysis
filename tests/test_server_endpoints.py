"""
End-to-End API Server Integration Test
Validates all REST endpoints against the live running server at http://localhost:8080
"""

import unittest
import urllib.request
import json

SERVER_URL = "http://localhost:8080"

class TestServerEndpoints(unittest.TestCase):
    def test_root_html(self):
        req = urllib.request.urlopen(f"{SERVER_URL}/")
        self.assertEqual(req.status, 200)
        content = req.read().decode("utf-8")
        self.assertIn("QuantumHealth AI", content)
        self.assertIn("Base1.pdf", content)

    def test_benchmark_metrics_endpoint(self):
        req = urllib.request.urlopen(f"{SERVER_URL}/api/benchmark/metrics")
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertIn("table_9_brain_tumor", data)
        self.assertIn("table_10_brain_stroke", data)
        self.assertEqual(data["table_9_brain_tumor"]["models"]["Convolutional Neural Network (CNN)"]["accuracy"], 98.0)
        self.assertEqual(data["table_10_brain_stroke"]["models"]["Logistic Regression"]["accuracy"], 94.7)

    def test_dataset_samples_endpoint(self):
        req = urllib.request.urlopen(f"{SERVER_URL}/api/dataset/samples")
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertIn("mri_samples", data)
        self.assertIn("patient_samples", data)
        self.assertGreater(len(data["mri_samples"]["yes_scans"]), 0)
        self.assertGreater(len(data["mri_samples"]["no_scans"]), 0)
        self.assertEqual(len(data["patient_samples"]), 6)

    def test_quantum_bell_states_endpoint(self):
        req = urllib.request.urlopen(f"{SERVER_URL}/api/quantum/bell-states")
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertIn("bell_states", data)
        self.assertIn("Q0", data["bell_states"])

    def test_quantum_simulate_endpoint(self):
        payload = json.dumps({"theta1": 0.785, "phi1": 1.571, "lam1": 0.0}).encode("utf-8")
        req = urllib.request.Request(
            f"{SERVER_URL}/api/quantum/simulate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req)
        self.assertEqual(resp.status, 200)
        data = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(len(data["basis_states"]), 8)
        self.assertIn("bloch_vectors", data)

    def test_stroke_prediction_endpoint(self):
        patient = {
            "age": 67.0,
            "gender": "Male",
            "hypertension": 1,
            "heart_disease": 0,
            "ever_married": "Yes",
            "work_type": "Private",
            "Residence_type": "Urban",
            "avg_glucose_level": 228.69,
            "bmi": 36.6,
            "smoking_status": "formerly smoked",
            "alcohol_intake": 3
        }
        payload = json.dumps(patient).encode("utf-8")
        req = urllib.request.Request(
            f"{SERVER_URL}/api/predict/stroke",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req)
        self.assertEqual(resp.status, 200)
        data = json.loads(resp.read().decode("utf-8"))
        self.assertIn("probabilities", data)
        self.assertIn("overall_risk_percentage", data)
        self.assertIn("recommendations", data)
        self.assertGreater(len(data["recommendations"]), 0)

    def test_tumor_prediction_endpoint(self):
        # Fetch sample names first
        samples_req = urllib.request.urlopen(f"{SERVER_URL}/api/dataset/samples")
        samples_data = json.loads(samples_req.read().decode("utf-8"))
        first_yes = samples_data["mri_samples"]["yes_scans"][0]

        payload = json.dumps({"sample_name": first_yes, "sample_type": "yes"}).encode("utf-8")
        req = urllib.request.Request(
            f"{SERVER_URL}/api/predict/tumor",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req)
        self.assertEqual(resp.status, 200)
        data = json.loads(resp.read().decode("utf-8"))
        self.assertIn("prediction", data)
        self.assertIn("filtered_previews", data)
        self.assertIn("qmft_denoised", data["filtered_previews"])
        self.assertIn("qe_lbp_edges", data["filtered_previews"])
        self.assertIn("clahe_enhanced", data["filtered_previews"])

if __name__ == "__main__":
    unittest.main()
