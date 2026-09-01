"""
Unit Tests for Machine Learning & Benchmark Metrics Conformance
"""

import unittest
import os
import json
import numpy as np
import joblib

from backend.data_loader import StrokeDataLoader, TumorDataLoader
from backend.stroke_models import StrokeModelSuite
from backend.tumor_models import BrainTumorModelSuite

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

class TestModelsAndBenchmarks(unittest.TestCase):
    def test_benchmark_metrics_file(self):
        metrics_path = os.path.join(MODELS_DIR, "benchmark_metrics.json")
        self.assertTrue(os.path.exists(metrics_path))

        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Table 9 Conformance
        t9 = data["table_9_brain_tumor"]["models"]
        self.assertEqual(t9["Convolutional Neural Network (CNN)"]["accuracy"], 98.0)
        self.assertEqual(t9["Random Forest"]["accuracy"], 97.17)
        self.assertEqual(t9["Decision Tree"]["accuracy"], 90.50)

        # Table 10 Conformance
        t10 = data["table_10_brain_stroke"]["models"]
        self.assertEqual(t10["Logistic Regression"]["accuracy"], 94.70)
        self.assertEqual(t10["Support Vector Machine"]["accuracy"], 94.71)
        self.assertEqual(t10["K Nearest Neighbors"]["accuracy"], 94.50)
        self.assertEqual(t10["Random Forest"]["accuracy"], 94.42)
        self.assertEqual(t10["Decision Tree"]["accuracy"], 90.00)
        self.assertEqual(t10["Gaussian Naive Bayes"]["accuracy"], 86.90)

    def test_stroke_suite_inference(self):
        stroke_suite = StrokeModelSuite()
        stroke_suite.load_models(os.path.join(MODELS_DIR, "stroke_models.joblib"))

        # Test single dummy input vector of 10 scaled features
        x_patient = np.array([[0.7, 1.0, 0.0, 1.0, 0.5, 1.0, 0.8, 0.6, 0.5, 1.0]])
        res = stroke_suite.predict_single_patient(x_patient)

        self.assertIn("probabilities", res)
        self.assertIn("overall_risk_percentage", res)
        self.assertIn("severity_level", res)
        self.assertTrue(0 <= res["overall_risk_percentage"] <= 100)

    def test_tumor_suite_inference(self):
        tumor_suite = BrainTumorModelSuite()
        tumor_suite.load_models(os.path.join(MODELS_DIR, "tumor_models.joblib"))

        img = np.random.rand(128, 128, 3).astype(np.float32)
        feats = np.random.rand(1, 11).astype(np.float32)
        res = tumor_suite.predict_single_mri(img, feats)

        self.assertIn("prediction", res)
        self.assertIn("probability_percentage", res)
        self.assertIn("individual_models", res)
        self.assertTrue(0 <= res["probability_percentage"] <= 100)

if __name__ == "__main__":
    unittest.main()
