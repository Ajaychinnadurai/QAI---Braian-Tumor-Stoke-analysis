"""
Unit Tests for Real Data Loading & Preprocessing
"""

import unittest
import numpy as np
import os
from backend.data_loader import StrokeDataLoader, TumorDataLoader, apply_qmft_denoising, apply_clahe_enhancement
from PIL import Image

class TestDataLoader(unittest.TestCase):
    def test_stroke_data_loading_and_preprocessing(self):
        loader = StrokeDataLoader()
        X_train, X_test, y_train, y_test, df = loader.load_and_preprocess()

        # Check shapes & non-emptiness
        self.assertEqual(len(X_train) + len(X_test), 5110)
        self.assertEqual(X_train.shape[1], 10)
        self.assertTrue(np.all(X_train >= 0.0) and np.all(X_train <= 1.0))
        self.assertFalse(np.isnan(X_train).any())
        self.assertFalse(np.isnan(X_test).any())

        # Test single patient transformation
        sample_patient = {
            "age": 67.0,
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
        x_vec = loader.transform_single_patient(sample_patient)
        self.assertEqual(x_vec.shape, (1, 10))
        self.assertTrue(np.all(x_vec >= 0.0) and np.all(x_vec <= 1.0))

    def test_tumor_mri_loading_and_filtering(self):
        loader = TumorDataLoader(target_size=(64, 64))
        X_img, y, paths = loader.load_real_dataset()

        # Check authentic dataset counts
        self.assertEqual(len(X_img), 253)
        self.assertEqual(sum(y == 1), 155)  # YES scans
        self.assertEqual(sum(y == 0), 98)   # NO scans
        self.assertEqual(X_img.shape, (253, 64, 64, 3))
        self.assertTrue(np.all(X_img >= 0.0) and np.all(X_img <= 1.0))

        # Check feature extraction
        feats = loader.extract_tabular_features_from_images(X_img[:10])
        self.assertEqual(feats.shape, (10, 11))
        self.assertFalse(np.isnan(feats).any())

if __name__ == "__main__":
    unittest.main()
