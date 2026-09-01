"""
Data Loader & Preprocessing Pipeline for Real Datasets
Handles:
1. Real Kaggle Stroke EHR Dataset (healthcare-dataset-stroke-data.csv, 5110 records)
   - Imputation, Label Encoding, MinMaxScaler, Class Weighting
2. Real Kaggle Brain MRI Scans (253 images: yes/no)
   - Contour cropping, MinMax scaling, Quantum-inspired filters (QMFT, QE-LBP, CLAHE)
Directly aligns with Section 4.1 and Section 4.2 of Base1.pdf.
"""

import os
import glob
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Dict, Any, List

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
STROKE_CSV_PATH = os.path.join(DATA_DIR, "stroke", "healthcare-dataset-stroke-data.csv")
TUMOR_YES_DIR = os.path.join(DATA_DIR, "brain_tumor", "yes")
TUMOR_NO_DIR = os.path.join(DATA_DIR, "brain_tumor", "no")

# --- 1. Real Stroke Data Pipeline ---

class StrokeDataLoader:
    def __init__(self, csv_path: str = STROKE_CSV_PATH):
        self.csv_path = csv_path
        self.scaler = MinMaxScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.class_weights: Dict[int, float] = {}

    def load_and_preprocess(self, test_size: float = 0.20, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Loads and preprocesses real EHR stroke records."""
        df = pd.read_csv(self.csv_path)

        # 1. Drop patient ID
        if "id" in df.columns:
            df = df.drop(columns=["id"])

        # 2. Impute missing BMI with median value
        if df["bmi"].isnull().sum() > 0:
            df["bmi"] = df["bmi"].fillna(df["bmi"].median())

        # 3. Handle Other gender if present
        df = df[df["gender"] != "Other"].reset_index(drop=True)

        # 4. Encode Categorical Columns
        cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # 5. Extract Feature Matrix & Target Vector
        X = df.drop(columns=["stroke"]).values
        y = df["stroke"].values
        self.feature_names = [c for c in df.columns if c != "stroke"]

        # 6. Compute Balanced Class Weights
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        self.class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

        # 7. Stratified Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 8. Scale Features to [0, 1] using MinMaxScaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, df

    def transform_single_patient(self, patient_dict: Dict[str, Any]) -> np.ndarray:
        """Transforms single patient inputs into a normalized vector."""
        vec = []
        vec.append(1 if patient_dict.get("gender") == "Male" else 0)
        vec.append(float(patient_dict.get("age", 45)))
        vec.append(1 if patient_dict.get("hypertension") else 0)
        vec.append(1 if patient_dict.get("heart_disease") else 0)
        vec.append(1 if patient_dict.get("ever_married", "Yes") == "Yes" else 0)

        work_type_map = {"Private": 2, "Self-employed": 3, "Govt_job": 0, "children": 4, "Never_worked": 1}
        vec.append(work_type_map.get(patient_dict.get("work_type", "Private"), 2))

        vec.append(1 if patient_dict.get("Residence_type", "Urban") == "Urban" else 0)
        vec.append(float(patient_dict.get("avg_glucose_level", 105.0)))
        vec.append(float(patient_dict.get("bmi", 28.0)))

        smoking_map = {"formerly smoked": 1, "never smoked": 2, "smokes": 3, "Unknown": 0}
        vec.append(smoking_map.get(patient_dict.get("smoking_status", "never smoked"), 2))

        raw_array = np.array([vec])
        return self.scaler.transform(raw_array)


# --- 2. Real Brain Tumor MRI Pipeline ---

def crop_brain_contour(img: Image.Image) -> Image.Image:
    """
    Crops extreme background and skull contour using Otsu-thresholding,
    focusing models purely on brain tissue parenchyma and intracranial lesions.
    """
    np_img = np.array(img.convert("L"))
    
    # Simple Otsu threshold
    hist, bin_edges = np.histogram(np_img, bins=256, range=(0, 256))
    total = np_img.size
    current_max, threshold = 0, 0
    sum_total = np.dot(np.arange(256), hist)
    sum_back, weight_back = 0, 0

    for i in range(256):
        weight_back += hist[i]
        if weight_back == 0:
            continue
        weight_fore = total - weight_back
        if weight_fore == 0:
            break
        sum_back += i * hist[i]
        mean_back = sum_back / weight_back
        mean_fore = (sum_total - sum_back) / weight_fore
        var_between = weight_back * weight_fore * ((mean_back - mean_fore) ** 2)
        if var_between > current_max:
            current_max = var_between
            threshold = i

    thresh_val = max(threshold, 35)
    mask = np_img > thresh_val
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    
    # Add small margin
    h, w = np_img.shape
    y0 = max(0, y0 - 4)
    x0 = max(0, x0 - 4)
    y1 = min(h, y1 + 4)
    x1 = min(w, x1 + 4)
    
    return img.crop((x0, y0, x1, y1))

def apply_qmft_denoising(img: Image.Image) -> Image.Image:
    """Simulates Quantum Matched Filter (QMFT) active noise reduction."""
    return img.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.EDGE_ENHANCE)

def apply_qe_lbp_filter(img: Image.Image) -> Image.Image:
    """Simulates Quantum Entropy Local Binary Pattern (QE-LBP) edge & microcalcification detection."""
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    return edges

def apply_clahe_enhancement(img: Image.Image) -> Image.Image:
    """Enhances local contrast akin to CLAHE (Section 4.1 Step c)."""
    return ImageOps.autocontrast(img, cutoff=2)

class TumorDataLoader:
    def __init__(self, yes_dir: str = TUMOR_YES_DIR, no_dir: str = TUMOR_NO_DIR, target_size: Tuple[int, int] = (128, 128)):
        self.yes_dir = yes_dir
        self.no_dir = no_dir
        self.target_size = target_size

    def preprocess_image(self, img: Image.Image) -> Tuple[np.ndarray, Image.Image]:
        """Applies contour cropping, CLAHE enhancement, and resizing."""
        cropped = crop_brain_contour(img.convert("RGB"))
        enhanced = apply_clahe_enhancement(cropped)
        resized = enhanced.resize(self.target_size, Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0  # Normalized [0, 1]
        return arr, resized

    def load_real_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Loads all authentic MRI images from yes/ and no/ folders.
        Returns:
            X: Array of shape (N, H, W, 3) normalized to [0, 1]
            y: Array of shape (N,) where 1 = YES (tumor), 0 = NO (no tumor)
            file_paths: List of original file paths
        """
        images = []
        labels = []
        file_paths = []

        # Load YES (Tumor present = 1)
        yes_files = sorted(glob.glob(os.path.join(self.yes_dir, "*.*")))
        for p in yes_files:
            try:
                with Image.open(p) as img:
                    arr, _ = self.preprocess_image(img)
                    images.append(arr)
                    labels.append(1)
                    file_paths.append(p)
            except Exception:
                pass

        # Load NO (Healthy = 0)
        no_files = sorted(glob.glob(os.path.join(self.no_dir, "*.*")))
        for p in no_files:
            try:
                with Image.open(p) as img:
                    arr, _ = self.preprocess_image(img)
                    images.append(arr)
                    labels.append(0)
                    file_paths.append(p)
            except Exception:
                pass

        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        return X, y, file_paths

    def extract_tabular_features_from_images(self, X_images: np.ndarray) -> np.ndarray:
        """
        Extracts statistical/texture features (mean, std, percentiles, gradients)
        for Decision Tree and Random Forest comparative baselines (Table 9).
        """
        features = []
        for img in X_images:
            # Grayscale channel
            gray = np.mean(img, axis=-1)
            f_mean = np.mean(gray)
            f_std = np.std(gray)
            f_max = np.max(gray)
            f_min = np.min(gray)
            f_p25 = np.percentile(gray, 25)
            f_p50 = np.percentile(gray, 50)
            f_p75 = np.percentile(gray, 75)
            f_p90 = np.percentile(gray, 90)
            # Spatial energy / center mass
            h, w = gray.shape
            center_patch = gray[h//4:3*h//4, w//4:3*w//4]
            f_center_mean = np.mean(center_patch)
            f_center_std = np.std(center_patch)
            # Gradients
            dy, dx = np.gradient(gray)
            f_grad_mag = np.mean(np.sqrt(dx**2 + dy**2))
            
            vec = [f_mean, f_std, f_max, f_min, f_p25, f_p50, f_p75, f_p90, f_center_mean, f_center_std, f_grad_mag]
            features.append(vec)
        return np.array(features, dtype=np.float32)

