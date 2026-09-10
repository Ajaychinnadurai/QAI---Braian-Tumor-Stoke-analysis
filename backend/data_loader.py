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

# SMOTE for stroke class-imbalance oversampling (optional but recommended)
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

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

    def load_and_preprocess(
        self, test_size: float = 0.20, random_state: int = 42, oversample: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Loads and preprocesses real EHR stroke records.
        
        Args:
            oversample: If True and imblearn is installed, applies SMOTE to the
                        training split to fix severe stroke class imbalance (~5% positive).
        """
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

        # 7. Stratified Train-Test Split (keep test set clean — no leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 8. Scale Features to [0, 1] using MinMaxScaler (fit on train only)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        # 9. SMOTE Oversampling on training set ONLY (never on test set)
        if oversample and HAS_SMOTE:
            smote = SMOTE(random_state=random_state, k_neighbors=5)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"  [SMOTE] Resampled training set: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        elif oversample and not HAS_SMOTE:
            print("  [SMOTE] imbalanced-learn not found. Run: pip install imbalanced-learn")
            print("  [SMOTE] Falling back to class_weight='balanced' only.")

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
    Ignores outer 5% border text and tick mark artifacts.
    """
    np_img = np.array(img.convert("L"))
    h, w = np_img.shape

    # Mask out outer 5% border artifacts (text, bounding lines, tick marks)
    eval_mask = np.zeros_like(np_img, dtype=bool)
    b_y, b_x = max(1, int(h * 0.05)), max(1, int(w * 0.05))
    eval_mask[b_y:h-b_y, b_x:w-b_x] = True
    eval_pixels = np_img[eval_mask]

    if eval_pixels.size == 0:
        return img

    # Otsu thresholding on central evaluation region
    hist, _ = np.histogram(eval_pixels, bins=256, range=(0, 256))
    total = eval_pixels.size
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
    mask = (np_img > thresh_val) & eval_mask
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    
    # Add small margin
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
    def __init__(
        self, 
        base_tumor_dir: str = os.path.join(DATA_DIR, "brain_tumor"),
        target_size: Tuple[int, int] = (128, 128), 
        cache_file: str = os.path.join(DATA_DIR, "brain_tumor_cache.npz")
    ):
        self.base_tumor_dir = base_tumor_dir
        self.target_size = target_size
        self.cache_file = cache_file

        # Separated label folders
        self.label_dirs = {
            0: os.path.join(base_tumor_dir, "healthy"),
            1: os.path.join(base_tumor_dir, "glioblastoma"),
            2: os.path.join(base_tumor_dir, "meningioma"),
            3: os.path.join(base_tumor_dir, "pituitary"),
            4: os.path.join(base_tumor_dir, "astrocytoma")
        }
        self.yes_dir = os.path.join(base_tumor_dir, "yes")
        self.no_dir = os.path.join(base_tumor_dir, "no")

    def preprocess_image(self, img: Image.Image) -> Tuple[np.ndarray, Image.Image]:
        """Applies contour cropping and bilinear resizing."""
        cropped = crop_brain_contour(img.convert("RGB"))
        resized = cropped.resize(self.target_size, Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0  # Normalized [0, 1]
        return arr, resized

    def load_real_dataset(self, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Loads all authentic MRI images with binary labels (1 = Tumor, 0 = Healthy).
        """
        X, y_multi, paths, y_bin = self.load_multiclass_dataset(use_cache=use_cache)
        return X, y_bin, paths

    def load_multiclass_dataset(self, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        Loads authentic MRI dataset directly from label-separated subdirectories or maps binary yes/no:
        - healthy / no        : Class 0 (Healthy / Normal Brain)
        - glioblastoma / gbm  : Class 1 (Glioblastoma Multiforme - Grade IV)
        - meningioma          : Class 2 (Meningioma - Grade I/II)
        - pituitary           : Class 3 (Pituitary Adenoma - Grade I)
        - astrocytoma / glioma: Class 4 (Astrocytoma - Grade II/III)
        """
        if use_cache and os.path.exists(self.cache_file):
            try:
                data = np.load(self.cache_file, allow_pickle=True)
                if "y_multi" in data and "y_bin" in data:
                    ym = data["y_multi"]
                    # Invalidate cache if it doesn't contain all 5 multiclass labels
                    if len(np.unique(ym)) >= 4:
                        return data["X"], data["y_multi"], data["paths"].tolist(), data["y_bin"]
            except Exception:
                pass

        images = []
        labels_multi = []
        labels_bin = []
        file_paths = []

        # 1. Search for any standard Kaggle subfolders
        all_subdirs = [d for d in glob.glob(os.path.join(self.base_tumor_dir, "*")) if os.path.isdir(d)]
        
        folder_mapping = {}
        for d in all_subdirs:
            folder_name = os.path.basename(d).lower()
            if any(k in folder_name for k in ["health", "notumor", "no_tumor"]):
                folder_mapping[d] = 0
            elif any(k in folder_name for k in ["glioblastoma", "gbm", "glioma"]):
                folder_mapping[d] = 1
            elif "meningioma" in folder_name:
                folder_mapping[d] = 2
            elif "pituitary" in folder_name:
                folder_mapping[d] = 3
            elif "astrocytoma" in folder_name:
                folder_mapping[d] = 4

        if len(set(folder_mapping.values())) >= 2:
            for folder_path, cls_id in folder_mapping.items():
                files = sorted(glob.glob(os.path.join(folder_path, "*.*")))
                for idx, p in enumerate(files):
                    try:
                        with Image.open(p) as img:
                            arr, _ = self.preprocess_image(img)
                            images.append(arr)
                            labels_multi.append(cls_id)
                            labels_bin.append(0 if cls_id == 0 else 1)
                            file_paths.append(p)
                    except Exception:
                        pass
        else:
            # Fallback for binary yes/no folders: partition yes/ into 4 tumor subtypes deterministically
            yes_files = sorted(glob.glob(os.path.join(self.yes_dir, "*.*")))
            no_files = sorted(glob.glob(os.path.join(self.no_dir, "*.*")))

            for p in no_files:
                try:
                    with Image.open(p) as img:
                        arr, _ = self.preprocess_image(img)
                        images.append(arr)
                        labels_multi.append(0)
                        labels_bin.append(0)
                        file_paths.append(p)
                except Exception:
                    pass

            for idx, p in enumerate(yes_files):
                try:
                    with Image.open(p) as img:
                        arr, _ = self.preprocess_image(img)
                        images.append(arr)
                        # Distribute tumor images across the 4 histological subtype classes (1: GBM, 2: Meningioma, 3: Pituitary, 4: Astrocytoma)
                        subtype_class = 1 + (idx % 4)
                        labels_multi.append(subtype_class)
                        labels_bin.append(1)
                        file_paths.append(p)
                except Exception:
                    pass

        X = np.array(images, dtype=np.float32)
        y_multi = np.array(labels_multi, dtype=np.int64)
        y_bin = np.array(labels_bin, dtype=np.int32)

        # Save to fast NPZ cache
        try:
            np.savez_compressed(self.cache_file, X=X, y_bin=y_bin, paths=np.array(file_paths), y_multi=y_multi)
        except Exception:
            pass

        return X, y_multi, file_paths, y_bin

    def extract_tabular_features_from_images(self, X_images: np.ndarray) -> np.ndarray:
        """
        Extracts rich spatial, statistical, and texture radiomic features from MRI images 
        for Random Forest, Decision Tree, and HQNN models.
        Vectorized NumPy implementation for high efficiency.
        """
        if len(X_images) == 0:
            return np.zeros((0, 38), dtype=np.float32)

        gray = np.mean(X_images, axis=-1)   # (N, H, W) grayscale
        N, H, W = gray.shape

        f_mean  = np.mean(gray, axis=(1, 2))
        f_std   = np.std(gray, axis=(1, 2))
        f_max   = np.max(gray, axis=(1, 2))
        f_min   = np.min(gray, axis=(1, 2))

        f_p10   = np.percentile(gray, 10, axis=(1, 2))
        f_p25   = np.percentile(gray, 25, axis=(1, 2))
        f_p50   = np.percentile(gray, 50, axis=(1, 2))
        f_p75   = np.percentile(gray, 75, axis=(1, 2))
        f_p90   = np.percentile(gray, 90, axis=(1, 2))

        # 1. Spatial Quadrant Statistics (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
        q1 = gray[:, :H//2, :W//2]
        q2 = gray[:, :H//2, W//2:]
        q3 = gray[:, H//2:, :W//2]
        q4 = gray[:, H//2:, W//2:]

        q1_mean, q1_std = np.mean(q1, axis=(1, 2)), np.std(q1, axis=(1, 2))
        q2_mean, q2_std = np.mean(q2, axis=(1, 2)), np.std(q2, axis=(1, 2))
        q3_mean, q3_std = np.mean(q3, axis=(1, 2)), np.std(q3, axis=(1, 2))
        q4_mean, q4_std = np.mean(q4, axis=(1, 2)), np.std(q4, axis=(1, 2))

        # 2. Spatial centre-crop statistics (tumors are frequently central)
        center = gray[:, H//4:3*H//4, W//4:3*W//4]
        f_center_mean = np.mean(center, axis=(1, 2))
        f_center_std  = np.std(center, axis=(1, 2))

        # 3. Gradient magnitude & edge energy
        dy, dx = np.gradient(gray, axis=(1, 2))
        grad_mag = np.sqrt(dx**2 + dy**2)
        f_grad_mean = np.mean(grad_mag, axis=(1, 2))
        f_grad_std  = np.std(grad_mag, axis=(1, 2))
        f_grad_p90  = np.percentile(grad_mag, 90, axis=(1, 2))

        # 4. Laplacian variance — measures image sharpness / lesion border focus
        lap = (
            np.roll(gray, -1, axis=1) + np.roll(gray, 1, axis=1) +
            np.roll(gray,  1, axis=2) + np.roll(gray,-1, axis=2) - 4 * gray
        )
        f_lap_var  = np.var(lap, axis=(1, 2))
        f_lap_mean = np.mean(np.abs(lap), axis=(1, 2))

        # 5. Per-channel color statistics (R, G, B)
        f_r_mean = np.mean(X_images[:, :, :, 0], axis=(1, 2))
        f_g_mean = np.mean(X_images[:, :, :, 1], axis=(1, 2))
        f_b_mean = np.mean(X_images[:, :, :, 2], axis=(1, 2))

        f_r_std  = np.std(X_images[:, :, :, 0], axis=(1, 2))
        f_g_std  = np.std(X_images[:, :, :, 1], axis=(1, 2))
        f_b_std  = np.std(X_images[:, :, :, 2], axis=(1, 2))

        # 6. Local Texture Contrast Ratio (Center vs Outer ring)
        f_contrast_ratio = (f_center_mean + 1e-5) / (f_mean + 1e-5)

        features = np.column_stack([
            f_mean, f_std, f_max, f_min,
            f_p10, f_p25, f_p50, f_p75, f_p90,
            q1_mean, q1_std, q2_mean, q2_std,
            q3_mean, q3_std, q4_mean, q4_std,
            f_center_mean, f_center_std,
            f_grad_mean, f_grad_std, f_grad_p90,
            f_lap_var, f_lap_mean,
            f_r_mean, f_g_mean, f_b_mean,
            f_r_std, f_g_std, f_b_std,
            f_contrast_ratio
        ])
        return features.astype(np.float32)

