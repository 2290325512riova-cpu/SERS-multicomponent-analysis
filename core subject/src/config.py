"""
Global configuration for the SERS multi-component analysis project.
Strategy 3: MBA treated as a regular component (NOT internal standard).

Current clean mainline:
    - pure benchmark on the complete molar-coded pure-mixture design
    - soil kept as external validation, not mixed into main training
"""
from pathlib import Path

# ====================== Paths ======================
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "总混合光谱"
PURE_DATA_DIR = DATA_ROOT / "纯净物混合光谱"
SOIL_DATA_DIR = DATA_ROOT / "真实样品（土壤）混合光谱"
DATA_DIR = PURE_DATA_DIR
OUTPUT_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

# ====================== Data Version / Runtime Profile ======================
DATA_VERSION = "pure63_mainline"
TASK_PROFILE = "pure_presence_positive_molar"
ACTIVE_SPLIT_FILE = "cv_split_pure63_main.csv"
NEXT_CLEAN_SPLIT_FILE = ACTIVE_SPLIT_FILE
RANDOM_SPLIT_FILE = "cv_split_random_5fold.csv"
RANDOM_SPLIT_QC_BY_FOLD_FILE = "cv_split_random_5fold_qc_by_fold.csv"
RANDOM_SPLIT_QC_BY_STRATUM_FILE = "cv_split_random_5fold_qc_by_stratum.csv"
RANDOM_STRATIFY_COLS = (
    'has_thiram',
    'has_mg',
    'has_mba',
    'c_thiram',
    'c_mg',
    'c_mba',
)

ARTIFACTS_DIR = OUTPUT_DIR / DATA_VERSION
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
MODELS_DIR = ARTIFACTS_DIR / "models"
MAINLINE_FORMAL_DIR = MODELS_DIR / "mainline_formal_experiment"
MAINLINE_SCREENING_DIR = MAINLINE_FORMAL_DIR / "01_candidate_screening"
MAINLINE_OPTIMIZATION_DIR = MAINLINE_FORMAL_DIR / "02_representative_model_optimization"
MAINLINE_LOCKED_DIR = MAINLINE_FORMAL_DIR / "03b_locked_main_results"
MAINLINE_LEAKAGE_DIR = MAINLINE_FORMAL_DIR / "04_leakage_analysis"
MAINLINE_RANDOM_DIR = MAINLINE_LEAKAGE_DIR  # random split results co-locate with leakage
MAINLINE_SHAP_DIR = MAINLINE_FORMAL_DIR / "05_shap_explainability"
PAPER_MAIN_DIR = MODELS_DIR / "paper_main"
PAPER_MAIN_BENCHMARK_DIR = PAPER_MAIN_DIR / "benchmark_random_cv"
PAPER_MAIN_AUGMENTATION_DIR = PAPER_MAIN_DIR / "augmentation_ablation"
PAPER_MAIN_OPTUNA_DIR = PAPER_MAIN_DIR / "optuna_final"
PAPER_MAIN_SHAP_DIR = PAPER_MAIN_DIR / "shap_peak_cluster"
PAPER_MAIN_SOIL_DIR = PAPER_MAIN_DIR / "soil_validation"
PAPER_MAIN_DATA_QUALITY_DIR = PAPER_MAIN_DIR / "data_quality"
FIGURES_DIR = ROOT / "figures" / DATA_VERSION
FIG_EDA = FIGURES_DIR / "eda"
FIG_PREPROCESS = FIGURES_DIR / "preprocessing"
FIG_MODELS = FIGURES_DIR / "models"

# ====================== Preprocessing ======================
WN_MIN = 400.0
WN_MAX = 1800.0
WN_STEP = 1.0
SG_WINDOW = 11
SG_POLY = 3
ALS_LAM = 1e6
ALS_P = 0.01
ALS_NITER = 10
COSMIC_THRESHOLD = 7.0
COSMIC_WINDOW = 5

# ====================== Splitting ======================
N_FOLDS = 5
SEED = 42

# ====================== DL Hyperparameters ======================
DL_LR = 3e-4
DL_EPOCHS = 200
DL_PATIENCE = 20
DL_BATCH = 128
DL_WEIGHT_DECAY = 1e-4
DL_VAL_FRAC = 0.15
DL_SCHEDULER = 'cosine'

# ====================== Substance Naming ======================
SUBSTANCE_CN = {'美': 'Thiram', 'K': 'MG', 'm': 'MBA'}
CONC_LEVELS = [0, 4, 5, 6]
POSITIVE_CONC_LEVELS = [4, 5, 6]
CONC_LEVEL_TO_MOLAR = {
    0: 0.0,
    4: 1e-4,
    5: 1e-5,
    6: 1e-6,
}
CONC_LEVEL_TEXT = {
    0: 'absent',
    4: '1e-4 M',
    5: '1e-5 M',
    6: '1e-6 M',
}

# ====================== Task Definitions ======================
TASKS = [
    {'id': 'P1_thiram_presence', 'name': 'Thiram Presence', 'col': 'has_thiram', 'classes': [0, 1]},
    {'id': 'P2_mg_presence', 'name': 'MG Presence', 'col': 'has_mg', 'classes': [0, 1]},
    {'id': 'P3_mba_presence', 'name': 'MBA Presence', 'col': 'has_mba', 'classes': [0, 1]},
    {
        'id': 'G1_thiram_molar_grade',
        'name': 'Thiram Positive Molar Grade',
        'col': 'c_thiram',
        'classes': [4, 5, 6],
        'filter_col': 'has_thiram',
        'filter_value': 1,
    },
    {
        'id': 'G2_mg_molar_grade',
        'name': 'MG Positive Molar Grade',
        'col': 'c_mg',
        'classes': [4, 5, 6],
        'filter_col': 'has_mg',
        'filter_value': 1,
    },
    {
        'id': 'G3_mba_molar_grade',
        'name': 'MBA Positive Molar Grade',
        'col': 'c_mba',
        'classes': [4, 5, 6],
        'filter_col': 'has_mba',
        'filter_value': 1,
    },
]

TASKS_SUPPLEMENTARY = [
    {'id': 'S1_mixture_order', 'name': 'Mixture Complexity', 'col': 'mixture_order', 'classes': [1, 2, 3]},
]

TASKS_MT_FULL = TASKS + TASKS_SUPPLEMENTARY

# ====================== KAN Hyperparameters ======================
KAN_GRID_SIZE = 5
KAN_SPLINE_ORDER = 3

# ====================== Data Augmentation ======================
AUG_N = 4
AUG_ENABLED = True
MIXUP_ENABLED = True
MIXUP_ALPHA = 0.3
LABEL_SMOOTHING = 0.1

# ====================== Focal Loss ======================
FOCAL_LOSS_GAMMA = 2.0
USE_CLASS_WEIGHT = True

# ====================== Preprocessing Registry ======================
PREPROCESS_VARIANTS = {
    'raw': {
        'name': 'Interpolated raw spectra',
        'short_label': 'Raw',
        'cache_file': 'X_raw.npy',
    },
    'p1': {
        'name': 'Cosmic removal + SG smooth + ALS baseline + SNV',
        'short_label': 'SNV',
        'cache_file': 'X_p1.npy',
    },
    'p2': {
        'name': 'SG smooth + ALS baseline + 1st derivative + SNV',
        'short_label': '1st-Deriv+SNV',
        'cache_file': 'X_p2.npy',
    },
    'p3': {
        'name': 'ALS baseline + Vector normalization',
        'short_label': 'Vector-Norm',
        'cache_file': 'X_p3.npy',
    },
    'p4': {
        'name': 'Cosmic removal + SG smooth + ALS baseline (no normalization)',
        'short_label': 'No-Norm',
        'cache_file': 'X_p4.npy',
    },
    'p5': {
        'name': 'Cosmic removal + SG smooth + ALS baseline + 2nd derivative + SNV',
        'short_label': '2nd-Deriv+SNV',
        'cache_file': 'X_p5.npy',
    },
}

PREPROCESS_TAGS = tuple(PREPROCESS_VARIANTS.keys())
PREPROCESS_NAMES = {tag: cfg['name'] for tag, cfg in PREPROCESS_VARIANTS.items()}
PREPROCESS_SHORT_LABELS = {tag: cfg['short_label'] for tag, cfg in PREPROCESS_VARIANTS.items()}
PREPROCESS_CACHE_FILES = {tag: cfg['cache_file'] for tag, cfg in PREPROCESS_VARIANTS.items()}
