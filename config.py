"""
Configuration file for Pneumonia Detection Project
Contains all hyperparameters and paths
"""

import os

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Create directories if they don't exist
for dir_path in [RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# DATA PARAMETERS
# ============================================================================
IMAGE_SIZE = 224  # Standard size for pretrained models
BATCH_SIZE = 32
NUM_WORKERS = 4  # For DataLoader (increased for faster loading)
NUM_CLASSES = 1  # Binary classification (sigmoid output)

# DataLoader optimizations
PIN_MEMORY = True  # Faster data transfer to GPU
PERSISTENT_WORKERS = True  # Keep workers alive between epochs

# Class names
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# Used when DATA_DIR contains flat class folders:
# dataset/NORMAL/*.jpeg and dataset/PNEUMONIA/*.jpeg
DATA_SPLIT_RATIOS = {
    'train': 0.70,
    'val': 0.15,
    'test': 0.15,
}

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# CNN Backbone options: 'resnet18', 'resnet50', 'densenet121', 'efficientnet_b0', 'efficientnet_b3'
CNN_BACKBONE = 'efficientnet_b0'  # Current model

# Transformer options: 'vit_base_patch16_224', 'swin_tiny_patch4_window7_224'
TRANSFORMER_BACKBONE = 'vit_base_patch16_224'

# Dropout - REDUCED for EfficientNet
DROPOUT = 0.2  # ← CHANGED from 0.3 (EfficientNet is already efficient)

# For batch experiments (train multiple models)
EXPERIMENT_BACKBONES = ['resnet18', 'efficientnet_b0', 'densenet121']  # Quick test set

# ============================================================================
# TRAINING PARAMETERS - OPTIMIZED FOR EFFICIENTNET
# ============================================================================
LEARNING_RATE = 3e-4  # ← CHANGED from 1e-4 (EfficientNet needs higher LR)
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 10  # ← CHANGED from 7 (give more time to converge)

# Mixed precision training (2-3x speedup, no accuracy loss)
USE_AMP = False  # ← CHANGED to False for CPU training (you're on CPU)

# Advanced training techniques - ADJUSTED FOR CLASS IMBALANCE
USE_FOCAL_LOSS = False  # ← CHANGED to False (use weighted BCE instead)
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 1.0  # ← CHANGED from 2.0 (less aggressive)

USE_LABEL_SMOOTHING = False  # ← CHANGED to False (bad for imbalanced data)
LABEL_SMOOTHING = 0.0  # ← CHANGED from 0.1

USE_COSINE_ANNEALING = True
COSINE_T_MAX = 20
COSINE_ETA_MIN = 1e-6

USE_WARMUP = True
WARMUP_EPOCHS = 3

# Loss function - CRITICAL FIX FOR CLASS IMBALANCE
POS_WEIGHT = 2.5  # ← CHANGED from 1.0 (CRITICAL: gives more weight to pneumonia)

# Test-time augmentation (TTA) for inference
USE_TTA = True
TTA_TRANSFORMS = 5

# ============================================================================
# AUGMENTATION PARAMETERS - REDUCED FOR EFFICIENTNET
# ============================================================================
# Basic augmentation
ROTATION_DEGREES = 15
HORIZONTAL_FLIP_PROB = 0.5

# Style randomization (for bias mitigation)
BRIGHTNESS = 0.3
CONTRAST = 0.3
SATURATION = 0.15

# Advanced augmentation techniques - DISABLED FOR NOW
USE_ADVANCED_AUG = False  # ← CHANGED to False (too much regularization)
RAND_AUG_N = 2
RAND_AUG_M = 9
USE_MIXUP = False  # ← CHANGED to False (disable for initial training)
MIXUP_ALPHA = 0.2
USE_CUTMIX = False  # ← CHANGED to False (disable for initial training)
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5

# ============================================================================
# LUNG SEGMENTATION PARAMETERS
# ============================================================================
USE_LUNG_SEGMENTATION = False
LUNG_MASK_ALPHA = 1.0

# ============================================================================
# EXPLAINABILITY PARAMETERS
# ============================================================================
GRADCAM_LAYER = 'layer4'  # For ResNet-18
NUM_GRADCAM_SAMPLES = 50

# Faithfulness evaluation
MASK_RATIOS = [0.1, 0.3, 0.5, 0.7]

# ============================================================================
# DEVICE
# ============================================================================
DEVICE = 'cuda'  # Will be set to 'cpu' if CUDA not available

# ============================================================================
# RANDOM SEED
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# GRADIENT CLIPPING (NEW - ADDED FOR EFFICIENTNET STABILITY)
# ============================================================================
USE_GRADIENT_CLIPPING = True  # ← NEW: Prevent exploding gradients
GRADIENT_CLIP_VALUE = 1.0  # ← NEW: Max gradient norm
