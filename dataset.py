"""
Dataset loading and preprocessing for Pneumonia Detection
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import cv2

import config

try:
    from advanced_augmentation import get_advanced_transforms
    ADVANCED_AUG_AVAILABLE = True
except ImportError:
    ADVANCED_AUG_AVAILABLE = False


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
SPLITS = ('train', 'val', 'test')


def _list_image_paths(class_dir: str):
    """Return sorted image paths from a class folder."""
    if not os.path.isdir(class_dir):
        return []
    return [
        os.path.join(class_dir, img_name)
        for img_name in sorted(os.listdir(class_dir))
        if img_name.lower().endswith(IMAGE_EXTENSIONS)
    ]


def _has_split_layout(root_dir: str) -> bool:
    return all(
        os.path.isdir(os.path.join(root_dir, split, class_name))
        for split in SPLITS
        for class_name in config.CLASS_NAMES
    )


def _has_flat_layout(root_dir: str) -> bool:
    return all(
        os.path.isdir(os.path.join(root_dir, class_name))
        for class_name in config.CLASS_NAMES
    )


def _split_flat_paths(paths, split: str):
    """Deterministically split a flat class folder into train/val/test paths."""
    if split not in SPLITS:
        raise ValueError(f"Invalid split: {split}. Choose from {SPLITS}")

    paths = list(paths)
    if not paths:
        return []

    ratios = getattr(
        config,
        'DATA_SPLIT_RATIOS',
        {'train': 0.70, 'val': 0.15, 'test': 0.15},
    )
    rng = np.random.default_rng(getattr(config, 'RANDOM_SEED', 42))
    indices = np.arange(len(paths))
    rng.shuffle(indices)
    shuffled = [paths[i] for i in indices]

    train_end = int(len(shuffled) * ratios.get('train', 0.70))
    val_end = train_end + int(len(shuffled) * ratios.get('val', 0.15))

    if split == 'train':
        return shuffled[:train_end]
    if split == 'val':
        return shuffled[train_end:val_end]
    return shuffled[val_end:]


class ChestXrayDataset(Dataset):
    """
    Custom Dataset for Chest X-ray images
    Supports either directory structure:
    dataset/
        NORMAL/
        PNEUMONIA/
    or:
    data/chest_xray/
        train/
            NORMAL/
            PNEUMONIA/
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transform: Optional[transforms.Compose] = None,
                 use_lung_mask: bool = False, lung_segmenter=None):
        """
        Args:
            root_dir: Root directory containing train/val/test folders
            split: One of 'train', 'val', 'test'
            transform: Optional transform to be applied on images
            use_lung_mask: Whether to apply lung segmentation masking
            lung_segmenter: LungSegmenter instance (required if use_lung_mask=True)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.use_lung_mask = use_lung_mask
        self.lung_segmenter = lung_segmenter
        
        if use_lung_mask and lung_segmenter is None:
            raise ValueError("lung_segmenter must be provided when use_lung_mask=True")
        
        self.image_paths = []
        self.labels = []

        if not os.path.exists(root_dir):
            raise ValueError(f"Dataset root {root_dir} does not exist.")

        if _has_split_layout(root_dir):
            split_dir = os.path.join(root_dir, split)
            for label, class_name in enumerate(config.CLASS_NAMES):
                class_dir = os.path.join(split_dir, class_name)
                paths = _list_image_paths(class_dir)
                self.image_paths.extend(paths)
                self.labels.extend([label] * len(paths))
            layout = 'split'
        elif _has_flat_layout(root_dir):
            for label, class_name in enumerate(config.CLASS_NAMES):
                class_dir = os.path.join(root_dir, class_name)
                paths = _split_flat_paths(_list_image_paths(class_dir), split)
                self.image_paths.extend(paths)
                self.labels.extend([label] * len(paths))
            layout = 'flat'
        else:
            raise ValueError(
                f"Dataset root {root_dir} must contain either train/val/test class folders "
                f"or flat {config.CLASS_NAMES} class folders."
            )
        
        print(f"Loaded {len(self.image_paths)} images from {split} set ({layout} layout)")
        print(f"  - NORMAL: {self.labels.count(0)}")
        print(f"  - PNEUMONIA: {self.labels.count(1)}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: Preprocessed image tensor
            label: 0 for NORMAL, 1 for PNEUMONIA
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply lung masking if enabled
        if self.use_lung_mask:
            image_np = np.array(image)
            mask = self.lung_segmenter.segment(image_np)
            masked_image = self.lung_segmenter.apply_mask(image_np, mask, alpha=1.0)
            image = Image.fromarray(masked_image.astype(np.uint8))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split: str = 'train', use_augmentation: bool = True) -> transforms.Compose:
    """
    Get transformation pipeline for different splits
    
    Args:
        split: One of 'train', 'val', 'test'
        use_augmentation: Whether to apply data augmentation (only for train)
    
    Returns:
        transforms.Compose object
    """
    
    # Use advanced augmentation if available and configured
    if ADVANCED_AUG_AVAILABLE and config.USE_ADVANCED_AUG:
        return get_advanced_transforms(split)
    
    # ImageNet normalization (standard for pretrained models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if split == 'train' and use_augmentation:
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomRotation(config.ROTATION_DEGREES),
            transforms.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
            # Style randomization for bias mitigation
            transforms.ColorJitter(
                brightness=config.BRIGHTNESS,
                contrast=config.CONTRAST,
                saturation=config.SATURATION
            ),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Validation/Test transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize
        ])


def get_data_loaders(data_dir: str = config.DATA_DIR, 
                     batch_size: int = config.BATCH_SIZE,
                     num_workers: int = config.NUM_WORKERS,
                     use_augmentation: bool = True,
                     use_lung_mask: bool = False,
                     lung_segmenter=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets
    
    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        use_augmentation: Whether to use data augmentation for training
        use_lung_mask: Whether to apply lung segmentation masking
        lung_segmenter: LungSegmenter instance (required if use_lung_mask=True)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Create datasets
    train_dataset = ChestXrayDataset(
        root_dir=data_dir,
        split='train',
        transform=get_transforms('train', use_augmentation),
        use_lung_mask=use_lung_mask,
        lung_segmenter=lung_segmenter
    )
    
    val_dataset = ChestXrayDataset(
        root_dir=data_dir,
        split='val',
        transform=get_transforms('val', use_augmentation=False),
        use_lung_mask=use_lung_mask,
        lung_segmenter=lung_segmenter
    )
    
    test_dataset = ChestXrayDataset(
        root_dir=data_dir,
        split='test',
        transform=get_transforms('test', use_augmentation=False),
        use_lung_mask=use_lung_mask,
        lung_segmenter=lung_segmenter
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader


def calculate_class_weights(data_dir: str = config.DATA_DIR) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance
    
    Args:
        data_dir: Root directory of the dataset
    
    Returns:
        pos_weight: Weight for positive class (pneumonia)
    """
    train_dataset = ChestXrayDataset(
        root_dir=data_dir,
        split='train',
        transform=get_transforms('train', use_augmentation=False)
    )
    
    num_normal = train_dataset.labels.count(0)
    num_pneumonia = train_dataset.labels.count(1)
    
    # pos_weight = num_negative / num_positive
    pos_weight = num_normal / num_pneumonia
    
    print(f"\nClass distribution in training set:")
    print(f"  NORMAL: {num_normal}")
    print(f"  PNEUMONIA: {num_pneumonia}")
    print(f"  Calculated pos_weight: {pos_weight:.4f}")
    
    return torch.tensor([pos_weight])


if __name__ == "__main__":
    """
    Test the dataset loading
    """
    print("Testing dataset loading...")
    print(f"Expected data directory: {config.DATA_DIR}")
    print("\nPlease ensure the dataset is downloaded and extracted to:")
    print(f"  {config.DATA_DIR}")
    print("\nDirectory structure can be flat:")
    print("  dataset/")
    print("    NORMAL/")
    print("    PNEUMONIA/")
    print("\nOr pre-split:")
    print("  data/chest_xray/")
    print("    train/")
    print("      NORMAL/")
    print("      PNEUMONIA/")
    print("    val/")
    print("      NORMAL/")
    print("      PNEUMONIA/")
    print("    test/")
    print("      NORMAL/")
    print("      PNEUMONIA/")
    print("\n" + "="*70)
    
    # Check if dataset exists
    if not os.path.exists(config.DATA_DIR):
        print(f"\n⚠ Dataset not found at {config.DATA_DIR}")
        print("\nTo download the dataset:")
        print("1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("2. Download and extract to the data/ folder")
        print("3. Run this script again")
    else:
        try:
            # Create data loaders
            train_loader, val_loader, test_loader = get_data_loaders()
            
            # Calculate class weights
            pos_weight = calculate_class_weights()
            
            # Test loading a batch
            print("\n" + "="*70)
            print("Testing batch loading...")
            for images, labels in train_loader:
                print(f"Batch shape: {images.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
                print(f"Labels in batch: {labels.tolist()}")
                break
            
            print("\n✓ Dataset loading successful!")
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
