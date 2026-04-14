#!/usr/bin/env python
"""
Fine-tune MobileNetV2 on Places365-256px dataset

This script:
1. Loads a sample from Places365 dataset
2. Creates a MobileNetV2 model configured for Places365
3. Fine-tunes the model on the dataset
4. Saves the fine-tuned model for use in the app

Usage:
    python train_places365.py [--limit 500] [--epochs 5] [--batch-size 32]
"""

import argparse
import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from places365_utils import (
        load_places365_dataset,
        create_places365_model,
        fine_tune_places365,
        HAS_DATASETS,
    )
except ImportError as e:
    print(f"Error importing Places365 utilities: {e}")
    print("Make sure 'datasets' library is installed: pip install datasets")
    sys.exit(1)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_FOLDER, exist_ok=True)


def prepare_dataset(
    dataset,
    image_size: int = 224,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare dataset for training with image transforms.
    
    Args:
        dataset: Hugging Face dataset object
        image_size: Target image size
        batch_size: Batch size for DataLoader
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    print("Preparing dataset...")
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Convert dataset to tensors
    images = []
    labels = []
    
    for idx, sample in enumerate(dataset):
        if idx % 50 == 0:
            print(f"  Processing: {idx}/{len(dataset)}")
        
        try:
            # Handle different dataset formats
            if isinstance(sample, dict):
                if "image" in sample:
                    img = sample["image"]
                elif "img" in sample:
                    img = sample["img"]
                else:
                    continue
                
                # Convert PIL Image if needed
                if not isinstance(img, Image.Image):
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img.astype('uint8'))
                    else:
                        continue
                
                # Get label
                if "label" in sample:
                    label = sample["label"]
                else:
                    label = idx % 365  # Fallback
            else:
                continue
            
            # Apply transform
            img_tensor = transform(img)
            images.append(img_tensor)
            labels.append(label)
        except Exception as e:
            print(f"  Warning: Skipped sample {idx}: {e}")
            continue
    
    if not images:
        raise ValueError("No valid images found in dataset")
    
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    print(f"Loaded {len(images)} images")
    
    # Split into train/val
    train_size = int(0.8 * len(images))
    train_dataset = TensorDataset(
        images_tensor[:train_size],
        labels_tensor[:train_size]
    )
    val_dataset = TensorDataset(
        images_tensor[train_size:],
        labels_tensor[train_size:]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MobileNetV2 on Places365 dataset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of samples to use (default: 500)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output model path (default: models/places365_mobilenet.pt)"
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.join(MODELS_FOLDER, "places365_mobilenet.pt")
    
    print("=" * 60)
    print("Places365 Model Fine-tuning")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Dataset Split: {args.split}")
    print(f"Max Samples: {args.limit}")
    print(f"Output Path: {args.output}")
    print()
    
    # Load dataset
    print("Loading Places365 dataset...")
    dataset_info = load_places365_dataset(
        split=args.split,
        limit=args.limit
    )
    dataset = dataset_info["dataset"]
    
    # Prepare dataloaders
    train_loader, val_loader = prepare_dataset(
        dataset,
        batch_size=args.batch_size
    )
    
    # Create model
    print("\nCreating Places365 model...")
    model = create_places365_model(num_classes=365, pretrained=True)
    model = model.to(DEVICE)
    print(f"Model: MobileNetV2 (365 classes)")
    
    # Fine-tune
    print("\nStarting fine-tuning...")
    print("-" * 60)
    history = fine_tune_places365(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=args.output,
    )
    
    print("-" * 60)
    print("\nTraining complete!")
    print(f"Model saved to: {args.output}")
    
    # Print summary
    if history.get("val_acc"):
        print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    if history.get("train_loss"):
        print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
    
    print("\nYou can now use this model in the app!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
