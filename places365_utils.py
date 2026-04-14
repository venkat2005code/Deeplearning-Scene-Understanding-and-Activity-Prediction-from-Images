"""
Places365 Scene Classification Integration

This module provides functions to:
- Load the Places365-256px dataset from Hugging Face
- Fine-tune a MobileNetV2 model on Places365 data
- Predict scenes using Places365 labels
"""

import os
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# Try to import datasets library, with fallback
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not installed. Install with: pip install datasets")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Places365 scene categories (365 categories, but we'll use a curated subset)
PLACES365_SCENE_CATEGORIES = {
    "street": "Road/Urban",
    "highway": "Road/Urban",
    "parking_lot": "Urban Infrastructure",
    "beach": "Natural/Beach",
    "sea": "Natural/Water",
    "ocean": "Natural/Water",
    "forest": "Natural/Vegetation",
    "mountain": "Natural/Landscape",
    "valley": "Natural/Landscape",
    "field": "Natural/Open Space",
    "lake": "Natural/Water",
    "river": "Natural/Water",
    "garden": "Natural/Vegetation",
    "park": "Natural/Open Space",
    "office": "Indoor/Work",
    "bedroom": "Indoor/Residential",
    "living_room": "Indoor/Residential",
    "kitchen": "Indoor/Residential",
    "bathroom": "Indoor/Residential",
    "classroom": "Indoor/Education",
    "grocery_store": "Indoor/Commercial",
    "restaurant": "Indoor/Commercial",
    "bar": "Indoor/Social",
    "lobby": "Indoor/Commercial",
    "hospital": "Indoor/Medical",
    "church": "Indoor/Religious",
    "airport": "Indoor/Transportation",
    "train_station": "Indoor/Transportation",
    "stadium": "Outdoor/Sports",
    "playground": "Outdoor/Recreation",
}


def load_places365_dataset(split: str = "train", cache_dir: str = None, limit: int = None) -> Dict:
    """
    Load Places365-256px dataset from Hugging Face.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        cache_dir: Directory to cache the dataset
        limit: Maximum number of samples to load (for testing)
    
    Returns:
        Dictionary with 'images', 'labels', 'label_names'
    """
    if not HAS_DATASETS:
        raise ImportError("Please install the 'datasets' library: pip install datasets")
    
    if cache_dir is None:
        cache_dir = os.path.join(BASE_DIR, ".cache", "places365")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Loading Places365-256px dataset ({split} split)...")
    try:
        dataset = load_dataset(
            "ljnlonoljpiljm/places365-256px",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Limit samples if specified (useful for testing)
    if limit is not None:
        dataset = dataset.select(range(min(len(dataset), limit)))
    
    print(f"Loaded {len(dataset)} samples from Places365")
    
    return {
        "dataset": dataset,
        "num_samples": len(dataset),
    }


def create_places365_model(num_classes: int = 365, pretrained: bool = True) -> nn.Module:
    """
    Create a MobileNetV2 model configured for Places365 classification.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained ImageNet weights
    
    Returns:
        PyTorch model ready for fine-tuning
    """
    if pretrained:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    else:
        model = models.mobilenet_v2(weights=None)
    
    # Replace classifier for Places365 classes
    model.classifier[-1] = nn.Linear(1280, num_classes)
    
    return model.to(DEVICE)


def fine_tune_places365(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 5,
    learning_rate: float = 0.001,
    save_path: str = None,
) -> Dict:
    """
    Fine-tune a MobileNetV2 model on Places365 data.
    
    Args:
        model: PyTorch model to fine-tune
        train_loader: DataLoader with training data
        val_loader: DataLoader with validation data (optional)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        save_path: Path to save the fine-tuned model
    
    Returns:
        Dictionary with training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")
        else:
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}")
    
    # Save model
    if save_path is None:
        save_path = os.path.join(MODELS_FOLDER, "places365_mobilenet.pt")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return history


def label_to_scene_category(label: str) -> str:
    """
    Map a Places365 label to a broader scene category.
    
    Args:
        label: Places365 label string
    
    Returns:
        Broader scene category
    """
    label_lower = label.lower().replace(" ", "_")
    
    # Direct match
    if label_lower in PLACES365_SCENE_CATEGORIES:
        return PLACES365_SCENE_CATEGORIES[label_lower]
    
    # Partial match
    for category_key, category_value in PLACES365_SCENE_CATEGORIES.items():
        if category_key in label_lower or label_lower in category_key:
            return category_value
    
    # Keyword-based categorization
    keywords = {
        "road|street|highway|traffic": "Road/Urban",
        "beach|sea|ocean|shore": "Natural/Water",
        "forest|tree|wood|jungle": "Natural/Vegetation",
        "mountain|hill|valley": "Natural/Landscape",
        "field|meadow|grass|plain": "Natural/Open Space",
        "office|desk|work": "Indoor/Work",
        "bedroom|bed|room|home": "Indoor/Residential",
        "kitchen|dining": "Indoor/Residential",
        "restaurant|bar|cafe|food": "Indoor/Commercial",
        "classroom|school|university": "Indoor/Education",
        "church|temple|shrine": "Indoor/Religious",
        "hospital|medical|clinic": "Indoor/Medical",
        "stadium|court|field": "Outdoor/Sports",
        "park|playground|recreation": "Outdoor/Recreation",
    }
    
    for keyword_pattern, category in keywords.items():
        for keyword in keyword_pattern.split("|"):
            if keyword.lower() in label_lower:
                return category
    
    return "Unknown"


def predict_scene_places365(
    image_path: str,
    model: nn.Module,
    device: torch.device = DEVICE,
    top_k: int = 3,
) -> Dict:
    """
    Predict scene categories for an image using Places365-trained model.
    
    Args:
        image_path: Path to image file
        model: Fine-tuned model
        device: Torch device to use
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image: {e}")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    
    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        pred_dict = {
            "label_index": int(idx),
            "confidence": float(prob),
            "confidence_percent": float(prob) * 100,
        }
        predictions.append(pred_dict)
    
    return {
        "predictions": predictions,
        "image_path": image_path,
        "model_name": "Places365 MobileNetV2",
    }


# Export main functions
__all__ = [
    "load_places365_dataset",
    "create_places365_model",
    "fine_tune_places365",
    "label_to_scene_category",
    "predict_scene_places365",
    "PLACES365_SCENE_CATEGORIES",
]
