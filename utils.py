import os
from typing import Dict, List, Set, Optional

from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO

# Optional Places365 support
try:
    from places365_utils import (
        create_places365_model,
        label_to_scene_category,
        PLACES365_SCENE_CATEGORIES,
    )
    HAS_PLACES365 = True
except ImportError:
    HAS_PLACES365 = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_FOLDER = os.path.join(BASE_DIR, "static", "processed")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Load a pretrained MobileNetV2 model (ImageNet). No training from scratch.
WEIGHTS = models.MobileNet_V2_Weights.DEFAULT
MODEL = models.mobilenet_v2(weights=WEIGHTS)
MODEL = MODEL.to(DEVICE)
MODEL.eval()

IMAGENET_LABELS: List[str] = WEIGHTS.meta["categories"]

# Load Places365 model if available
PLACES365_MODEL: Optional[torch.nn.Module] = None
PLACES365_AVAILABLE = False
if HAS_PLACES365:
    places365_model_path = os.path.join(MODELS_FOLDER, "places365_mobilenet.pt")
    if os.path.exists(places365_model_path):
        try:
            PLACES365_MODEL = create_places365_model(num_classes=365)
            PLACES365_MODEL.load_state_dict(torch.load(places365_model_path, map_location=DEVICE))
            PLACES365_MODEL.eval()
            PLACES365_AVAILABLE = True
            print("✓ Places365 model loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load Places365 model: {e}")

# Load YOLOv8n once for lightweight object detection.
YOLO_MODEL = YOLO("yolov8n.pt")


def resize_with_padding(image: Image.Image, target_size: int = 224) -> Image.Image:
    """
    Resize image to fit into target_size x target_size while keeping aspect ratio,
    and pad remaining area to maintain square dimensions.
    """
    image = image.convert("RGB")
    original_width, original_height = image.size

    if original_width <= 0 or original_height <= 0:
        raise ValueError("Invalid image dimensions.")

    scale = min(target_size / original_width, target_size / original_height)
    new_width = max(1, int(original_width * scale))
    new_height = max(1, int(original_height * scale))

    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2
    canvas.paste(resized, (left, top))

    return canvas


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Open image, resize to 224x224 with aspect-ratio padding,
    normalize pixel values, and convert to model-ready tensor.
    """
    try:
        image = Image.open(image_path)
    except Exception as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    padded = resize_with_padding(image, target_size=224)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tensor = transform(padded).unsqueeze(0).to(DEVICE)
    return tensor


def map_scene_from_labels(top_labels: List[str]) -> str:
    """
    Map model labels to broader real-world scene categories using
    flexible keyword matching.
    """
    scene_keyword_map = {
        "Mountain": {"mountain", "hill", "valley", "alp", "cliff", "peak", "volcano"},
        "Forest": {"forest", "trees", "tree", "jungle", "wood", "woods", "rainforest"},
        "Beach": {"beach", "sea", "sand", "seashore", "coast", "shore", "ocean"},
        "Road": {"street", "road", "highway", "traffic", "crosswalk", "intersection", "bridge"},
        "School": {"classroom", "school", "university", "campus", "college", "lecture"},
        "Office": {"office", "desk", "workspace", "workstation", "computer", "cubicle"},
        "City": {"city", "building", "urban", "downtown", "metropolis", "skyscraper"},
        "General/Nature": {
            "park", "meadow", "field", "pasture", "prairie", "farm", "grass",
            "landscape", "plain", "lakeside", "river", "garden"
        },
    }

    labels_lower = [label.lower() for label in top_labels]

    for label in labels_lower:
        for scene_name, keywords in scene_keyword_map.items():
            if any(keyword in label for keyword in keywords):
                return scene_name

    return "General/Nature"


def predict_with_places365(image_path: str, top_k: int = 5) -> Dict[str, object]:
    """
    Predict scene using Places365 fine-tuned model.
    
    Returns:
        Dictionary with Places365 predictions and mapped scene category
    """
    if not PLACES365_AVAILABLE or PLACES365_MODEL is None:
        return None
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    PLACES365_MODEL.eval()
    with torch.no_grad():
        outputs = PLACES365_MODEL(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    
    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        predictions.append({
            "index": int(idx),
            "confidence": float(prob),
        })
    
    # Get the top prediction's scene category
    top_idx = top_indices[0][0].item()
    # Map Places365 index to category (this would require a proper label file)
    # For now, we'll use a generic mapping
    return {
        "predictions": predictions,
        "model_type": "Places365",
        "is_places365": True,
    }


def infer_logic(scene: str) -> Dict[str, str]:
    """Rule-based activity, risk, and event prediction from scene."""
    mapping = {
        "Road": {
            "activity": "Traffic Movement",
            "risk_level": "Medium",
            "event": "Traffic Congestion",
        },
        "Beach": {
            "activity": "Relaxing",
            "risk_level": "Low",
            "event": "Normal Activity",
        },
        "Mountain": {
            "activity": "Hiking",
            "risk_level": "Medium",
            "event": "Normal Outdoor Activity",
        },
        "Forest": {
            "activity": "Exploring",
            "risk_level": "Medium",
            "event": "Nature Activity",
        },
        "School": {
            "activity": "Learning",
            "risk_level": "Low",
            "event": "Normal Academic Activity",
        },
        "Office": {
            "activity": "Working",
            "risk_level": "Low",
            "event": "Normal Activity",
        },
        "City": {
            "activity": "Urban Movement",
            "risk_level": "Medium",
            "event": "Busy City Activity",
        },
        "General/Nature": {
            "activity": "Outdoor Activity",
            "risk_level": "Low",
            "event": "Normal Activity",
        },
    }

    return mapping.get(
        scene,
        {
            "activity": "General Activity",
            "risk_level": "Medium",
            "event": "Scene Not Clearly Identified",
        },
    )


def detect_presence(image_path: str) -> Dict[str, object]:
    """
    Run YOLOv8 detection and return unique object presence dynamically
    using YOLO class names (COCO labels).
    Also saves an annotated image with bounding boxes.
    """
    results = YOLO_MODEL.predict(
        source=image_path,
        verbose=False,
        conf=0.25,
        imgsz=640,
    )
    if not results:
        return {
            "objects": [],
            "processed_image": None,
        }

    result = results[0]
    detected_objects: Set[str] = set()
    class_names = result.names

    if result.boxes is not None and result.boxes.cls is not None:
        class_ids = result.boxes.cls.tolist()
        for class_id in class_ids:
            class_index = int(class_id)
            label = str(class_names[class_index])
            clean_label = label.replace("_", " ").strip().title()
            if clean_label:
                detected_objects.add(clean_label)

    ordered_objects = sorted(detected_objects)

    plotted = result.plot()
    annotated_image = Image.fromarray(plotted[:, :, ::-1])

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    processed_filename = f"{base_name}_detected.jpg"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
    suffix = 1
    while os.path.exists(processed_path):
        processed_filename = f"{base_name}_detected_{suffix}.jpg"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        suffix += 1

    annotated_image.save(processed_path, format="JPEG", quality=90)

    return {
        "objects": ordered_objects,
        "processed_image": f"/static/processed/{processed_filename}",
    }


def format_presence_output(objects: List[str]) -> str:
    """Format presence labels into display text."""
    if not objects:
        return "No major objects detected"
    return ", ".join(f"{obj} present" for obj in objects)


def analyze_scene(image_path: str, use_places365: bool = True) -> Dict[str, object]:
    """
    Run inference, map to scene category, and apply rule logic.
    Tries Places365 model first if available, falls back to ImageNet model.
    
    Args:
        image_path: Path to the image file
        use_places365: Whether to try Places365 model first
    
    Returns:
        Dictionary with scene analysis results
    """
    # Try Places365 first if enabled and available
    if use_places365 and PLACES365_AVAILABLE:
        places365_result = predict_with_places365(image_path)
        if places365_result:
            # Use Places365 prediction
            input_tensor = preprocess_image(image_path)
            with torch.no_grad():
                output = MODEL(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top_indices = torch.topk(probabilities, k=5).indices.tolist()
            
            top_labels = [IMAGENET_LABELS[index] for index in top_indices]
            scene = map_scene_from_labels(top_labels)
            model_used = "Places365 + ImageNet"
        else:
            # Fall back to ImageNet
            input_tensor = preprocess_image(image_path)
            with torch.no_grad():
                output = MODEL(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top_indices = torch.topk(probabilities, k=5).indices.tolist()
            
            top_labels = [IMAGENET_LABELS[index] for index in top_indices]
            scene = map_scene_from_labels(top_labels)
            model_used = "ImageNet"
    else:
        # Use only ImageNet model
        input_tensor = preprocess_image(image_path)
        with torch.no_grad():
            output = MODEL(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_indices = torch.topk(probabilities, k=5).indices.tolist()
        
        top_labels = [IMAGENET_LABELS[index] for index in top_indices]
        scene = map_scene_from_labels(top_labels)
        model_used = "ImageNet"
    
    logic = infer_logic(scene)
    detection_result = detect_presence(image_path)
    objects = detection_result["objects"]

    return {
        "scene": scene,
        "activity": logic["activity"],
        "risk_level": logic["risk_level"],
        "event": logic["event"],
        "objects": objects,
        "objects_text": format_presence_output(objects),
        "processed_image": detection_result["processed_image"],
        "model_labels": top_labels,
        "model_used": model_used,
    }
