# Scene Understanding and Activity Prediction from Images

A beginner-friendly deep learning + web app project using **Flask**, **PyTorch**, **MobileNetV2**, **HTML**, **CSS**, and **JavaScript**.

## 🌟 Features

### Core Capabilities
- Upload image of any size  
- Real-time image preview with analysis
- Automatic preprocessing:
  - Aspect-ratio preserving resize with padding
  - Final size: 224 x 224
  - Normalization for pretrained models
- Scene prediction using MobileNetV2
- Rule-based activity/risk/event prediction
- Object detection with YOLO
- Modern responsive UI with loading animation
- Invalid file handling with user feedback

### Model Enhancements (NEW)
- **Dual-model support**: ImageNet + Places365 scene classification
- **Fine-tuning capability**: Train custom models on Places365 dataset
- **Graceful fallback**: Uses ImageNet if Places365 model not available
- **Model status badge**: Real-time indicator of active model
- **Scene-specific mapping**: 365+ scene categories via Places365

## 🏗️ Project Structure

```
app.py                          # Flask application
utils.py                        # Core analysis utilities
places365_utils.py              # Places365 integration (NEW)
train_places365.py              # Fine-tuning script (NEW)
requirements.txt
templates/
  index.html
static/
  style.css
  script.js
  processed/                    # Detection visualization
uploads/                        # Uploaded images
models/                         # Fine-tuned models (NEW)
```

## Scene Mapping Logic

### ImageNet (Default)
- `street`, `highway` → **Road** 
- `beach`, `sea` → **Beach**
- `office`, `desk` → **Office**
- And more via keyword matching

### Places365 (Optional)
Up to 365 scene categories including:
- Road/Urban, Natural/Beach, Natural/Water
- Natural/Vegetation, Natural/Landscape, Natural/Open Space  
- Indoor/Work, Indoor/Residential, Indoor/Commercial
- Indoor/Education, Indoor/Medical, Outdoor/Recreation
- And 340+ more scenes

### Activity & Risk Prediction Rules

| Scene | Activity | Risk Level | Event |
|-------|----------|-----------|-------|
| Road | Traffic Movement | Medium | Traffic Congestion |
| Beach | Relaxing | Low | Normal Activity |
| Mountain | Hiking | Medium | Outdoor Activity |
| Forest | Exploring | Medium | Nature Activity |
| Office | Working | Low | Normal Activity |

## ⚙️ Setup and Run

### 1. Create and activate virtual environment (recommended)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

### 4. Open in browser

```
http://127.0.0.1:5000
```

## 🚀 Using Places365 Models (Optional)

### Option 1: Quick Test with Sample Data

Fine-tune on a small sample of Places365 data:

```bash
python train_places365.py --limit 500 --epochs 3
```

**Parameters:**
- `--limit`: Number of samples to use (default: 500)
- `--epochs`: Training epochs (default: 5)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--split`: Dataset split - train/validation/test (default: train)
- `--output`: Custom output model path

### Option 2: Full Training

```bash
python train_places365.py --limit 5000 --epochs 10 --batch-size 64
```

### Option 3: Custom Training

```bash
python train_places365.py \
  --limit 10000 \
  --epochs 15 \
  --batch-size 128 \
  --learning-rate 0.0005 \
  --output models/custom_places365.pt
```

**What Happens:**
1. Loads Places365-256px dataset from Hugging Face
2. Converts to PyTorch tensors
3. Applies augmentation (rotation, flip, color jitter)
4. Splits into 80% train / 20% validation
5. Fine-tunes MobileNetV2 classifier
6. Saves model to `models/places365_mobilenet.pt`
7. App automatically uses the new model on next restart

## 📊 Example Output

### ImageNet Model
```
Scene: Road
Activity: Traffic Movement  
Risk Level: Medium
Event: Traffic Congestion
Objects: Car, Person, Traffic Light
Model Used: ImageNet
```

### Places365 Model
```
Scene: Road/Urban  
Activity: Traffic Movement
Risk Level: Medium
Event: Traffic Congestion
Objects: Car, Person, Bus
Model Used: Places365 + ImageNet
```

## 🔧 API Endpoints

### GET `/api/status`
Returns current model status:
```json
{
  "places365_available": true,
  "models": {
    "imagenet": "MobileNetV2 (pretrained)",
    "places365": "Available"
  },
  "status": "Ready"
}
```

### POST `/analyze`
Upload image for analysis:
```bash
curl -X POST -F "image=@photo.jpg" http://127.0.0.1:5000/analyze
```

Response includes: scene, activity, risk_level, event, objects, model_used

## ⚡ Performance Tips

- **GPU Acceleration**: Use CUDA for faster inference (automatic if available)
- **First Run**: Model weights may download automatically (~50MB for ImageNet)
- **Batch Processing**: The fine-tuning script can process thousands of images
- **Memory**: Use `--batch-size 16` on low-memory systems

## 📦 Dependencies

- `Flask≥3.0.0` - Web framework
- `Pillow≥10.0.0` - Image processing  
- `torch≥2.2.0` - Deep learning
- `torchvision≥0.17.0` - Computer vision
- `ultralytics≥8.2.0` - YOLOv8 object detection
- `datasets≥2.14.0` - Hugging Face datasets (for Places365)

## 🎓 Learning Resources

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Places365 Dataset](http://places2.csail.mit.edu/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## ⚠️ Notes

- On first run, pretrained model weights may download automatically
- Places365 model training requires internet connection for dataset download
- Cached datasets are stored in `.cache/places365/`
- Fine-tuned models are saved to `models/` directory
- Scene detection falls back to ImageNet if Places365 not available
- GPU recommended for fast training (CPU still works, will be slower)

## 📝 License

This project is open source and available for educational use.

---

**Last Updated:** March 31, 2026  
**Version:** 2.0 (Places365 Integration)
