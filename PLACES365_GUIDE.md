# Places365 Integration Guide

## 🎯 Quick Start

### 1. Run the App (Works Now!)
```bash
# Dependencies already installed
python app.py
# Open: http://127.0.0.1:5000
```

The app will use **ImageNet by default** and show status in header.

### 2. Optional: Fine-tune on Places365

```bash
# Small training (fast, for testing)
python train_places365.py --limit 500 --epochs 3

# Medium training (balanced)  
python train_places365.py --limit 2000 --epochs 5

# Full training (best results)
python train_places365.py --limit 10000 --epochs 15
```

### 3. Restart App to Use Fine-tuned Model
```bash
python app.py
```

Next time you upload an image:
- Status badge will show **"Using Places365 + ImageNet"** (green)
- Results will include: `"model_used": "Places365 + ImageNet"`

---

## 📁 New Files

### `places365_utils.py`
Handles Places365 integration:
- `load_places365_dataset()` - Download & cache Places365 data
- `create_places365_model()` - Create MobileNetV2 for 365 scene classes
- `fine_tune_places365()` - Train the model
- `predict_scene_places365()` - Inference with Places365 model
- `label_to_scene_category()` - Map 365 labels to categories

### `train_places365.py`
Standalone training script:
```bash
python train_places365.py \
  --limit 1000 \          # max images to use
  --epochs 5 \            # training epochs
  --batch-size 32 \       # batch size
  --learning-rate 0.001 \ # learning rate
  --split train \         # dataset split
  --output models/custom.pt
```

---

## 🔄 How the Dual-Model Works

```
User uploads image
       ↓
   App checks for Places365 model file
       ↓
  ┌─── Places365 model exists? ───┐
  │                                │
 YES                              NO
  │                                │
  ↓                                ↓
Use Places365          Use ImageNet (default)
(365 scene classes)    (ImageNet labels)
  │                                │
  ├────────────┬───────────────────┤
  ↓                               ↓
Map to scene category | Same mapping logic
  ↓                               ↓
Apply rule logic ← Both models use same rules →
  ↓                               ↓
Return results with model info
```

---

## 🎨 UI Updates

### Header Status Badge
- **Green indicator**: Places365 model loaded
- **Amber indicator**: ImageNet (default)
- **Tooltip**: Shows "Places365-trained model active" or similar

### Results Panel
New field shows: **"Model Used"**
- `ImageNet` - Default model
- `Places365 + ImageNet` - Fine-tuned Places365 model active

---

## 🚀 Training Parameters Explained

| Parameter | Default | Recommended | Notes |
|-----------|---------|-------------|-------|
| `--limit` | 500 | 2000-10000 | More images = better accuracy |
| `--epochs` | 5 | 10-15 | More epochs = longer training |
| `--batch-size` | 32 | 32-128 | Larger = faster but more memory |
| `--learning-rate` | 0.001 | 0.0005-0.001 | Lower = more stable |

### Quick Decision Tree
- **Fast test (5 min)**: `--limit 500 --epochs 3`
- **Good quality (30 min)**: `--limit 2000 --epochs 5 --batch-size 64`
- **Best quality (2+ hrs)**: `--limit 10000 --epochs 15 --batch-size 128`

---

## 📊 Example Training Output

```
============================================================
Places365 Model Fine-tuning
============================================================
Device: cuda
Epochs: 5
Batch Size: 32
Learning Rate: 0.001
Dataset Split: train
Max Samples: 1000
Output Path: models/places365_mobilenet.pt

Loading Places365-256px dataset...
Loaded 1000 samples from Places365

Preparing dataset...
  Processing: 0/1000
  Processing: 50/1000
  ...
Training samples: 800
Validation samples: 200

Creating Places365 model...
Model: MobileNetV2 (365 classes)

Starting fine-tuning...
------------------------------------------------------------
Epoch 1: Train Loss=3.2456, Val Loss=2.8901, Val Acc=35.50%
Epoch 2: Train Loss=2.1234, Val Loss=1.9876, Val Acc=52.75%
...
Epoch 5: Train Loss=0.8765, Val Loss=0.9234, Val Acc=78.50%

Training complete!
Model saved to: models/places365_mobilenet.pt
Final Validation Accuracy: 78.50%
```

---

## 🐛 Troubleshooting

### Model not loading after training
- **Check**: File exists at `models/places365_mobilenet.pt`
- **Solution**: Run training script again
- **Fallback**: App will use ImageNet automatically

### Training is slow
- **Reduce**: `--limit` or `--batch-size`
- **Enable GPU**: PyTorch will auto-detect CUDA
- **Check**: `--epochs 3` for testing

### Out of memory during training
- **Reduce**: `--batch-size 16` (from default 32)
- **Or reduce**: `--limit 500` (smaller dataset)

### Dataset download fails
- **Check**: Internet connection
- **Cache**: Data stored in `.cache/places365/`
- **Manual**: Delete `.cache` and retry (fresh download)

### Places365 not showing in UI
- **Check**: Model file path `models/places365_mobilenet.pt`
- **Check**: App was restarted after training
- **Check**: Training completed without errors

---

## 🔧 API Reference

### GET `/api/status`
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
Same as before, now includes `"model_used"` field:
```json
{
  "scene": "Road/Urban",
  "activity": "Traffic Movement",
  "risk_level": "Medium",
  "event": "Traffic Congestion",
  "objects": ["Car", "Person"],
  "objects_text": "Car present, Person present",
  "processed_image": "/static/processed/...",
  "model_labels": [...],
  "model_used": "Places365 + ImageNet"
}
```

---

## 📚 Learning Notes

### What is Places365?
- 365 indoor + outdoor scene classes
- 10 million images from MIT/CMU
- Better for scene understanding than ImageNet
- ↳ ImageNet has mostly object classes
- ↳ Places365 has scene/location classes

### Why MobileNetV2?
- Lightweight (14MB vs 100MB+ for larger models)
- Fast inference (~50ms per image)
- Mobile-device compatible
- Good accuracy-speed tradeoff

### Why Transfer Learning?
- Start with pretrained ImageNet weights
- Fine-tune only last layers
- Trains 10-100x faster
- Needs less data
- Better generalization

---

## 🎓 Next Steps

1. **Try the app**: Upload images and see results
2. **Train a model**: `python train_places365.py --limit 1000`
3. **Compare results**: See how Places365 differs from ImageNet
4. **Experiment**: Adjust parameters and retrain
5. **Extend**: Add custom scene categories to mapping

---

## 📞 Support Files

- **App code**: [app.py](app.py)
- **Utils**: [utils.py](utils.py) (updated)
- **Places365**: [places365_utils.py](places365_utils.py) (new)
- **Training**: [train_places365.py](train_places365.py) (new)
- **Full docs**: [README.md](README.md) (updated)

---

**Version**: 2.0 (Places365 Integration)  
**Updated**: March 31, 2026
