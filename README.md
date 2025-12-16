# Retina Fundus Classification

Lightweight pipeline to train a fundus image classifier (cataract, glaucoma, normal, diabetic retinopathy) with optional early stopping on validation accuracy.

**[Live Demo](https://fundus-dx-ml.vercel.app/)**

![RetinaAI Web Interface](screenshot.png)

## Architecture

- **Frontend**: Hosted on [Vercel](https://fundus-dx-ml.vercel.app/). Built with React and Tailwind CSS.
- **Backend**: Hosted on [Hugging Face Spaces](https://huggingface.co/spaces/Depryx/fundus-ml). Serves the PyTorch model via FastAPI.
- **Model**: ResNet18 fine-tuned on fundus images.

## Local Development

### Setup
```bash
pip install -r requirements.txt
```
Python 3.10+ recommended; GPU or Apple Silicon preferred but CPU works.

### Prepare splits
```bash
python organize_data.py
```
Creates `data/processed/train` and `data/processed/val` with an 80/20 split per class (seeded for repeatability).

### Train
```bash
python train.py \
  --data-dir data/processed \
  --num-epochs 25 \
  --batch-size 32 \
  --target-acc 0.90
```
- Uses ResNet18 with ImageNet weights; saves the best weights to `best_model.pth`.
- Training uses horizontal flip + small rotation; validation is resized + normalized only.

### Evaluate
```bash
python evaluate_model.py
```
Runs the saved model against the validation set and prints a classification report with precision, recall, and F1-score per class. Loads `best_model.pth` and reads images from `data/processed/val/`.

### Run Locally

**Backend API**:
```bash
pip install fastapi uvicorn[standard]
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Keep `best_model.pth` in the project root.

**Frontend**:
```bash
cd frontend
npm install
npm run dev
```
UI expects the API at `http://localhost:8000/predict` by default.

## Data layout
- Place images under `data/raw/dataset/` with one folder per class: `cataract/`, `glaucoma/`, `normal/`, `diabetic_retinopathy/`.
- Processed train/val splits are written to `data/processed/`.

## How it works
- **Technical**: Uses torchvision `ImageFolder` to load class directories; applies light augmentations; fine-tunes ResNet18's final layer on your classes with SGD + StepLR; tracks best validation accuracy and saves weights.
- **Plain English**: It takes your labeled folders of eye photos, lightly jumbles the training pictures, and teaches a pre-trained model to tell the conditions apart. It checks progress on a validation set each epoch, keeps the best version, and can stop once it's accurate enough.

## Domain Robustness Testing

When testing on images from different sources (e.g., Google Images), models can struggle due to **domain shift** - differences in camera type, lighting, resolution, and preprocessing between training and test images.

### Test External Images

```bash
# Add your images to test_external_images/{cataract,diabetic_retinopathy,glaucoma,normal}/
python test_external.py              # Baseline
python test_external.py --clahe      # With CLAHE preprocessing
python test_external.py --tta        # With Test-Time Augmentation
python test_external.py --clahe --tta
```

### Techniques Tested

| Technique | Description |
|-----------|-------------|
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization - normalizes lighting/contrast differences between imaging devices |
| **TTA** | Test-Time Augmentation - runs multiple augmented versions of the input and averages predictions |

### Our Findings

| Mode | Accuracy | Avg Confidence |
|------|----------|----------------|
| Baseline | 7/7 (100%) | ~89% |
| CLAHE | 7/7 (100%) | 87% |
| **TTA** | 7/7 (100%) | **92%** |
| CLAHE + TTA | 6/7 (86%) | 86% |

**Key takeaways:**

1. **TTA alone works best** - improves confidence without breaking predictions
2. **CLAHE is mixed** - helps some classes (DR confidence increased) but hurts others (cataract confidence dropped)
3. **Combining both made things worse** - over-processing caused a misclassification
4. **To benefit from CLAHE**, you'd need to retrain with CLAHE applied to training images so the model learns the CLAHE-processed distribution

**Recommendation:** Use `--tta` flag for inference on external images. See `DOMAIN_ROBUSTNESS_PLAN.md` for implementation details.
