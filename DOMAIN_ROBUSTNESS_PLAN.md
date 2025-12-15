# Domain Robustness Improvement Plan

## Problem

Model accuracy is high on images similar to training data but poor on images from different sources (e.g., Google Images). This is a **domain shift** problem - the model learned features specific to the training data's camera type, lighting, resolution, and preprocessing.

## Solution Overview

Three changes based on established practices in diabetic retinopathy deep learning research:

1. **CLAHE Preprocessing** (inference) - Normalize images from any source
2. **Improved Data Augmentation** (training) - Make model robust to variations
3. **Test-Time Augmentation** (inference) - Average multiple predictions for stability

---

## 1. CLAHE Preprocessing

**What:** Contrast Limited Adaptive Histogram Equalization
**Where:** `api/main.py` (inference pipeline)
**Why:** Normalizes contrast and lighting differences between imaging devices. This is the most commonly cited preprocessing step in fundus image research.

### Implementation

```python
import cv2
import numpy as np

def preprocess_fundus(image):
    """
    Apply CLAHE to normalize fundus images from different sources.
    Standard practice in DR detection pipelines.
    """
    img = np.array(image)

    # Convert RGB to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Apply CLAHE to L channel (luminance)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to RGB
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(img)
```

### Changes Required

- Add `opencv-python` to dependencies
- Apply `preprocess_fundus()` before the transform pipeline in `/predict` endpoint

---

## 2. Improved Data Augmentation

**What:** More realistic augmentation aligned with research best practices
**Where:** `train.py`
**Why:** Current augmentation is too mild (only horizontal flip + 10° rotation)

### Current vs Proposed

| Augmentation | Current | Proposed |
|--------------|---------|----------|
| Horizontal Flip | Yes | Yes |
| Vertical Flip | No | No (anatomical orientation matters) |
| Rotation | 10° | 15-20° |
| Color Jitter | No | Yes (brightness, contrast, saturation) |
| Random Crop | No | Yes (resize to 256, crop to 224) |
| Gaussian Blur | No | Optional (simulates focus variation) |

### Implementation

```python
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
```

### Note

After updating augmentation, the model needs to be **retrained** for changes to take effect.

---

## 3. Test-Time Augmentation (TTA)

**What:** Run multiple augmented versions of input, average predictions
**Where:** `api/main.py`
**Why:** Reduces prediction variance on out-of-distribution images

### Implementation

```python
def predict_with_tta(image, model, base_transform, device, n_augments=5):
    """
    Apply test-time augmentation and average predictions.
    """
    tta_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ])

    predictions = []

    # Original prediction
    input_tensor = base_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions.append(torch.softmax(model(input_tensor), dim=1))

    # Augmented predictions
    for _ in range(n_augments - 1):
        aug_image = tta_transforms(image)
        input_tensor = base_transform(aug_image).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions.append(torch.softmax(model(input_tensor), dim=1))

    # Average all predictions
    return torch.mean(torch.stack(predictions), dim=0)
```

### Trade-off

TTA increases inference time by ~n_augments factor. Consider making it optional via query parameter.

---

## Implementation Order

1. **CLAHE preprocessing** - Highest impact, no retraining needed
2. **TTA** - No retraining needed, easy to add
3. **Augmentation + retrain** - Requires retraining but provides lasting improvement

---

## Dependencies to Add

```
opencv-python
```

---

## References

- [Detection of DR in Retinal Fundus Images Using CNN](https://www.mdpi.com/2079-9292/11/17/2740)
- [Hybrid Neural Network for DR Subtype Classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794511/)
- [AI-Enhanced Detection of DR From Fundus Images](https://pmc.ncbi.nlm.nih.gov/articles/PMC11424092/)
