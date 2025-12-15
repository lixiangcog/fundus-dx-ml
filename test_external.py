"""
Test external fundus images against the current model.

Usage:
    python test_external.py                    # Baseline (no enhancements)
    python test_external.py --clahe            # With CLAHE preprocessing
    python test_external.py --tta              # With Test-Time Augmentation
    python test_external.py --clahe --tta      # Both enhancements

Add your images to the folders in test_external_images/:
    - cataract/
    - diabetic_retinopathy/
    - glaucoma/
    - normal/
"""

import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# Config
TEST_DIR = "test_external_images"
CLASS_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def load_model(model_path="best_model.pth"):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, device


def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalize
    fundus images from different sources. Standard practice in DR detection.
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


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_tta_transforms():
    """Test-time augmentation transforms."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ])


def predict_image(model, device, transform, image_path, use_clahe=False, use_tta=False, n_tta=5):
    image = Image.open(image_path).convert("RGB")

    # Apply CLAHE if enabled
    if use_clahe:
        image = apply_clahe(image)

    if use_tta:
        # Test-Time Augmentation: run multiple augmented predictions and average
        tta_transform = get_tta_transforms()
        predictions = []

        # Original image prediction
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions.append(torch.softmax(model(input_tensor), dim=1))

        # Augmented predictions
        for _ in range(n_tta - 1):
            aug_image = tta_transform(image)
            input_tensor = transform(aug_image).unsqueeze(0).to(device)
            with torch.no_grad():
                predictions.append(torch.softmax(model(input_tensor), dim=1))

        # Average all predictions
        probs = torch.mean(torch.stack(predictions), dim=0)
    else:
        # Single prediction
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)

    top_prob, top_idx = torch.topk(probs, 1)
    predicted = CLASS_NAMES[top_idx.item()]
    confidence = top_prob.item()

    all_probs = {CLASS_NAMES[i]: probs[0][i].item() for i in range(len(CLASS_NAMES))}

    return predicted, confidence, all_probs


def main():
    parser = argparse.ArgumentParser(description="Test external fundus images")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE preprocessing")
    parser.add_argument("--tta", action="store_true", help="Use Test-Time Augmentation")
    parser.add_argument("--tta-n", type=int, default=5, help="Number of TTA iterations")
    args = parser.parse_args()

    print("Loading model...")
    model, device = load_model()
    transform = get_transform()
    print(f"Using device: {device}")

    # Show configuration
    enhancements = []
    if args.clahe:
        enhancements.append("CLAHE")
    if args.tta:
        enhancements.append(f"TTA (n={args.tta_n})")

    mode_str = " + ".join(enhancements) if enhancements else "Baseline (no enhancements)"
    print(f"Mode: {mode_str}\n")

    print("=" * 70)
    print(f"EXTERNAL IMAGE TEST RESULTS - {mode_str}")
    print("=" * 70)

    results = []

    for expected_class in CLASS_NAMES:
        class_dir = Path(TEST_DIR) / expected_class

        if not class_dir.exists():
            continue

        images = [f for f in class_dir.iterdir()
                  if f.suffix.lower() in IMAGE_EXTENSIONS]

        if not images:
            continue

        print(f"\n--- {expected_class.upper()} ({len(images)} images) ---")

        for img_path in sorted(images):
            predicted, confidence, all_probs = predict_image(
                model, device, transform, img_path,
                use_clahe=args.clahe,
                use_tta=args.tta,
                n_tta=args.tta_n
            )

            correct = predicted == expected_class
            marker = "✓" if correct else "✗"

            results.append({
                'file': img_path.name,
                'expected': expected_class,
                'predicted': predicted,
                'confidence': confidence,
                'correct': correct
            })

            print(f"\n  {img_path.name}")
            print(f"    Predicted: {predicted} ({confidence:.1%}) {marker}")

            # Show all probabilities if wrong or low confidence
            if not correct or confidence < 0.7:
                sorted_probs = sorted(all_probs.items(), key=lambda x: -x[1])
                print(f"    All probs: ", end="")
                print(", ".join(f"{c}: {p:.1%}" for c, p in sorted_probs))

    # Summary
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        total = len(results)
        avg_confidence = sum(r['confidence'] for r in results) / total

        print("\n" + "=" * 70)
        print(f"SUMMARY: {correct_count}/{total} correct ({correct_count/total:.0%})")
        print(f"Average confidence: {avg_confidence:.1%}")

        # Per-class breakdown
        print("\nPer-class accuracy:")
        for cls in CLASS_NAMES:
            cls_results = [r for r in results if r['expected'] == cls]
            if cls_results:
                cls_correct = sum(1 for r in cls_results if r['correct'])
                cls_avg_conf = sum(r['confidence'] for r in cls_results) / len(cls_results)
                print(f"  {cls}: {cls_correct}/{len(cls_results)} (avg conf: {cls_avg_conf:.1%})")

        # Show misclassifications
        wrong = [r for r in results if not r['correct']]
        if wrong:
            print(f"\nMisclassified ({len(wrong)}):")
            for r in wrong:
                print(f"  {r['file']}: expected {r['expected']}, got {r['predicted']} ({r['confidence']:.1%})")

        print("=" * 70)
    else:
        print("\nNo images found! Add images to the folders in test_external_images/")


if __name__ == "__main__":
    main()
