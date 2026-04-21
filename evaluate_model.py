import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from sklearn.metrics import classification_report
import numpy as np

from shared import get_device, get_inference_transform, build_resnet18

def evaluate_model(data_dir, model_path):
    device = get_device()
    print(f"Using device: {device}")

    # Data transforms (same as validation transforms in train.py)
    data_transforms = get_inference_transform()

    # Load validation dataset
    val_dir = os.path.join(data_dir, 'val')
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found at {val_dir}")
        return

    image_dataset = datasets.ImageFolder(val_dir, data_transforms)
    dataloader = DataLoader(image_dataset, batch_size=32, shuffle=False, num_workers=4)
    class_names = image_dataset.classes
    print(f"Classes: {class_names}")

    # Load model
    model = build_resnet18(len(class_names))

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    evaluate_model(data_dir="data/processed", model_path="best_model.pth")
