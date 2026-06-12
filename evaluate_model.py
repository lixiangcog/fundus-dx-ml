import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

from shared import get_device, get_inference_transform, build_resnet18


def format_confusion_matrix(cm, class_names):
    width = max(len(name) for name in class_names) + 2
    lines = ["Confusion Matrix (rows = true, cols = predicted):"]
    header = " " * width + "".join(f"{name:>{width}}" for name in class_names)
    lines.append(header)
    for name, row in zip(class_names, cm):
        lines.append(f"{name:>{width}}" + "".join(f"{count:>{width}}" for count in row))
    return "\n".join(lines)

def evaluate_model(data_dir, model_path):
    device = get_device()
    print(f"Using device: {device}")

    # Data transforms (same as validation transforms in train.py)
    data_transforms = get_inference_transform()

    val_dir = os.path.join(data_dir, 'val')
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found at {val_dir}")
        return

    image_dataset = datasets.ImageFolder(val_dir, data_transforms)
    dataloader = DataLoader(image_dataset, batch_size=32, shuffle=False, num_workers=4)
    class_names = image_dataset.classes
    print(f"Classes: {class_names}")

    model = build_resnet18(len(class_names))

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded model from {model_path}")

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

    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)
    cm_text = format_confusion_matrix(cm, class_names)

    print("\nClassification Report:")
    print(report)
    print(cm_text)

    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join("reports", f"evaluation_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write(f"Evaluation Report - {timestamp.replace('_', ' ')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Data: {data_dir}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Classes: {class_names}\n\n")
        f.write("=" * 50 + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write(cm_text + "\n")
    print(f"\nSaved report to {report_path}")

if __name__ == "__main__":
    evaluate_model(data_dir="data/processed", model_path="best_model.pth")
