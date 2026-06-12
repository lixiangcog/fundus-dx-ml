import shutil
import random
from pathlib import Path

def split_dataset(source_dir, output_dir, val_split=0.2, seed=42):
    random.seed(seed)
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    
    classes = [d.name for d in source_path.iterdir() if d.is_dir()]
    
    print(f"Found classes: {classes}")
    
    for class_name in classes:
        class_source = source_path / class_name
        class_train = train_dir / class_name
        class_val = val_dir / class_name
        
        class_train.mkdir(parents=True, exist_ok=True)
        class_val.mkdir(parents=True, exist_ok=True)
        
        images = list(class_source.glob('*'))
        random.shuffle(images)
        
        split_idx = int(len(images) * (1 - val_split))
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        print(f"Processing {class_name}: {len(train_images)} train, {len(val_images)} val")
        
        for img in train_images:
            shutil.copy2(img, class_train / img.name)
            
        for img in val_images:
            shutil.copy2(img, class_val / img.name)
            
    print("Dataset split complete.")

if __name__ == "__main__":
    source = "data/raw/dataset"
    output = "data/processed"
    split_dataset(source, output)
