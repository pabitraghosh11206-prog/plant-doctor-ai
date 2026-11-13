"""
Convert Kaggle PlantVillage dataset (CSV format) 
into required directory structure for training
"""
import pandas as pd
import os

def create_plant_village_structure(csv_path='PlantVillage/'):
    """
    Convert CSV dataset to folder structure:
    
    Expected input: CSV with columns ['Label', 'Image File Name']
    
    Output structure:
    PlantVillage/
        train/
            class_name/
                img.jpg
        validation/
            class_name/
                img.jpg
    """
    # Create directories
    train_dir = "PlantVillage/train"
    val_dir = "PlantVillage/validation"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"Created directories: {train_dir} and {val_dir}")
    
    # Load CSV
    try:
        df = pd.read_csv(os.path.join(csv_path, 'plantvillage.csv'))
        print(f"Loaded dataset with {len(df)} images")
        
        class_names = []
        
        for _, row in df.iterrows():
            label = str(row['Label'])
            image_name = row['Image File Name']
            
            if pd.isna(label):
                continue
            
            # Create folder for this class
            class_folder = os.path.join(train_dir, label)
            os.makedirs(class_folder, exist_ok=True)
            
            # Copy or link the image file
            source_path = row['Image File Name']
            dest_path = os.path.join(class_folder, image_name)
            
            if os.path.exists(source_path):
                import shutil
                shutil.copy(source_path, dest_path)
                class_names.append(label)
            else:
                print(f"Warning: Image not found - {source_path}")
        
        # Create validation split (20% validation)
        unique_classes = list(set(class_names))
        train_samples = int(len(class_names) * 0.8)
        
        import random
        random.shuffle(unique_classes)
        val_classes = set(unique_classes[train_samples:])
        
        for class_name in val_classes:
            class_folder = os.path.join(val_dir, class_name)
            os.makedirs(class_folder, exist_ok=True)
        
        # Move validation files
        for label in unique_classes:
            if label in val_classes:
                img_folder = os.path.join(train_dir, label)
                if os.path.exists(img_folder):
                    for filename in os.listdir(img_folder):
                        src = os.path.join(img_folder, filename)
                        dst = os.path.join(val_dir, label, filename)
                        if os.path.exists(src):
                            import shutil
                            shutil.move(src, dst)
        
        print(f"\n✅ Directory structure created successfully!")
        print(f"\nTrain classes: {train_samples} samples")
        print(f"Validation classes: {len(val_classes)} samples")
        
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {csv_path}/plantvillage.csv")
        print("\nPlease download your dataset first:")
        print("kaggle datasets download -d <dataset-id>")
    except Exception as e:
        print(f"Error converting directory: {e}")


if __name__ == "__main__":
    create_plant_village_structure()
