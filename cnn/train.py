"""
Train a CNN model for Plant Disease Detection using Plant Village Dataset
"""
import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# Load configuration from plantvillage_dataset.json
config_path = "/Users/pabitraghosh/Desktop/Plant Doctor AI/plantvillage_dataset.json"

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get dataset info from JSON
    num_classes = len(config['classes'])  # Number of classes (3 for Potato)
    print(f"\nLoaded configuration from plantvillage_dataset.json")
    print(f"Total classes: {num_classes}")
    for i, cls in enumerate(config['classes']):
        print(f"  Class {i}: {cls['label']} ({cls['name']})")
else:
    num_classes = 15
    config = {'classes': []}

# Additional configuration
DATA_DIR = "/Users/pabitraghosh/Desktop/PlantVillage"  # Directory containing plant images
EPOCHS = 25
BATCH_SIZE = 32
IMG_SIZE = (224, 224)


def create_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes):
    """
    Create a CNN model for plant disease classification
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of classes (disease categories) - loaded from config or default
    
    Returns:
        Compiled Keras Model
    """
    # Base model using MobileNetV2 (efficient for mobile apps)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False
    )
    base_model.trainable = False  # Freeze base model layers
    
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # Connect to base model
    x = base_model(inputs, training=True)
    
    # Add global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dropout to prevent overfitting
    x = layers.Dropout(0.5)(x)
    
    # Dense layers for classification
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer with softmax for multi-class classification
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def load_data():
    """
    Load and preprocess data from the dataset
    
    Expected directory structure:
    PlantVillage/
        class_0/img1.jpg
        class_0/img2.jpg
        ...
    
    Returns:
        x_train, y_train, x_val, y_val: Training and validation data
    """
    import shutil
    import os
    
    # Create train and validation directories
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'validation')
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
    # Split data (80% train, 20% validation)
    import numpy as np
    
    for class_folder in os.listdir(DATA_DIR):
        class_path = os.path.join(DATA_DIR, class_folder)
        if os.path.isdir(class_path):
            files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            
            # Shuffle and split
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(files)
            
            split_idx = int(len(files) * 0.8)
            
            for i, file in enumerate(files):
                src = os.path.join(class_path, file)
                if i < split_idx:
                    dst = os.path.join(train_dir, class_folder, file)
                else:
                    dst = os.path.join(val_dir, class_folder, file)
                
                # Create subdirectory if needed
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)

    # Load training data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'train'),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    # Load validation data
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'validation'),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    return train_ds, val_ds


def get_image_generator():
    """
    Create augmented image generator for training
    
    Returns:
        ImageDataGenerator for augmentation, img_resizer for preprocessing
    """
    # Augmentation for training
    img_gen_train = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Simple rescaling for validation
    img_gen_val = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )

    # Image resizer for preprocessing
    def img_resizer(image):
        return tf.image.resize(image, IMG_SIZE)
    
    return img_gen_train, img_gen_val, img_resizer


def train_model():
    """
    Train the CNN model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("=" * 60)
    print("Plant Disease Detection - Model Training")
    print("=" * 60)

    # Create the model
    model = create_model()
    model.summary()

    # Check if data exists BEFORE loading
    if not os.path.exists(DATA_DIR):
        print(f"\nError: Directory '{DATA_DIR}' not found!")
        print("Please ensure your dataset is in the correct format.")
        return model

    # Load data
    train_ds, val_ds = load_data()

    # Get augmented image generators and img resizer
    img_gen_train, img_gen_val, img_resizer = get_image_generator()

    # Preprocess datasets using autotune prefetching
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (img_resizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (img_resizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CNN for Plant Disease Detection")
    parser.add_argument('--epochs', type=int, default=EPOCHS, 
                        help=f'Number of epochs (default: {EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    
    args = parser.parse_args()

    model, history = train_model()
    print("\n" + "=" * 60)
    print("Training complete! Best model saved as 'best_model.h5'")
    print("=" * 60)
