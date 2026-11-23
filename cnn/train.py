"""
Train and Fine-Tune a High-Confidence CNN Model for Plant Disease Detection
Using MobileNetV2 with Transfer Learning, Fine-Tuning, and Probability Calibration.
"""
import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.utils.class_weight import compute_class_weight

# Configuration & Defaults
CONFIG_PATH = "/Users/pabitraghosh/Desktop/Plant Doctor AI/plantvillage_dataset.json"
DATA_DIR = "/Users/pabitraghosh/Desktop/PlantVillage"
CLASS_INFO_PATH = "class_names.json"
BEST_MODEL_PATH = "best_model.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
WARMUP_EPOCHS = 10
FINE_TUNE_EPOCHS = 15
TOTAL_EPOCHS = WARMUP_EPOCHS + FINE_TUNE_EPOCHS


def load_dataset_metadata(config_path=CONFIG_PATH):
    """Load configuration from plantvillage_dataset.json if available"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        classes = config.get('classes', [])
        return [c['label'] for c in classes], config
    return None, None


def get_data_augmentation():
    """
    Data augmentation layers for invariant feature learning.
    Prevents overfitting to lab background artifacts and enhances field confidence.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.15),
    ], name="data_augmentation")


def create_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=3):
    """
    Create a robust CNN model using MobileNetV2 with:
    1. Built-in input preprocessing (handles standard [0, 255] RGB images automatically)
    2. Frozen base feature extractor during initial training
    3. Proper inference-mode BatchNormalization (training=False)
    4. Two-stage dense classification head with regularization
    
    Returns:
        (model, base_model)
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    
    # 1. Standard Keras Rescaling: maps [0, 255] -> [-1, 1] (MobileNetV2 standard)
    # Fully serializable across .h5, .keras, TFLite, and SavedModel without custom objects!
    preprocessed = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescaling")(inputs)
    
    # 2. Base feature extractor (MobileNetV2 pre-trained on ImageNet)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False
    )
    base_model.trainable = False
    
    # IMPORTANT: Pass training=False so frozen BatchNormalization layers use learned
    # ImageNet moving statistics rather than fluctuating mini-batch statistics!
    x = base_model(preprocessed, training=False)
    
    # 3. Dense Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(0.3, name="head_dropout1")(x)
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="head_dense1"
    )(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(0.2, name="head_dropout2")(x)
    
    # Output layer with softmax
    outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="plant_doctor_mobilenetv2")
    return model, base_model


def load_datasets(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    """
    Load train and validation datasets with prefetching and caching.
    Supports either pre-split train/val folders or single directory with validation_split.
    """
    train_folder = os.path.join(data_dir, 'train')
    val_folder = os.path.join(data_dir, 'validation')
    
    if os.path.exists(train_folder) and os.path.exists(val_folder):
        print(f"Loading datasets from pre-split directories: {train_folder} and {val_folder}")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_folder,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            label_mode='categorical',
            shuffle=True,
            seed=42
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_folder,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            label_mode='categorical',
            shuffle=False
        )
        class_names = train_ds.class_names
    else:
        print(f"Loading dataset with 80/20 split from {data_dir}")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset='training',
            seed=42,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            label_mode='categorical'
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset='validation',
            seed=42,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            label_mode='categorical'
        )
        class_names = train_ds.class_names

    return train_ds, val_ds, class_names


def compute_dataset_class_weights(train_ds, num_classes):
    """
    Compute balanced class weights to compensate for severe class imbalance
    (e.g. 1000 early blight vs 152 healthy leaves).
    """
    labels = []
    for _, y in train_ds:
        labels.extend(np.argmax(y.numpy(), axis=1))
    labels = np.array(labels)
    
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=labels)
    weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
    print(f"\nComputed balanced class weights:")
    for i, w in weight_dict.items():
        print(f"  Class {i}: weight = {w:.3f} ({np.sum(labels == i)} samples)")
    return weight_dict


def save_metadata(class_names, output_path=CLASS_INFO_PATH):
    """Save class names and metadata for predict.py and external clients"""
    metadata = {
        "class_names": class_names,
        "classes_info": [
            {
                "index": i,
                "label": name,
                "display_name": name.replace("___", " - ").replace("_", " ")
            }
            for i, name in enumerate(class_names)
        ],
        "image_size": list(IMG_SIZE),
        "input_dtype": "float32",
        "input_range": "0-255 RGB",
        "architecture": "MobileNetV2 (fine-tuned)"
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved class metadata to {output_path}")


def train_model(epochs_warmup=WARMUP_EPOCHS, epochs_finetune=FINE_TUNE_EPOCHS, batch_size=BATCH_SIZE):
    """
    Two-Stage Training Routine:
    Stage 1: Train top classification head with frozen base model
    Stage 2: Fine-tune top layers of MobileNetV2 with low learning rate
    """
    print("=" * 65)
    print("Plant Disease Detection - CNN Tuning & Training")
    print("=" * 65)

    # 1. Load data
    train_ds, val_ds, class_names = load_datasets(batch_size=batch_size)
    num_classes = len(class_names)
    print(f"\nDetected {num_classes} classes: {class_names}")

    # Save class metadata
    save_metadata(class_names)

    # Compute class weights
    class_weights = compute_dataset_class_weights(train_ds, num_classes)

    # Apply data augmentation to training data
    augmentation = get_data_augmentation()
    augmented_train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Optimize pipeline with caching and prefetching
    augmented_train_ds = augmented_train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # 2. Build model
    model, base_model = create_model(num_classes=num_classes)
    model.summary()

    # Loss function with label smoothing for probability calibration
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

    # -------------------------------------------------------------
    # STAGE 1: Feature Extraction / Head Training (Warmup)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"STAGE 1: Feature Extraction (Training Top Head for {epochs_warmup} epochs)")
    print("=" * 65)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=['accuracy']
    )

    callbacks_warmup = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-5
        ),
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True
        )
    ]

    history_warmup = model.fit(
        augmented_train_ds,
        validation_data=val_ds,
        epochs=epochs_warmup,
        class_weight=class_weights,
        callbacks=callbacks_warmup,
        verbose=1
    )

    # -------------------------------------------------------------
    # STAGE 2: Fine-Tuning MobileNetV2 Upper Layers
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"STAGE 2: Fine-Tuning (Unfreezing Upper Layers for {epochs_finetune} epochs)")
    print("=" * 65)

    # Unfreeze the base model
    base_model.trainable = True

    # Keep bottom ~100 layers frozen (general low-level edge/texture features)
    # Fine-tune upper layers (high-level leaf disease lesion patterns)
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"Total layers in MobileNetV2: {len(base_model.layers)}")
    print(f"Fine-tuning top {trainable_count} layers (frozen first {fine_tune_at} layers)")

    # Recompile with very low learning rate to avoid destroying pre-trained representations
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=loss_fn,
        metrics=['accuracy']
    )

    callbacks_finetune = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True
        )
    ]

    history_finetune = model.fit(
        augmented_train_ds,
        validation_data=val_ds,
        epochs=epochs_finetune,
        class_weight=class_weights,
        callbacks=callbacks_finetune,
        verbose=1
    )

    print("\n" + "=" * 65)
    print(f"Training Complete! Tuned model saved to: {BEST_MODEL_PATH}")
    keras_path = BEST_MODEL_PATH.replace('.h5', '.keras')
    model.save(keras_path)
    print(f"Also saved modern Keras format to: {keras_path}")
    print("=" * 65)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Tune CNN for Plant Disease Detection")
    parser.add_argument('--warmup_epochs', type=int, default=WARMUP_EPOCHS,
                        help=f'Warmup epochs for head (default: {WARMUP_EPOCHS})')
    parser.add_argument('--finetune_epochs', type=int, default=FINE_TUNE_EPOCHS,
                        help=f'Fine-tuning epochs (default: {FINE_TUNE_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    
    args = parser.parse_args()
    model = train_model(
        epochs_warmup=args.warmup_epochs,
        epochs_finetune=args.finetune_epochs,
        batch_size=args.batch_size
    )
