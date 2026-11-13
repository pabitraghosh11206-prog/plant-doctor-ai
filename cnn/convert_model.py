"""
Convert trained model to different formats for deployment
- SavedModel (TF 2.x format)
- TensorFlow Lite (.tflite) for mobile deployment
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tflite import tflite_converter_optimizer


def convert_to_savedmodel(model_path='best_model.h5', output_path='saved_model'):
    """
    Convert Keras h5 model to SavedModel format (TF 2.x standard)
    
    Args:
        model_path: Path to the trained h5 model
        output_path: Directory to save the SavedModel
    """
    print("=" * 60)
    print("Converting to SavedModel format...")
    print("=" * 60)

    # Load model
    model = keras.models.load_model(model_path)

    # Convert to SavedModel
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []  # Can enable optimizations for production

    tfsaved.save_saved_model(model, output_directory=output_path)

    print(f"\n✅ SavedModel saved to: {output_path}/")


def convert_to_tflite(model_path='best_model.h5', output_path='plant_doctor.tflite'):
    """
    Convert Keras model to TensorFlow Lite format for mobile deployment
    
    Args:
        model_path: Path to the trained h5 model
        output_path: Output file path for TFLite model
    """
    print("=" * 60)
    print("Converting to TensorFlow Lite (TFLite) format...")
    print("=" * 60)

    # Load model
    model = keras.models.load_model(model_path)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optional optimizations
    # For production, you can enable specific optimizations here
    
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"\n✅ TFLite model saved to: {output_path}")
    print(f"   Model size: {len(tflite_model) / 1024:.2f} KB")


def convert_to_pkl(model_path='best_model.h5', output_path='model.pkl'):
    """
    Convert Keras model to pickle format (simple, but TF version-specific)
    
    Args:
        model_path: Path to the trained h5 model
        output_path: Output file path for pickle model
    """
    print("=" * 60)
    print("Converting to pickle format...")
    print("=" * 60)

    # Load model
    model = keras.models.load_model(model_path)

    # Save as pickle
    tf.keras.models.save_model(model, output_path)

    print(f"\n✅ Model saved to: {output_path}")


def save_weights(model_path='best_model.h5', weights_path='model_weights.h5'):
    """
    Extract and save only model weights (keeps architecture in code)
    
    Args:
        model_path: Path to the trained h5 model
        weights_path: Output file path for weights
    """
    print("=" * 60)
    print("Extracting model weights...")
    print("=" * 60)

    # Load model
    model = keras.models.load_model(model_path)

    # Save weights only
    model.save_weights(weights_path)

    print(f"\n✅ Model weights saved to: {weights_path}")


def main():
    """
    Main function for model conversion
    """
    import sys
    
    print("\n" + "=" * 60)
    print("Plant Disease Detection - Model Converter")
    print("=" * 60)

    if len(sys.argv) < 2:
        # Show options
        print("\nUsage examples:")
        print("  python convert_model.py --savedmodel     -> Convert to SavedModel")
        print("  python convert_model.py --tflite         -> Convert to TFLite (mobile)")
        print("  python convert_model.py --pickle         -> Convert to pickle")
        print("  python convert_model.py --weights        -> Extract weights only")
        return

    args = sys.argv[1]

    if args == '--savedmodel' or args.startswith('savedmodel'):
        convert_to_savedmodel()

    elif args == '--tflite' or args.startswith('tflite'):
        convert_to_tflite()

    elif args == '--pickle' or args.startswith('pickle'):
        convert_to_pkl()

    elif args == '--weights' or args.startswith('weights'):
        save_weights()

    else:
        print("Invalid option. Use one of:")
        print("  --savedmodel, --tflite, --pickle, --weights")


if __name__ == "__main__":
    main()
