"""
Convert trained CNN model to different deployment formats:
- Native Keras format (.keras - recommended modern standard)
- TensorFlow Lite (.tflite - optimized for mobile Flutter/Android/iOS apps)
- TensorFlow SavedModel format (TF 2.x standard for serving / TensorFlow Serving)
- Model weights (.h5 / .weights.h5)
"""
import os
import sys
import argparse
import tensorflow as tf
from tensorflow import keras


def convert_to_keras(model_path='best_model.h5', output_path='best_model.keras'):
    """Save model in native modern Keras format"""
    print("=" * 60)
    print(f"Exporting to modern Keras format (.keras): {output_path}")
    print("=" * 60)
    model = keras.models.load_model(model_path)
    model.save(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Modern Keras model saved to: {output_path} ({size_mb:.2f} MB)")


def convert_to_savedmodel(model_path='best_model.h5', output_path='saved_model'):
    """Convert Keras model to TensorFlow SavedModel format"""
    print("=" * 60)
    print(f"Converting to SavedModel format: {output_path}/")
    print("=" * 60)
    model = keras.models.load_model(model_path)
    try:
        model.export(output_path)
    except Exception:
        tf.saved_model.save(model, output_path)
    print(f"✅ SavedModel exported successfully to: {output_path}/")


def convert_to_tflite(model_path='best_model.h5', output_path='plant_doctor.tflite', quantize=False):
    """
    Convert Keras model to TensorFlow Lite (.tflite) for mobile deployment.
    Optionally apply default quantization for 4x size reduction.
    """
    print("=" * 60)
    print(f"Converting to TFLite format: {output_path} (Quantize={quantize})")
    print("=" * 60)
    model = keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    size_mb = size_kb / 1024
    print(f"✅ TFLite model saved to: {output_path}")
    print(f"   Model size: {size_kb:.2f} KB ({size_mb:.2f} MB)")


def save_weights(model_path='best_model.h5', weights_path='model_weights.weights.h5'):
    """Extract and save only model weights"""
    print("=" * 60)
    print(f"Extracting model weights: {weights_path}")
    print("=" * 60)
    model = keras.models.load_model(model_path)
    model.save_weights(weights_path)
    print(f"✅ Model weights saved to: {weights_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert CNN model to deployment formats")
    parser.add_argument('--model', type=str, default='best_model.h5', help='Input model path')
    parser.add_argument('--savedmodel', action='store_true', help='Export to SavedModel')
    parser.add_argument('--tflite', action='store_true', help='Export to TFLite for mobile')
    parser.add_argument('--quantize', action='store_true', help='Apply dynamic range quantization to TFLite')
    parser.add_argument('--keras', action='store_true', help='Export to .keras format')
    parser.add_argument('--weights', action='store_true', help='Extract weights only')
    parser.add_argument('--all', action='store_true', help='Export all formats')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        print("\nDefaulting to converting --tflite and --keras:")
        args.tflite = True
        args.keras = True

    if args.all or args.keras:
        convert_to_keras(model_path=args.model)
    if args.all or args.tflite:
        convert_to_tflite(model_path=args.model, quantize=args.quantize)
    if args.all or args.savedmodel:
        convert_to_savedmodel(model_path=args.model)
    if args.all or args.weights:
        save_weights(model_path=args.model)


if __name__ == "__main__":
    main()
