"""
Predict plant diseases using trained CNN model with calibrated confidence scoring.
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image

DEFAULT_MODEL_PATH = 'best_model.h5'
METADATA_PATH = 'class_names.json'
FALLBACK_CONFIG_PATH = '/Users/pabitraghosh/Desktop/Plant Doctor AI/plantvillage_dataset.json'

DISEASE_INFO = {
    'Potato___Early_blight': {
        'display_name': 'Potato Early Blight',
        'pathogen': 'Fungus (Alternaria solani)',
        'symptoms': 'Brown/black concentric rings (target-like spots) on older leaves, yellowing margins.',
        'recommendation': 'Remove infected lower leaves. Apply copper-based or mancozeb fungicide. Avoid overhead watering.'
    },
    'Potato___Late_blight': {
        'display_name': 'Potato Late Blight',
        'pathogen': 'Oomycete (Phytophthora infestans)',
        'symptoms': 'Water-soaked dark lesions on leaf tips/margins, white mold on underside in humid conditions.',
        'recommendation': 'CRITICAL: Act immediately. Apply systemic fungicides (chlorothalonil, metalaxyl). Destroy severely infected foliage.'
    },
    'Potato___healthy': {
        'display_name': 'Potato Healthy Leaf',
        'pathogen': 'None (Healthy)',
        'symptoms': 'Uniform green foliage with normal texture and no lesions.',
        'recommendation': 'Maintain regular monitoring, balanced fertilization, and good soil drainage.'
    }
}


class PlantDiseasePredictor:
    """
    Robust predictor for plant disease classification with confidence scoring.
    """
    def __init__(self, model_path=DEFAULT_MODEL_PATH, metadata_path=METADATA_PATH, config_path=FALLBACK_CONFIG_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        self.model = keras.models.load_model(model_path)
        self.input_size = (224, 224)
        
        # Load class names
        self.class_names = []
        self.class_display_names = {}
        
        # Priority 1: class_names.json (generated during training)
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                self.class_names = meta.get('class_names', [])
                for info in meta.get('classes_info', []):
                    self.class_display_names[info['label']] = info.get('display_name', info['label'])
            except Exception as e:
                print(f"Warning loading {metadata_path}: {e}")

        # Priority 2: plantvillage_dataset.json
        if not self.class_names and config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                self.class_names = [c['label'] for c in cfg.get('classes', [])]
                for c in cfg.get('classes', []):
                    self.class_display_names[c['label']] = c.get('name', c['label'])
            except Exception as e:
                print(f"Warning loading {config_path}: {e}")

        # Priority 3: Fallback defaults
        if not self.class_names:
            self.class_names = [
                'Potato___Early_blight',
                'Potato___Late_blight',
                'Potato___healthy'
            ]

        # Determine if model requires manual scaling or has internal preprocessing
        # Newer models have Built-in MobileNetV2 preprocessing (expects [0, 255])
        self.has_internal_preprocessing = any(
            'preprocess_input' in layer.name or 'true_divide' in layer.name
            for layer in self.model.layers
        )

    def preprocess_image(self, img_input):
        """
        Load and format image to (1, 224, 224, 3) in range [0, 255].
        """
        if isinstance(img_input, str):
            img = image.load_img(img_input, target_size=self.input_size)
            img_array = image.img_to_array(img)
        elif isinstance(img_input, Image.Image):
            img = img_input.resize(self.input_size)
            img_array = image.img_to_array(img)
        elif isinstance(img_input, np.ndarray):
            if img_input.shape[:2] != self.input_size:
                img_pil = Image.fromarray(img_input.astype('uint8')).resize(self.input_size)
                img_array = image.img_to_array(img_pil)
            else:
                img_array = img_input.astype('float32')
        else:
            raise ValueError(f"Unsupported image input type: {type(img_input)}")

        # Ensure batch dimension
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)

        # Ensure float32 in [0, 255]
        if np.max(img_array) <= 1.0:
            img_array = img_array * 255.0

        return img_array

    def predict(self, img_path):
        """
        Predict plant disease and return calibrated probability distribution.
        """
        try:
            img_array = self.preprocess_image(img_path)
            raw_predictions = self.model.predict(img_array, verbose=0)[0]
            
            # Sort all predictions by descending probability
            class_prob_pairs = []
            for i, prob in enumerate(raw_predictions):
                label = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
                display = self.class_display_names.get(label, label.replace('___', ' - ').replace('_', ' '))
                class_prob_pairs.append({
                    'class_label': label,
                    'display_name': display,
                    'probability': float(prob),
                    'percentage': float(prob * 100.0)
                })
            
            # Sort descending
            class_prob_pairs.sort(key=lambda x: x['probability'], reverse=True)
            top_prediction = class_prob_pairs[0]
            confidence_pct = top_prediction['percentage']

            # Categorize confidence tier
            if confidence_pct >= 85.0:
                confidence_tier = "HIGH"
                confidence_desc = "High certainty. Symptoms strongly match characteristic pathology patterns."
            elif confidence_pct >= 65.0:
                confidence_tier = "MODERATE"
                confidence_desc = "Moderate certainty. Consider checking for early symptom progression."
            else:
                confidence_tier = "LOW / UNCERTAIN"
                confidence_desc = "Low certainty. Recommend taking a clearer photo in good lighting."

            info = DISEASE_INFO.get(top_prediction['class_label'], {})

            return {
                'success': True,
                'image_path': str(img_path),
                'predicted_class': top_prediction['class_label'],
                'display_name': top_prediction['display_name'],
                'confidence': confidence_pct,
                'confidence_tier': confidence_tier,
                'confidence_description': confidence_desc,
                'pathogen': info.get('pathogen', 'N/A'),
                'symptoms': info.get('symptoms', 'N/A'),
                'recommendation': info.get('recommendation', 'N/A'),
                'all_predictions': class_prob_pairs
            }
        except Exception as e:
            return {
                'success': False,
                'image_path': str(img_path),
                'error': str(e),
                'predicted_class': None,
                'confidence': 0.0,
                'all_predictions': []
            }


def print_prediction_result(result):
    """Print beautifully formatted diagnosis report with confidence metrics"""
    if not result.get('success'):
        print("\n" + "=" * 65)
        print("PREDICTION FAILED")
        print("=" * 65)
        print(f"Error: {result.get('error', 'Unknown error')}")
        return

    print("\n" + "=" * 65)
    print("PLANT DOCTOR AI - DIAGNOSIS REPORT")
    print("=" * 65)
    print(f"Image:            {os.path.basename(result['image_path'])}")
    print(f"Diagnosis:        {result['display_name']}")
    print(f"Confidence:       {result['confidence']:.2f}% [{result['confidence_tier']}]")
    print(f"Certainty Note:   {result['confidence_description']}")
    print(f"Pathogen:         {result['pathogen']}")
    print("-" * 65)
    print(f"Symptoms:         {result['symptoms']}")
    print(f"Recommendation:   {result['recommendation']}")

    print("\n" + "=" * 65)
    print("CLASS PROBABILITY DISTRIBUTION (Ranked):")
    print("=" * 65)
    for p in result['all_predictions']:
        pct = p['percentage']
        bar_len = int(round(pct / 5))  # 20 blocks = 100%
        bar = "█" * bar_len
        print(f"  {p['display_name']:<30} [{bar:<20}] {pct:6.2f}%")
    print("=" * 65 + "\n")


def predict_image(image_path, model_path=DEFAULT_MODEL_PATH):
    """Simple functional API to predict a single image"""
    predictor = PlantDiseasePredictor(model_path=model_path)
    return predictor.predict(image_path)


def main():
    if len(sys.argv) < 2:
        print("=" * 65)
        print("Plant Doctor AI - Image Disease Predictor")
        print("=" * 65)
        print("Usage: python predict.py <path_to_leaf_image.jpg> [model_path]")
        print("\nExample:")
        print("  python predict.py /path/to/leaf.jpg")
        return

    img_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL_PATH

    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    result = predict_image(img_path, model_path=model_path)
    print_prediction_result(result)


if __name__ == "__main__":
    main()
