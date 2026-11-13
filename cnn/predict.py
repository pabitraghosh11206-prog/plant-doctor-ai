"""
Predict plant diseases using trained CNN model
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image


class PlantDiseasePredictor:
    """
    Class to handle predictions for plant disease detection
    """
    
    def __init__(self, model_path='best_model.h5', class_names=None, config_path=None):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the trained h5 model
            class_names: List of disease names (order must match training)
            config_path: Path to JSON config file
        """
        self.model = keras.models.load_model(model_path)
        
        import json
        
        config_path_arg = "/Users/pabitraghosh/Desktop/Plant Doctor AI/plantvillage_dataset.json"
        if class_names is None:
            # Try to load from config file
            if config_path_arg and os.path.exists(config_path_arg):
                with open(config_path_arg, 'r') as f:
                    config_data = json.load(f)
                classes = config_data.get('classes', [])
                if classes:
                    self.class_names = [c['label'] for c in classes]
            else:
                # Default disease categories
                self.class_names = [
                    'Apple___Apple_scab',
                    'Apple___Black_rot',
                    'Apple___Cedar_apple_rust',
                    'Apple___Healthy',
                    'Cherry___Bacterial_canker'
                ]
        else:
            self.class_names = class_names
        
        # Preprocess input to match model expectations
        self.input_size = (224, 224)

    def preprocess_image(self, img_path):
        """
        Load and preprocess an image for prediction
        """
        img = image.load_img(img_path, target_size=self.input_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array

    def predict(self, img_path):
        """
        Predict the disease for a given image
        """
        try:
            img_array = self.preprocess_image(img_path)
            prediction = self.model.predict(img_array)[0]
            class_idx = np.argmax(prediction)
            class_prob = float(prediction[class_idx])
            top_classes = []
            top_probs = []
            
            for i in range(len(prediction)):
                top_classes.append(self.class_names[i])
                top_probs.append(round(float(prediction[i]), 3))
            
            result = {
                'image_path': img_path,
                'predicted_class': self.class_names[class_idx],
                'probability': class_prob * 100,
                'all_predictions': list(zip(top_classes, top_probs)),
                'success': True
            }
            
            return result
            
        except Exception as e:
            return {
                'image_path': img_path,
                'predicted_class': None,
                'probability': 0,
                'all_predictions': [],
                'success': False,
                'error': str(e)
            }


def print_prediction_result(result):
    """Print formatted prediction results"""
    if result['success']:
        print("\n" + "=" * 60)
        print("PREDICTION RESULT:")
        print("=" * 60)
        
        print(f"\nPredicted Disease: {result['predicted_class']}")
        print(f"Confidence: {result['probability']:.2f}%")
        
        print("\nTop Predictions (in descending order):")
        print("-" * 40)
        for cls, prob in result['all_predictions']:
            bar = "█" * int(prob * 20)
            print(f"{cls:<40} [{bar:<20}] {prob:.1f}%")
        
        print("=" * 60)
    else:
        if 'error' in result and result['error']:
            print(f"Error: {result['error']}")
        else:
            print("Prediction failed. Please ensure the model is trained.")


def predict_image(image_path='test.jpg', model_path='best_model.h5', config_path=None):
    """
    Simple function to predict a single image
    """
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train a model first.")
    
    predictor = PlantDiseasePredictor(model_path=model_path, config_path=config_path)
    return predictor.predict(image_path)


def main():
    """Main function to predict from an image file"""
    import sys
    
    if len(sys.argv) < 2:
        # Use default test image if no argument provided
        print("=" * 60)
        print("Plant Disease Detection - Image Prediction")
        print("=" * 60)
        result = predict_image()
        print_prediction_result(result)
    else:
        # Predict from command line argument
        img_path = sys.argv[1]
        print(f"Analyzing image: {img_path}")
        
        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at {img_path}")
            
            result = predict_image(image_path=img_path)
            print_prediction_result(result)
            
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
