"""
Evaluation and Confidence Calibration Analysis for Plant Doctor AI CNN Model
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix


def compute_calibration_metrics(y_true, y_probs, num_bins=10):
    """
    Compute Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and Brier Score.
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_stats = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            abs_diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += abs_diff * prop_in_bin
            mce = max(mce, abs_diff)
            bin_stats.append({
                'bin_range': f"({bin_lower:.2f}, {bin_upper:.2f}]",
                'count': int(np.sum(in_bin)),
                'accuracy': float(accuracy_in_bin),
                'confidence': float(avg_confidence_in_bin),
                'error': float(abs_diff)
            })

    # Brier Score (lower is better, 0 is perfect)
    num_classes = y_probs.shape[1]
    y_true_one_hot = np.eye(num_classes)[y_true]
    brier_score = np.mean(np.sum((y_probs - y_true_one_hot) ** 2, axis=1))

    return {
        'ece': float(ece),
        'mce': float(mce),
        'brier_score': float(brier_score),
        'bin_stats': bin_stats
    }


def evaluate_model(model_path='best_model.h5', data_dir='/Users/pabitraghosh/Desktop/PlantVillage/validation', is_legacy_model=False):
    """
    Run full evaluation on validation dataset.
    """
    print("=" * 65)
    print(f"Evaluating Model: {model_path}")
    print("=" * 65)

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None

    model = keras.models.load_model(model_path)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(224, 224),
        batch_size=32,
        shuffle=False,
        label_mode='categorical'
    )
    class_names = val_ds.class_names

    y_true_list = []
    images_list = []
    for x, y in val_ds:
        images_list.append(x.numpy())
        y_true_list.append(np.argmax(y.numpy(), axis=1))

    images = np.concatenate(images_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)

    # For legacy model, it expected un-normalized images or had no built-in preprocessing
    y_probs = model.predict(images, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)

    correct = (y_pred == y_true)
    accuracy = float(np.mean(correct))

    print(f"\nOverall Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Total Validation Samples: {len(y_true)}")

    print("\n" + "=" * 65)
    print("CLASSIFICATION REPORT:")
    print("=" * 65)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("=" * 65)
    print("CONFUSION MATRIX:")
    print("=" * 65)
    cm = confusion_matrix(y_true, y_pred)
    label_str = "True \\ Pred"
    header = f"{label_str:<24} | " + " | ".join([f"{name[:15]:<15}" for name in class_names])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<24} | " + " | ".join([f"{val:<15}" for val in row])
        print(row_str)

    print("\n" + "=" * 65)
    print("CONFIDENCE & CALIBRATION ANALYSIS:")
    print("=" * 65)
    print(f"Mean Confidence on ALL predictions:       {np.mean(confidences) * 100:.2f}%")
    if np.sum(correct) > 0:
        print(f"Mean Confidence on CORRECT predictions:   {np.mean(confidences[correct]) * 100:.2f}%")
        print(f"Min Confidence on CORRECT predictions:    {np.min(confidences[correct]) * 100:.2f}%")
    if np.sum(~correct) > 0:
        print(f"Mean Confidence on INCORRECT predictions: {np.mean(confidences[~correct]) * 100:.2f}%")
        print(f"Max Confidence on INCORRECT predictions: {np.max(confidences[~correct]) * 100:.2f}%")

    cal_metrics = compute_calibration_metrics(y_true, y_probs)
    print(f"Expected Calibration Error (ECE):         {cal_metrics['ece'] * 100:.2f}%")
    print(f"Maximum Calibration Error (MCE):          {cal_metrics['mce'] * 100:.2f}%")
    print(f"Brier Score (lower is better):            {cal_metrics['brier_score']:.4f}")

    print("\nCalibration Bins Breakdown:")
    for b in cal_metrics['bin_stats']:
        print(f"  Bin {b['bin_range']:<16} Count: {b['count']:<4} Acc: {b['accuracy']*100:6.2f}% | Conf: {b['confidence']*100:6.2f}% | Gap: {b['error']*100:5.2f}%")

    # High / Med / Low confidence breakdown
    high_conf = confidences >= 0.85
    med_conf = (confidences >= 0.65) & (confidences < 0.85)
    low_conf = confidences < 0.65
    print("\nConfidence Tiers:")
    print(f"  High Confidence (>= 85%):    {np.sum(high_conf)} ({np.mean(high_conf)*100:.1f}%) with {np.mean(correct[high_conf])*100:.2f}% accuracy" if np.sum(high_conf) > 0 else "  High: 0")
    print(f"  Medium Confidence (65-84%):  {np.sum(med_conf)} ({np.mean(med_conf)*100:.1f}%) with {np.mean(correct[med_conf])*100:.2f}% accuracy" if np.sum(med_conf) > 0 else "  Medium: 0")
    print(f"  Low Confidence (< 65%):      {np.sum(low_conf)} ({np.mean(low_conf)*100:.1f}%) with {np.mean(correct[low_conf])*100:.2f}% accuracy" if np.sum(low_conf) > 0 else "  Low: 0")

    return {
        'accuracy': accuracy,
        'mean_confidence': float(np.mean(confidences)),
        'correct_confidence': float(np.mean(confidences[correct])) if np.sum(correct) > 0 else 0.0,
        'incorrect_confidence': float(np.mean(confidences[~correct])) if np.sum(~correct) > 0 else 0.0,
        'ece': cal_metrics['ece'],
        'brier_score': cal_metrics['brier_score']
    }


if __name__ == '__main__':
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else 'best_model.h5'
    evaluate_model(model_file)
