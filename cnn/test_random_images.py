"""
Test Plant Doctor AI CNN Model with Random Images from 3 Classes
Generates Comprehensive Metrics and Multi-Panel Accuracy & Confidence Charts.
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Setup style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def sample_random_images(data_dir, classes, samples_per_class=30, seed=42):
    """
    Selects random images uniformly across all specified classes.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    sampled_data = []
    print(f"\nSampling {samples_per_class} random images from each of the {len(classes)} classes...")
    
    for class_idx, class_name in enumerate(classes):
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.exists(class_folder):
            raise FileNotFoundError(f"Class folder not found: {class_folder}")

        all_files = [
            f for f in os.listdir(class_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]
        
        if len(all_files) == 0:
            raise ValueError(f"No image files found in {class_folder}")

        num_to_sample = min(samples_per_class, len(all_files))
        selected_files = random.sample(all_files, num_to_sample)
        
        print(f"  • {class_name:25}: Sampled {len(selected_files)} / {len(all_files)} images")

        for f in selected_files:
            sampled_data.append({
                'filename': f,
                'filepath': os.path.join(class_folder, f),
                'actual_class': class_name,
                'actual_idx': class_idx
            })

    # Shuffle sampled images so test order is randomized
    random.shuffle(sampled_data)
    print(f"Total randomly sampled test images: {len(sampled_data)}")
    return sampled_data


def run_evaluation(model, sampled_data, classes, input_size=(224, 224)):
    """
    Runs model inference on all sampled images and computes metrics.
    """
    print("\nRunning inference on sampled images...")
    results = []
    
    for i, item in enumerate(sampled_data):
        img_path = item['filepath']
        actual_class = item['actual_class']
        actual_idx = item['actual_idx']

        # Load and preprocess image
        img = Image.open(img_path).convert('RGB').resize(input_size)
        img_arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0) # [1, 224, 224, 3]

        # Prediction
        probs = model.predict(img_arr, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = classes[pred_idx]
        confidence = float(probs[pred_idx] * 100.0)
        is_correct = (pred_class == actual_class)

        results.append({
            'index': i + 1,
            'filename': item['filename'],
            'filepath': img_path,
            'actual_class': actual_class,
            'actual_idx': actual_idx,
            'predicted_class': pred_class,
            'predicted_idx': pred_idx,
            'confidence': confidence,
            'is_correct': is_correct,
            'probabilities': [float(p) for p in probs]
        })

    return results


def compute_metrics(results, classes):
    """
    Computes accuracy, confidence statistics, confusion matrix, and calibration bins.
    """
    total = len(results)
    correct_count = sum(1 for r in results if r['is_correct'])
    overall_accuracy = (correct_count / total) * 100.0
    
    all_confidences = [r['confidence'] for r in results]
    correct_confidences = [r['confidence'] for r in results if r['is_correct']]
    incorrect_confidences = [r['confidence'] for r in results if not r['is_correct']]

    # Per-class metrics
    class_metrics = {}
    num_classes = len(classes)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for idx, c in enumerate(classes):
        c_results = [r for r in results if r['actual_class'] == c]
        c_total = len(c_results)
        c_correct = sum(1 for r in c_results if r['is_correct'])
        c_acc = (c_correct / c_total * 100.0) if c_total > 0 else 0.0
        c_confs = [r['confidence'] for r in c_results]
        
        class_metrics[c] = {
            'total': c_total,
            'correct': c_correct,
            'accuracy': c_acc,
            'mean_confidence': float(np.mean(c_confs)) if c_confs else 0.0,
            'median_confidence': float(np.median(c_confs)) if c_confs else 0.0,
            'min_confidence': float(np.min(c_confs)) if c_confs else 0.0,
            'max_confidence': float(np.max(c_confs)) if c_confs else 0.0,
            'confidences': c_confs
        }

    # Confusion Matrix
    for r in results:
        conf_matrix[r['actual_idx'], r['predicted_idx']] += 1

    # Calibration Bins
    bins = [(50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
    bin_stats = []
    for low, high in bins:
        in_bin = [r for r in results if (r['confidence'] > low if low > 50 else r['confidence'] >= low) and r['confidence'] <= high]
        bin_count = len(in_bin)
        if bin_count > 0:
            bin_acc = (sum(1 for r in in_bin if r['is_correct']) / bin_count) * 100.0
            bin_avg_conf = float(np.mean([r['confidence'] for r in in_bin]))
        else:
            bin_acc = 0.0
            bin_avg_conf = (low + high) / 2.0
        bin_stats.append({
            'range': f"{low}-{high}%",
            'count': bin_count,
            'accuracy': bin_acc,
            'avg_confidence': bin_avg_conf
        })

    summary = {
        'total_samples': total,
        'correct_predictions': correct_count,
        'overall_accuracy': overall_accuracy,
        'mean_confidence': float(np.mean(all_confidences)),
        'median_confidence': float(np.median(all_confidences)),
        'min_confidence': float(np.min(all_confidences)),
        'max_confidence': float(np.max(all_confidences)),
        'mean_confidence_correct': float(np.mean(correct_confidences)) if correct_confidences else 0.0,
        'mean_confidence_incorrect': float(np.mean(incorrect_confidences)) if incorrect_confidences else 0.0,
        'class_metrics': class_metrics,
        'confusion_matrix': conf_matrix.tolist(),
        'bin_stats': bin_stats
    }

    return summary


def create_visualization_chart(summary, results, classes, output_path='model_accuracy_confidence_chart.png'):
    """
    Creates a publication-quality 4-panel visual chart:
    1. Accuracy vs Confidence by Class and Overall (Dual Bar Chart)
    2. Confusion Matrix Heatmap
    3. Confidence Distribution by Class (Boxplot + Swarm Points)
    4. Confidence vs Accuracy Calibration Curve
    """
    print(f"\nGenerating visual charts: {output_path}...")
    
    clean_names = [c.replace('Tomato___', '').replace('_', ' ').title() for c in classes]
    palette = {
        'Early Blight': '#e67e22',
        'Late Blight': '#c0392b',
        'Healthy': '#27ae60',
        'Overall': '#2980b9'
    }

    fig = plt.figure(figsize=(18, 12), facecolor='#ffffff')
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25, top=0.92, bottom=0.08, left=0.07, right=0.95)

    # ----------------------------------------------------
    # Subplot 1: Accuracy vs Mean Confidence by Class (Grouped Bar Chart)
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    categories = clean_names + ['Overall']
    accuracies = [summary['class_metrics'][c]['accuracy'] for c in classes] + [summary['overall_accuracy']]
    confidences = [summary['class_metrics'][c]['mean_confidence'] for c in classes] + [summary['mean_confidence']]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#2ecc71', edgecolor='#27ae60', alpha=0.9)
    rects2 = ax1.bar(x + width/2, confidences, width, label='Mean Confidence (%)', color='#3498db', edgecolor='#2980b9', alpha=0.9)

    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Model Accuracy vs Mean Confidence by Class', fontsize=14, fontweight='bold', pad=12, color='#2c3e50')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11, fontweight='semibold')
    ax1.set_ylim(0, 115)
    ax1.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#dddddd', fontsize=11)
    ax1.axhline(100, color='#95a5a6', linestyle='--', linewidth=0.8, alpha=0.7)

    # Add data labels
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%',
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1e8449')
                     
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%',
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1f618d')

    # ----------------------------------------------------
    # Subplot 2: Confusion Matrix Heatmap
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    cm = np.array(summary['confusion_matrix'])
    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.figure.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    ax2.set_title('B. Confusion Matrix (Random Image Test)', fontsize=14, fontweight='bold', pad=12, color='#2c3e50')
    tick_marks = np.arange(len(classes))
    ax2.set_xticks(tick_marks)
    ax2.set_yticks(tick_marks)
    ax2.set_xticklabels(clean_names, fontsize=11, fontweight='semibold')
    ax2.set_yticklabels(clean_names, fontsize=11, fontweight='semibold')
    ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            row_sum = cm[i].sum()
            pct = (val / row_sum * 100.0) if row_sum > 0 else 0
            text_color = "white" if val > thresh else "black"
            ax2.text(j, i, f"{val}\n({pct:.1f}%)",
                     ha="center", va="center", color=text_color,
                     fontsize=11, fontweight='bold')

    # ----------------------------------------------------
    # Subplot 3: Confidence Distribution (Boxplot + Individual Points)
    # ----------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    data_by_class = [summary['class_metrics'][c]['confidences'] for c in classes]
    
    bp = ax3.boxplot(data_by_class, tick_labels=clean_names, patch_artist=True, widths=0.45,
                     boxprops=dict(facecolor='#ebf5fb', color='#2980b9', linewidth=1.5),
                     whiskerprops=dict(color='#2980b9', linewidth=1.5),
                     capprops=dict(color='#2980b9', linewidth=1.5),
                     medianprops=dict(color='#e74c3c', linewidth=2.0))

    # Overlay individual data points
    for idx, c in enumerate(classes):
        c_results = [r for r in results if r['actual_class'] == c]
        x_jitter = np.random.normal(idx + 1, 0.05, size=len(c_results))
        for r, xj in zip(c_results, x_jitter):
            color = '#27ae60' if r['is_correct'] else '#e74c3c'
            marker = 'o' if r['is_correct'] else 'X'
            s = 40 if r['is_correct'] else 80
            ax3.scatter(xj, r['confidence'], color=color, marker=marker, s=s, alpha=0.75, edgecolors='none', zorder=4)

    ax3.set_title('C. Confidence Distribution & Individual Predictions', fontsize=14, fontweight='bold', pad=12, color='#2c3e50')
    ax3.set_ylabel('Prediction Confidence (%)', fontsize=12, fontweight='bold')
    ax3.set_ylim(40, 105)
    
    # Legend for scatter
    green_dot = plt.Line2D([0], [0], marker='o', color='w', label='Correct Prediction', markerfacecolor='#27ae60', markersize=9)
    red_cross = plt.Line2D([0], [0], marker='X', color='w', label='Misclassification', markerfacecolor='#e74c3c', markersize=10)
    median_line = plt.Line2D([0], [0], color='#e74c3c', linewidth=2, label='Median Confidence')
    ax3.legend(handles=[green_dot, red_cross, median_line], loc='lower left', frameon=True, facecolor='#ffffff', edgecolor='#dddddd', fontsize=10)

    # ----------------------------------------------------
    # Subplot 4: Confidence vs Accuracy Calibration / Reliability Curve
    # ----------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    bin_stats = summary['bin_stats']
    populated_bins = [b for b in bin_stats if b['count'] > 0]
    
    bin_ranges = [b['range'] for b in populated_bins]
    bin_accs = [b['accuracy'] for b in populated_bins]
    bin_confs = [b['avg_confidence'] for b in populated_bins]
    bin_counts = [b['count'] for b in populated_bins]

    # Perfect calibration reference line
    ax4.plot([50, 100], [50, 100], '--', color='#7f8c8d', linewidth=1.5, label='Perfect Calibration (Conf = Acc)')

    # Model curve
    ax4.plot(bin_confs, bin_accs, 'o-', color='#8e44ad', linewidth=2.5, markersize=8, label='Model Calibration')
    
    # Annotate points with sample count
    for b in populated_bins:
        ax4.annotate(f"{b['range']}\n(n={b['count']})",
                     xy=(b['avg_confidence'], b['accuracy']),
                     xytext=(0, 10), textcoords="offset points",
                     ha='center', fontsize=9, fontweight='semibold', color='#512e5f',
                     bbox=dict(boxstyle="round,pad=0.2", fc="#f4ecf7", ec="#8e44ad", lw=0.8, alpha=0.85))

    ax4.set_title('D. Confidence Reliability & Calibration Curve', fontsize=14, fontweight='bold', pad=12, color='#2c3e50')
    ax4.set_xlabel('Average Bin Confidence (%)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Empirical Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_xlim(45, 105)
    ax4.set_ylim(45, 105)
    ax4.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#dddddd', fontsize=10)

    # Overall Figure Title
    fig.suptitle(
        f"Plant Doctor AI - CNN Model Accuracy & Confidence Evaluation\n"
        f"Evaluated on {summary['total_samples']} Random Images Across 3 Tomato Classes | "
        f"Overall Accuracy: {summary['overall_accuracy']:.2f}% | Mean Confidence: {summary['mean_confidence']:.2f}%",
        fontsize=16, fontweight='bold', color='#1a252f', y=0.98
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart successfully saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test Plant Doctor CNN Model Accuracy & Confidence")
    parser.add_argument('--model_path', type=str, default='best_model.h5', help='Path to trained model')
    parser.add_argument('--metadata_path', type=str, default='class_names.json', help='Path to class metadata')
    parser.add_argument('--data_dir', type=str, default='/Users/pabitraghosh/Desktop/PlantVillage/validation', help='Directory containing class folders')
    parser.add_argument('--samples_per_class', type=int, default=30, help='Number of random images to sample per class')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_chart', type=str, default='model_accuracy_confidence_chart.png', help='Output chart filename')
    parser.add_argument('--output_json', type=str, default='random_test_results.json', help='Output JSON filename')
    args = parser.parse_args()

    print("=" * 70)
    print("PLANT DOCTOR AI - CNN MODEL RANDOM IMAGE ACCURACY & CONFIDENCE TEST")
    print("=" * 70)

    # 1. Load metadata & classes
    with open(args.metadata_path, 'r') as f:
        meta = json.load(f)
    classes = meta['class_names']
    print(f"Target Classes ({len(classes)}): {classes}")

    # 2. Sample random images
    sampled_data = sample_random_images(
        data_dir=args.data_dir,
        classes=classes,
        samples_per_class=args.samples_per_class,
        seed=args.seed
    )

    # 3. Load model
    print(f"\nLoading model from {args.model_path}...")
    model = keras.models.load_model(args.model_path)
    print("Model loaded successfully.")

    # 4. Run inference
    results = run_evaluation(model, sampled_data, classes)

    # 5. Compute metrics
    summary = compute_metrics(results, classes)

    # 6. Print console summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Random Images Tested:       {summary['total_samples']}")
    print(f"Correct Predictions:              {summary['correct_predictions']} / {summary['total_samples']}")
    print(f"Overall Accuracy:                 {summary['overall_accuracy']:.2f}%")
    print(f"Mean Confidence (All):            {summary['mean_confidence']:.2f}%")
    print(f"Median Confidence:                {summary['median_confidence']:.2f}%")
    print(f"Confidence Range:                 [{summary['min_confidence']:.2f}%, {summary['max_confidence']:.2f}%]")
    print(f"Mean Confidence on Correct:       {summary['mean_confidence_correct']:.2f}%")
    if summary['mean_confidence_incorrect'] > 0:
        print(f"Mean Confidence on Incorrect:     {summary['mean_confidence_incorrect']:.2f}%")
    print("-" * 70)
    print("PER-CLASS BREAKDOWN:")
    for c in classes:
        cm = summary['class_metrics'][c]
        display_name = c.replace('Tomato___', '').replace('_', ' ').title()
        print(f"  • {display_name:18}: Accuracy: {cm['accuracy']:6.2f}% ({cm['correct']}/{cm['total']}) | Mean Conf: {cm['mean_confidence']:6.2f}% | Range: [{cm['min_confidence']:.1f}% - {cm['max_confidence']:.1f}%]")

    print("-" * 70)
    print("CONFIDENCE CALIBRATION BINS:")
    for b in summary['bin_stats']:
        print(f"  • Bin {b['range']:10}: Count={b['count']:3d} | Empirical Accuracy: {b['accuracy']:6.2f}% | Avg Conf: {b['avg_confidence']:6.2f}%")
    print("=" * 70)

    # 7. Generate visualization chart
    create_visualization_chart(summary, results, classes, output_path=args.output_chart)

    # 8. Save JSON summary
    export_data = {
        'summary': summary,
        'predictions': results
    }
    with open(args.output_json, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"Full prediction details exported to {args.output_json}")


if __name__ == '__main__':
    main()
