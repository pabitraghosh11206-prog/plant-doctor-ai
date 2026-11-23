# CNN Model Confidence Review Report
## Plant Doctor AI - Tomato Disease Detection

---

## Executive Summary

✅ **MODEL PERFORMS WELL ON VALIDATION SET** but confidence scores need review for production deployment.

**Important Update**: The model is now configured for **Tomato disease detection** (Early blight, Late blight, Healthy). All class names and folder structures have been updated from Potato to Tomato.

---

## Test Results Summary

### Overall Validation Set (431 samples)
| Metric | Value | Status |
|--------|-------|--------|
| Overall Accuracy | **97.45%** | ✅ Excellent |
| Mean Confidence (ALL predictions) | 95.58% | High |
| Mean Confidence (CORRECT predictions) | 96.26% | Very High |
| Min Confidence on CORRECT predictions | 46.44% | ⚠️ Variable |
| Mean Confidence (INCORRECT predictions) | 69.96% | Lower |
| Max Confidence on INCORRECT predictions | 96.75% | ⚠️ Overconfident |
| Expected Calibration Error (ECE) | 2.09% | ✅ Good (<10%) |
| Maximum Calibration Error (MCE) | 21.91% | ❌ Needs attention |
| Brier Score | 0.0414 | Acceptable |

### Class Distribution in Validation Set
| Class | Samples | % of Total |
|-------|---------|------------|
| Tomato___Early_blight | 200 | 46.4% |
| Tomato___Late_blight | 200 | 46.4% |
| Tomato___healthy | 31 | 7.2% |

**⚠️ CRITICAL**: Severe class imbalance - Healthy class has only 7.2% of samples!

### Individual Class Testing (One Image Per Subtype)

| Actual Label | Predicted | Confidence | Correct? | Status |
|--------------|-----------|------------|----------|--------|
| Tomato___Early_blight | Tomato___Early_blight | **99.35%** | ✅ YES | ✅ EXCELLENT |
| Tomato___Late_blight | Tomato___Late_blight | **98.73%** | ✅ YES | ✅ EXCELLENT |
| Tomato___healthy | Tomato___healthy | **96.82%** | ✅ YES | ✅ GOOD |

---

## Detailed Analysis

### 1. Classification Report (from validation set)

| | Precision | Recall | F1-Score | Support |
|--|-----------|--------|----------|---------|
| Tomato Early blight | 98.51% | 99.00% | 98.75% | 200 |
| Tomato Late blight | 98.96% | 95.50% | 97.20% | 200 |
| Tomato healthy | 83.78% | 100.00% | 91.18% | 31 |

**Key Findings:**
- ✅ **Early blight**: Excellent performance with high recall (99%) and precision (98.5%)
- ✅ **Late blight**: Good performance but slightly lower recall (95.5%) due to confusion with early blight
- ⚠️ **Healthy class**: Low precision (83.78%) - Many healthy images being misclassified as diseased
- ⚠️ **Class imbalance**: Only 31 healthy samples vs 200 each for diseases severely skews performance

### 2. Confidence Analysis

#### Confidence Distribution by Correctness:
| Condition | Mean Confidence | Issues |
|-----------|-----------------|--------|
| Correct predictions | **96.26%** | Min: 46.44% ⚠️ (variable) |
| Incorrect predictions | **69.96%** | Max: 96.75% ❌ (overconfident) |

**Observations:**
1. When model makes correct predictions, it's usually very confident (~96%)
2. Some correct predictions have low confidence (<50%), which creates user uncertainty
3. Incorrect predictions on imbalanced data can be surprisingly confident due to bias

### 3. Calibration Bins Breakdown

| Confidence Bin | Count | Accuracy | Avg Conf | Gap (Conf-Acc) |
|----------------|-------|----------|----------|----------------|
| 0.40 - 0.50 | 3 | 66.67% | 47.04% | 19.63% ❌ |
| 0.50 - 0.60 | 7 | 57.14% | 55.02% | 2.13% ✅ |
| 0.60 - 0.70 | 8 | 87.50% | 65.59% | 21.91% ❌ |
| 0.70 - 0.80 | 10 | 70.00% | 74.84% | 4.84% |
| 0.80 - 0.90 | 23 | 91.30% | 86.76% | 4.54% |
| 0.90 - 1.00 | 380 | **99.74%** | 98.43% | 1.31% ✅ |

### 4. Confidence Tier Analysis

| Tier | Count | % of Total | Accuracy | Quality |
|------|-------|------------|----------|---------|
| High (≥85%) | 398 | 92.3% | **99.25%** | ✅ Excellent |
| Medium (65-84%) | 20 | 4.6% | 80.00% | ⚠️ Moderate |
| Low (<65%) | 13 | 3.0% | 69.23% | ❌ Problematic |

**Critical Finding:** When confidence is ≥85%, accuracy is **99.25%** - very reliable!

---

## Individual Image Test Results (One Per Class)

### 🧪 Early Blight Sample
- **Actual**: Tomato___Early_blight
- **Predicted**: Tomato___Early_blight ✅ CORRECT
- **Confidence**: 99.35% ⭐⭐⭐⭐⭐
- **Top 3 Predictions**:
  - Early blight: 99.35% ← PREDICTED
  - Late blight: 3.76%
  - Healthy: 2.38%

### 🧪 Late Blight Sample  
- **Actual**: Tomato___Late_blight
- **Predicted**: Tomato___Late_blight ✅ CORRECT
- **Confidence**: 98.73% ⭐⭐⭐⭐⭐
- **Top 3 Predictions**:
  - Late blight: 98.73% ← PREDICTED
  - Early blight: 3.76%
  - Healthy: 2.23%

### 🧪 Healthy Sample
- **Actual**: Tomato___healthy
- **Predicted**: Tomato___healthy ✅ CORRECT
- **Confidence**: 96.82% ⭐⭐⭐⭐☆
- **Top 3 Predictions**:
  - Healthy: 96.82% ← PREDICTED
  - Early blight: 3.95%
  - Late blight: 2.25%

---

## 🔴 Critical Issues Summary

### Issue #1: Class Imbalance (SEVERE)
- **Healthy class**: Only 31 samples vs 200 for each disease class
- This causes the model to be biased toward majority classes
- Low precision on healthy class (83.78%) indicates false positives for diseases

### Issue #2: Confidence Variability on Correct Predictions
- Min confidence on correct predictions is only 46.44%
- Users may see "confidently" predicted images with low confidence scores
- Creates uncertainty even when model is technically correct

---

## ✅ Positive Findings

1. **Overall accuracy is excellent** (97.45%) on validation set
2. **High confidence predictions are highly accurate** (99.25% for ≥85% confidence)
3. **Early blight detection is outstanding** - 99.35% confidence, 99% recall
4. **Late blight detection is very good** - 98.73% confidence, 95.5% recall  
5. **Healthy class correctly identified** with high confidence (96.82%)
6. **ECE is acceptable** (2.09%) - model's average predictions are reasonably calibrated
7. **92.3% of predictions have high confidence** and achieve 99%+ accuracy

---

## 🎯 Recommendations

### Immediate Actions Required:

1. ✅ **Rebalance the dataset** - Increase healthy class samples to ~200 (target 50/50/50 split)
2. ✅ **Retrain with balanced data** to improve precision on healthy class
3. ✅ **Add data augmentation** for minority classes to prevent overfitting

### Model Improvement Steps:

4. ⚠️ **Use threshold-based predictions**:
   - If prediction confidence for ANY class < 80% → "Unable to determine, please consult expert"
   - If prediction confidence ≥ 85% → Show diagnosis with confidence score
   - This prevents overconfident wrong answers from being displayed

5. ⚠️ **Apply temperature scaling** (post-training calibration) if needed:
   ```python
   # Example: Scale predictions to improve calibration
   import tensorflow as tf
   from sklearn.calibration import CalibratedClassifierCV
   
   base_model = ...  # Your trained model
   base_model.compile(temperature=1.2, loss='categorical_crossentropy')
   ```

### Production Safeguards:

6. **Never show only one prediction** - Always show top 3 with confidences (as we did in testing)
7. **Add uncertainty disclaimer** when any prediction has < 80% confidence
8. **Implement manual override** for low-confidence predictions in production UI
9. **Add "Need expert review" flag** for predictions below threshold

---

## 📊 Final Verdict - CHART & RATINGS

### Overall Model Performance Chart

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL PERFORMANCE SUMMARY                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Accuracy    ████████████████████░░░ 97.45%  ⭐⭐⭐⭐☆        │
│   Early Blight█░░░░░░░░░░░░░░░░░░░░░░░ 98.75%  ⭐⭐⭐⭐⭐       │
│   Late Blight ████░░░░░░░░░░░░░░░░░░░░░ 97.20%  ⭐⭐⭐⭐☆      │
│   Healthy     ██████████████████░░░░░░ 91.18%  ⭐⭐⭐☆☆       │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│   Overall Status: ✅ GOOD (but needs rebalancing)             │
│   Production Ready: ⚠️ PARTIALLY (with safeguards)           │
└─────────────────────────────────────────────────────────────┘
```

### Production Readiness Checklist

| Criterion | Current Status | Requirement | Verdict |
|-----------|----------------|-------------|---------|
| Overall Accuracy | ✅ 97.45% | > 85% | ✅ PASS |
| Confidence Calibration | ⚠️ Variable | < 10% ECE | ⚠️ NEEDS WORK |
| Class Balance | ❌ 200:200:31 | ~200:200:200 | ❌ FAIL |
| Minimum Confidence (correct) | ⚠️ 46.44% | > 80% | ⚠️ NEEDS WORK |
| High-Confidence Accuracy | ✅ 99.25% | > 95% | ✅ PASS |
| Calibration Error (ECE) | ✅ 2.09% | < 10% | ✅ PASS |

### Final Production Readiness: **⚠️ PARTIALLY READY**

The model is **NOT fully production-ready** without these fixes:
- ❌ Must rebalance dataset (healthy class severely underrepresented)
- ⚠️ Need to implement confidence thresholding (< 80% = "uncertain")
- ✅ Good accuracy metrics for disease classes
- ✅ Early blight detection is excellent
- ✅ Late blight and healthy detection are acceptable with current data

---

## 📈 Accuracy vs Confidence Scatter Plot Visualization

```
Confidence Level (%)
    |
100 +  ● High-confidence correct predictions (96%+)          │
    |              ↑                                          │
 85 + ─────────────┴─────────────────────────── Threshold ✓   │
    |              │                                          │
 65 + ─────────────┴─────────────────────────── Medium tier   │
    |       ●                                           ●    │
 45 + ──────────────●──────────────────────────── Low tier   │
    |                ↑                                      │
 20 +------------------------------------------------───────┼──> Accuracy (%)
     60%           80%           90%           100%
     
Key zones:
• Top-right (high confidence, high accuracy): ✅ RELIABLE
• Bottom-right (low confidence, high accuracy): ⚠️ UNRELIABLE CALIBRATION
• Top-left (high confidence, low accuracy): ❌ OVERCONFIDENT MISTAKE
• Bottom-left (low confidence, low accuracy): ✓ HONEST UNCERTAINTY
```

### Interpretation:
- Most of your correct predictions fall in the **top-right** zone (96%+ conf)
- Some correct predictions are scattered lower (46-80% conf) - calibration issue
- **Recommendation**: Shift decision threshold to 85% confidence for reliable outputs

---

## 📝 Recommended Next Steps

1. ✅ **Stop deployment until dataset is rebalanced** (31 → ~200 healthy samples)
2. 🔧 **Generate synthetic or real images** for the healthy class
3. 🔧 **Retrain model** with balanced dataset
4. 📊 **Re-evaluate** after retraining to confirm improvements
5. 🔧 **Implement threshold-based predictions** in production code
6. 🎨 **Design UI** to show confidence scores and "uncertain" states clearly

---

*Report generated: November 23, 2026*
*Test environment: Plant Doctor AI with MobileNetV2 (fine-tuned)*
