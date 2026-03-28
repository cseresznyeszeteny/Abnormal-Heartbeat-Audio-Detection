# Abnormal Heartbeat Detection: CNN vs. Gradient Boosting

**Course:** UGST4088 — Introduction to Machine Learning and Data Mining 2025/26 Fall
**Institution:** Central European University, Vienna
**Author:** Zeteny Cseresznyes

## Overview

Compares two approaches to binary classification of heartbeat audio recordings (Normal vs. Abnormal):

1. **CNN on Mel-spectrograms** — end-to-end deep learning on audio images
2. **XGBoost on engineered features** — classical ML on handcrafted audio features (MFCCs, ZCR, RMS, spectral properties)

The central question: does performance differ due to the audio representation (Mel-spectrogram vs. features) or the learning algorithm (CNN vs. XGBoost)?

## Dataset

- **Training:** 3,163 recordings from 1,568 patients (Northeast Brazil, 2014–2015), collected via digital phonocardiography at four valve positions (Aortic, Mitral, Pulmonary, Tricuspid)
- **Test Set A:** iStethoscope mobile app recordings (noisy, real-world)
- **Test Set B:** Digiscope clinical recordings (high-quality, Pascal Challenge 2011)

## Contents

- `UGST4088_heartbeatsCNN_Cseresznyes.ipynb` — Full pipeline notebook
- `UGST4088_report_Cseresznyes.pdf` — Written report

## Pipeline

### Preprocessing
- Resample to 2000 Hz, pad/crop to 5 seconds
- **CNN:** Mel-spectrogram (64 bands, 25–600 Hz, normalized to dB scale)
- **XGBoost:** ZCR, RMS, tempogram, spectral centroid/bandwidth, 12 MFCCs (mean + std)

### Data Augmentation (CNN training only)
- Random 5s cropping, Gaussian noise, SpecAugment-style frequency masking

### Models

**CNN (TensorFlow/Keras)**
```
Conv2D(32) → BN → MaxPool
Conv2D(64) → BN → MaxPool
Flatten → Dense(128) → Dropout(0.5) → Sigmoid
```
- Adam (lr=1e-4), binary cross-entropy, early stopping on val recall

**XGBoost**
- Hyperparameters tuned with Optuna (30 trials, 5-fold CV, recall objective)
- Best config: 350 estimators, max_depth=8, lr=0.015

## Results

**Held-out evaluation set:**

| Model | Accuracy | Abnormal Recall |
|-------|----------|-----------------|
| CNN | 52% | 0.92 |
| XGBoost | **97%** | 0.97 |
| Ensemble | **97%** | 0.97 |

**Generalization to test sets:**

| Model | iStethoscope (mobile) | Digiscope (clinical) |
|-------|-----------------------|----------------------|
| CNN | 51% | 55% |
| XGBoost | 41% | 53% |

All models degrade significantly on out-of-distribution recordings, highlighting cross-device generalization challenges.

## Key Findings

- XGBoost with engineered features outperformed the CNN on the evaluation set, despite CNN being the primary approach
- CNN was sensitive to class imbalance — high abnormal recall (0.92) but very low normal recall (0.15)
- Both models struggled to generalize across recording devices
- Recall was the primary optimization metric to minimize false negatives in medical screening

## Requirements

```bash
pip install tensorflow xgboost optuna librosa scikit-learn pandas numpy matplotlib seaborn
```

## Author

- LinkedIn: [zeteny-cseresznyes](https://www.linkedin.com/in/zeteny-cseresznyes-236ba8329/)
- GitHub: [cseresznyeszeteny](https://github.com/cseresznyeszeteny)
