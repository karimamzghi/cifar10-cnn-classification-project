# cifar10-cNN-classification-project

## Project Overview

This project explores image classification using **Convolutional Neural Networks (CNNs)** on the **CIFAR-10** dataset.

The objective is to design, train, evaluate, compare, and deploy multiple CNN architectures while following machine learning engineering best practices.

This project is part of the AI Engineering Bootcamp and is being developed incrementally as a production-style machine learning project.

---

## Objectives

- Build a baseline CNN from scratch
- Experiment with multiple CNN architectures
- Compare different hyperparameters and training strategies
- Evaluate models using multiple performance metrics
- Track every experiment
- Apply augmentation and transfer
- Deploy the best performing model

---

## Project Structure

```
## Project Structure

``` cifar10-cnn-classification-project
cifar10-cnn-classification-project/
│
├── deployment/
│   └── app.py                        # Deployment application
│
├── models/                           # Saved trained models (.keras)
│
├── notebooks/
│   ├── 01_baseline_cnn.ipynb
│   ├── 02_deeper_cnn.ipynb
│   ├── 03_dropout_cnn.ipynb
│   ├── 04_adam_optimizer.ipynb
│   ├── 05_batch_normalization.ipynb
│   ├── 06_data_augmentation.ipynb
│   ├── 07_early_stopping.ipynb
│   ├── 08_larger_cnn.ipynb
│   ├── 09_mobilenet_frozen.ipynb
│   └── 10_mobilenet_finetuned.ipynb
│
├── results/
│   ├── confusion_matrices/           # Saved confusion matrices
│   ├── plots/                        # Training curves
│   └── model_tracking.csv            # Experiment tracking
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Project configuration
│   ├── data_loader.py                # Load cached or prepare dataset
│   ├── dataset_processing.py         # Dataset preprocessing & caching
│   ├── experiment_tracker.py         # Save experiment metadata
│   │
│   └── training/
│       ├── __init__.py
│       ├── model_builder.py          # CNN architectures
│       ├── trainer.py                # Compile, train & save models
│       ├── evaluator.py              # Accuracy, precision, recall & confusion matrix
│       └── predictor.py              # Predict single images
│
├── README.md
└── .gitignore
```

---

# Dataset

Dataset: **CIFAR-10**

- 60,000 RGB images
- 10 classes
- Image size: 32 × 32 pixels

Classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

# Data Preparation

Current preprocessing pipeline:

- Load CIFAR-10 dataset
- Train / Validation / Test split
- Normalize image pixels to [0,1]
- One-hot encode labels

Current split:
Concatenated all the data and then did the split to get even data sets.

| Dataset | Images |
|---------|-------:|
| Train | 40,000 |
| Validation | 10,000 |
| Test | 10,000 |

---

# Model Experiments

This project tracks every experiment performed.

Each experiment records:

- Model architecture
- Optimizer
- Learning rate
- Batch size
- Number of epochs
- Training accuracy
- Validation accuracy
- Test accuracy
- Training loss
- Validation loss
- Test loss
- Saved model path
- Confusion matrix path
- Notes

Results are automatically stored in:

```
results/model_tracking.csv
```

---

# Model Architectures

## Model 1 — Baseline CNN

**Purpose:** Establish a baseline performance for comparison with more advanced CNN architectures.

```text
Input (32×32×3)
│
├── Conv2D (32)
├── MaxPooling2D
├── Conv2D (32)
├── MaxPooling2D
├── Flatten
├── Dense (64)
└── Softmax (10)
```

**Optimizer:** SGD

**Loss Function:** Categorical Crossentropy

---

## Model 2 — Deep CNN

**Purpose:** Evaluate whether increasing the network depth improves feature extraction and classification performance.

```text
Input (32×32×3)
│
├── Conv2D (32)
├── Conv2D (32)
├── MaxPooling2D
│
├── Conv2D (64)
├── Conv2D (64)
├── MaxPooling2D
│
├── Flatten
├── Dense (64)
└── Softmax (10)
```

**Optimizer:** SGD

**Loss Function:** Categorical Crossentropy

---

## Model 3 — Deep CNN with Dropout

**Purpose:** Reduce overfitting by introducing Dropout before the output layer.

```text
Input (32×32×3)
│
├── Conv2D (32)
├── Conv2D (32)
├── MaxPooling2D
│
├── Conv2D (64)
├── Conv2D (64)
├── MaxPooling2D
│
├── Flatten
├── Dense (64)
├── Dropout (0.5)
└── Softmax (10)
```

**Optimizer:** SGD

**Loss Function:** Categorical Crossentropy

---

## Model 4 — Deep CNN with Adam Optimizer

**Purpose:** Compare the Adam optimizer against SGD while keeping the architecture unchanged.

**Architecture**

Same as Model 3.

**Optimizer:** Adam

**Loss Function:** Categorical Crossentropy

---

## Model 5 — Deep CNN with Batch Normalization

**Purpose:** Improve convergence speed and training stability using Batch Normalization.

```text
Input (32×32×3)
│
├── Conv2D (32)
├── BatchNormalization
├── Conv2D (32)
├── BatchNormalization
├── MaxPooling2D
│
├── Conv2D (64)
├── BatchNormalization
├── Conv2D (64)
├── BatchNormalization
├── MaxPooling2D
│
├── Flatten
├── Dense (64)
├── BatchNormalization
├── Dropout (0.5)
└── Softmax (10)
```

**Optimizer:** Adam

**Loss Function:** Categorical Crossentropy

---

## Model 6 — Augmented CNN

**Purpose:** Improve model generalization through image augmentation.

**Architecture**

Same as Model 5 with an image augmentation layer.

**Data Augmentation**

- Random Flip
- Random Rotation
- Random Zoom

**Optimizer:** Adam

---

## Model 7 — Augmented CNN with Early Stopping

**Purpose:** Prevent overfitting by stopping training once validation performance stops improving.

**Architecture**

Same as Model 6.

**Training Strategy**

- Early Stopping
- Restore Best Weights

**Optimizer:** Adam

---

## Model 8 — Large Augmented CNN

**Purpose:** Evaluate whether increasing the network capacity improves classification performance.

```text
Input (32×32×3)
│
├── Data Augmentation
│
├── Conv2D (32)
├── Conv2D (32)
├── MaxPooling2D
│
├── Conv2D (64)
├── Conv2D (64)
├── MaxPooling2D
│
├── Conv2D (128)
├── Conv2D (128)
├── MaxPooling2D
│
├── Flatten
├── Dense (128)
├── Dropout (0.5)
└── Softmax (10)
```

**Optimizer:** Adam

---

## Model 9 — MobileNetV2 Transfer Learning

**Purpose:** Leverage pretrained ImageNet features for image classification.

```text
Input (32×32×3)
│
├── MobileNetV2 (Frozen)
├── GlobalAveragePooling2D
├── Dense (128)
├── Dropout (0.5)
└── Softmax (10)
```

**Optimizer:** Adam

**Transfer Learning**

- ImageNet pretrained weights
- Base model frozen

---

## Model 10 — MobileNetV2 Fine-Tuning

**Purpose:** Fine-tune the upper MobileNetV2 layers to adapt pretrained features to the CIFAR-10 dataset.

```text
Input (32×32×3)
│
├── MobileNetV2 (Fine-tuned)
├── GlobalAveragePooling2D
├── Dense (128)
├── Dropout (0.5)
└── Softmax (10)
```

**Optimizer:** Adam

**Learning Rate:** 1e-5

**Transfer Learning**
- Pretrained MobileNetV2 weights loaded from ImageNet.
- Only the final MobileNetV2 layers were unfrozen for training.
- A small learning rate (1e-5) was used to fine-tune the pretrained features.

# Evaluation
Each experiment generates:

- Saved trained model (.keras)
- Confusion Matrix
- Precision
- Recall
- Accuracy
- Experiment metadata

# Results
Results are automatically stored under:

```
models/
results/model_tracking.csv
results/confusion_matrices/
```

---

# Planned Experiments

- [x] Baseline CNN
- [X] Deeper CNN               
- [X] Dropout (0.3–0.5)        
- [X] Adam instead of SGD Optimizer effect
- [X] Batch Normalization
- [X] Data augmentation (flip, rotation, zoom)
- [X] Early stopping on val_loss
- [X] Best-so-far combo (e.g. deep + BN + dropout + aug)
- [X] MobileNetV2 frozen (transfer learning)
- [X] MobileNetV2 fine-tuned (unfreeze top, lr=1e-5)

---

# Deployment

Planned deployment:

- TBD

---
# Author

**Karima Mzoughi**

- GitHub: https://github.com/karimamzghi
- LinkedIn: https://www.linkedin.com/in/karimamzghi/
