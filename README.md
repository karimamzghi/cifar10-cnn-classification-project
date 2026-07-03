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

# Baseline Model

Current baseline architecture:

```
Input (32×32×3) -> Conv2D (32) -> MaxPooling -> Conv2D (32) -> MaxPooling -> Flatten -> Dense (64) -> Softmax (10)

```

Optimizer:

- SGD

Loss Function:

- Categorical Crossentropy

---

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
