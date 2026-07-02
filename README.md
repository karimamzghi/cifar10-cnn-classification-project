# cifar10-cNN-classification-project

## 📌 Project Overview

This project explores image classification using **Convolutional Neural Networks (CNNs)** on the **CIFAR-10** dataset.

The objective is to design, train, evaluate, compare, and deploy multiple CNN architectures while following machine learning engineering best practices.

This project is part of the AI Engineering Bootcamp and is being developed incrementally as a production-style machine learning project.

---

## 🎯 Objectives

- Build a baseline CNN from scratch
- Experiment with multiple CNN architectures
- Compare different hyperparameters and training strategies
- Evaluate models using multiple performance metrics
- Track every experiment
- Apply augmentation and transfer
- Deploy the best performing model

---

## 📂 Project Structure

```
cifar10-cnn-classification-project/

├── deployment/
│
├── models/
│
├── notebooks/
│   └── baseline_cnn.ipynb
│
├── results/
│   ├── confusion_matrices/
│   └── model_tracking.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│
├── README.md
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
Input (32×32×3)

↓

Conv2D (32)

↓

MaxPooling

↓

Conv2D (64)

↓

MaxPooling

↓

Flatten

↓

Dense (64)

↓

Softmax (10)
```

Optimizer:

- SGD

Loss Function:

- Categorical Crossentropy

---

# Evaluation

Each model will be evaluated using:

- Accuracy
- Confusion Matrix

Current confusion matrices are stored under:

```
results/confusion_matrices/
```

---

# Planned Experiments

- [x] Baseline CNN
- [ ] Deeper CNN               
- [ ] Dropout (0.3–0.5)        
- [ ] Adam instead of SGD Optimizer effect
- [ ] Batch Normalization
- [ ] Data augmentation (flip, rotation, zoom)
- [ ] Early stopping on val_loss
- [ ] Best-so-far combo (e.g. deep + BN + dropout + aug)
- [ ] MobileNetV2 frozen (transfer learning)
- [ ] MobileNetV2 fine-tuned (unfreeze top, lr=1e-5)

---

# Deployment

Planned deployment:

- Vercel

---

