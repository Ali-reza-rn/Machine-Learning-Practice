# MNIST Digit Classification with SVM

## Project Description
This project develops a handwritten digit recognition system (0-9)<br>
using the Support Vector Machine (SVM) algorithm and the MNIST dataset.<br>
### About MNIST Dataset:<br>
The MNIST database (Modified National Institute of Standards and Technology database)<br>
is a large database of handwritten digits that is<br>
commonly used for training various image processing systems

## Project Features
- Use of the MNIST dataset for digit recognition (0-9)
- Implementation of the SVM algorithm using scikit-learn
- Analysis of model performance with various metrics
- Display of the ROC curve for model evaluation
- Hyperparameter tuning for performance optimization
- Comparison of two different SVM models


## System Requirements
```
scikit-learn
matplotlib
numpy
```

## Installation
```bash
pip install scikit-learn matplotlib numpy
```

## Project Structure
```
└── mnist_dataset
    ├── mnist_SVM.ipynb
    ├── readme.md
    └── requirements.txt

```

## Implementation Steps

### 1. Data Loading
- Use load_digits() from scikit-learn
- Convert 8x8 images to 1D vectors

### 3. First Model Training
```python
model = SVC(probability=True)
model.fit(x_train, y_train)
```
### 4. First Model Evaluation
- Accuracy: 98.5%
- Classification Report
- Confusion Matrix
- ROC Curve

### 5. Hyperparameter Tuning
- Use validation curve
- Test parameter C in range [0.0001, 10000]
- Select best C=2

### 6. Optimized Model Training
```python
new_model = SVC(C=2, class_weight='balanced')
new_model.fit(x_train, y_train)
```
### 7. Results Comparison
- First model: 98.5% accuracy
- Second model: 98.9% accuracy
## Results

### First Model Performance
- Overall accuracy: 98.5%
- Best performance: Digit 6 (100% precision & recall)
- Worst performance: Digits 8 and 9
### Optimized Model Performance
- Overall accuracy: 98.9%
- Improved performance: Across all classes
## ROC Curve Analysis
```text
ROC Curve is a graphical plot used to evaluate binary classification models.<br>
It shows the trade-off between True Positive Rate (sensitivity)
and False Positive Rate (1-specificity) at various thresholds.
The curve plots TPR vs FPR, where a perfect classifier hugs the top-left corner (TPR=1, FPR=0).
AUC provides a single metric to compare models: AUC=1 is perfect, AUC=0.5 is random chance.
This MNIST result shows typical performance with all digit classes achieving high AUC scores.
```
## Validation Curve Analysis
The validation curve shows optimal C parameter should be in range 0.01 to 1 based on general SVM theory.<br>
However, our specific MNIST dataset results show different behavior.
Experimental Results

### Default SVM (C=1.0): 98.5% accuracy
### Tuned SVM (C=2.0): 98.8% accuracy

### Why C=2 Works Better
Our MNIST dataset has specific characteristics that allow higher C values:<br>
Large dataset (1797 samples) with high-dimensional features (784 pixels)<br>
Applied data augmentation and class balancing
Image data typically requires less regularization than text data
## Conclusion
This project successfully demonstrates the effectiveness of Support<br>
Vector Machine (SVM) for handwritten digit recognition on the MNIST dataset.<br>
Through systematic experimentation and hyperparameter tuning.