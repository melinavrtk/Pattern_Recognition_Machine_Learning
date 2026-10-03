# Pattern Recognition & Machine Learning Pipeline

This repository contains a complete, Object-Oriented Python pipeline for pattern recognition and image classification. It transitions from raw image data to advanced classification and dimensionality reduction, implementing core machine learning algorithms from scratch using `numpy` and `scipy`.

## 📂 Pipeline Architecture

### 1. Feature Extraction (`01_texture_feature_extraction.py`)
* Extracts **1st-order statistical features** (Mean, Standard Deviation, Skewness, Kurtosis).
* Extracts **2nd-order texture features** using Gray-Level Co-occurrence Matrices (GLCM). Computes Contrast, Correlation, Energy, and Homogeneity averaged across 4 angles ($0^\circ, 45^\circ, 90^\circ, 135^\circ$) for rotation invariance.

### 2. Data Preprocessing (`02_data_preprocessing.py`)
* Custom mathematical implementations for data normalization without relying on high-level ML wrappers.
* Features **Min-Max Scaling** and **Z-score Standardization**, dynamically saving parameters (min, max, mean, std) from training data to apply accurately to unknown test patterns.

### 3. Classification Models (`03_machine_learning_classifiers.py`)
* **Minimum Distance (MD) Classifier**: Evaluates unknown patterns based on Euclidean distance to class centroids.
* **Gaussian Naive Bayes Classifier**: Calculates prior probabilities and covariance matrices for probabilistic classification.
* **k-Nearest Neighbors (k-NN)**: A vectorized, distance-based voting classifier.

### 4. Dimensionality Reduction (`04_dimensionality_reduction_lda.py`)
* Implements **Linear Discriminant Analysis (LDA)** to project multi-dimensional data into a 1D feature space.
* Maximizes class separability based on the **Fisher Criterion** and includes comprehensive `matplotlib` visualizations of the original vs. transformed feature spaces.

## 🛠️ Skills & Technologies Highlighted
* **Mathematics & Linear Algebra**: Matrix operations, covariance, eigenvectors, and statistical modeling.
* **Computer Vision**: ROI extraction and spatial texture analysis (`skimage.feature`, `cv2`).
* **Machine Learning Foundation**: Building foundational classification and dimensionality reduction algorithms from the ground up.
  
