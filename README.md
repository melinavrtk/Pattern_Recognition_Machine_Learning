# Pattern Recognition & Machine Learning Pipeline

This repository contains a comprehensive, Object-Oriented Python pipeline for pattern recognition, predictive modeling, and deep learning. It transitions from raw data processing and "from-scratch" algorithm implementations to advanced scikit-learn benchmarking suites and TensorFlow/Keras neural networks.

## 📂 Pipeline Architecture & Modules

### Module 1: Feature Extraction & Data Preprocessing
* **`01_texture_feature_extraction.py`**: Extracts 1st-order statistical features and 2nd-order texture features using Gray-Level Co-occurrence Matrices (GLCM).
* **`02_data_preprocessing.py`**: Custom mathematical implementations for Min-Max Scaling and Z-score Standardization without relying on high-level ML wrappers.

### Module 2: Foundational Classifiers & Dimensionality Reduction
* **`03_machine_learning_classifiers.py`**: Implementation of Minimum Distance (MD), Gaussian Naive Bayes, and k-NN classifiers from scratch.
* **`04_dimensionality_reduction_lda.py`**: Implements Linear Discriminant Analysis (LDA) projecting multi-dimensional data into a 1D space using the Fisher Criterion.
* **`05_mdc_visualizer.py` & `06_multiclass_mdc_visualizer.py`**: Geometric visualizers for Minimum Distance Classifiers handling binary and multi-class problems.
* **`08_knn_vs_mdc_comparison.py`**: Direct algorithmic comparison and visualization of distance-based decision boundaries.

### Module 3: Advanced Feature Engineering & Selection
* **`07_feature_selection_mdc.py`**: Evaluates combinatorial feature spaces to extract optimal predictive feature pairs.
* **`09_sklearn_classifiers_feature_selection.py`**: Dynamic feature pairing across multiple scikit-learn estimators.
* **`11_advanced_feature_engineering_and_evaluation.py`**: An advanced suite implementing Principal Component Analysis (PCA), Recursive Feature Elimination (RFE), and Statistical Hypothesis Testing (t-test) for dimensionality reduction and feature ranking.

### Module 4: Comprehensive Validation & Benchmarking
* **`10_comprehensive_ml_benchmark.py`**: An extensive benchmarking tool evaluating 10 different machine learning classifiers (SVM, Random Forests, Multi-Layer Perceptrons, etc.) with automated ROC curve plotting.
* Utilizes robust validation techniques including **K-Fold Cross-Validation, Leave-One-Out (LOO), and Bootstrap resampling**.

### Module 5: Unsupervised Learning & Clustering
* **`12_kmeans_clustering_from_scratch.py`**: A pure `numpy` implementation of the k-Means clustering algorithm, visualizing centroid convergence step-by-step.
* **`13_advanced_clustering_suite.py`**: Evaluates model stability across multiple epochs using 8 advanced clustering algorithms (Gaussian Mixture Models, Spectral Clustering, DBSCAN, Affinity Propagation, Birch, MeanShift).

### Module 6: Deep Learning
* **`14_neural_network_mlp_tensorflow.py`**: Implements a Multi-Layer Perceptron (MLP) Neural Network using **TensorFlow/Keras** to classify complex, synthetically generated feature spaces.

## 🛠️ Skills & Technologies Highlighted
* **Languages & Frameworks**: Python, `numpy`, `scipy`, `scikit-learn`, `TensorFlow`, `Keras`, `pandas`, `matplotlib`.
* **Machine Learning Foundation**: Building core algorithms from scratch (Distance Metrics, Covariance Matrices, Eigenvectors).
* **Deep Learning**: Sequential modeling, Dense layers, ReLU/Softmax activations, categorical cross-entropy loss.
* **Model Validation**: Algorithmic stability, robust cross-validation, and ROC/AUC analysis.
