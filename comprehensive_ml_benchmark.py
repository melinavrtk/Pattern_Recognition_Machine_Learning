# =========================================================
# Pattern Recognition & Machine Learning Pipeline
# Step 10: Comprehensive Machine Learning Benchmarking Suite
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, RocCurveDisplay, roc_auc_score
from sklearn.datasets import load_breast_cancer

# Classifiers
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

class MLBenchmarker:
    """
    A comprehensive suite to evaluate multiple Machine Learning classifiers 
    across different feature combinations to find the optimal model.
    """

    def __init__(self):
        # Dictionary mapping classifier names to their scikit-learn objects
        self.classifiers = {
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "LDA": LinearDiscriminantAnalysis(),
            "LogReg": LogisticRegression(class_weight='balanced', max_iter=1000),
            "Bayesian": GaussianNB(),
            "MDC": NearestCentroid(),
            "Perceptron": Perceptron(tol=1e-3, random_state=0),
            "MLP": MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=1000, hidden_layer_sizes=(10, 2), random_state=1),
            "SVM": NuSVC(kernel='rbf', degree=3, class_weight='balanced'),
            "RandomForest": RandomForestClassifier(n_estimators=10, random_state=42),
            "CART": DecisionTreeClassifier(class_weight='balanced', random_state=42)
        }

    def load_medical_data(self, use_dummy=True):
        """ Loads dataset. Uses a built-in medical dataset for quick demonstration. """
        if use_dummy:
            data = load_breast_cancer()
            X = data.data
            y = data.target
            fNames = data.feature_names
            return X, y, fNames
        else:
            # Placeholder for your local CSV loading logic
            pass

    def plot_2d_scatter(self, X, y, title, fNames):
        """ 2D Scatter visualization for the best features. """
        plt.figure(figsize=(7, 7))
        n_classes = len(np.unique(y))
        
        for i_class in range(n_classes):
            idx = np.where(y == i_class)
            plt.scatter(X[idx, 0], X[idx, 1], label=f'Class {i_class}')
            
        plt.grid(True)
        plt.title(title)
        plt.xlabel(fNames[0])
        plt.ylabel(fNames[1])
        plt.legend(loc='best')
        plt.show()

    def plot_roc_curve(self, clf, X_test, y_test, title):
        """ Modern ROC Curve plotting using scikit-learn API. """
        plt.figure(figsize=(7, 7))
        ax = plt.gca()
        RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax)
        plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
        plt.grid(True)
        plt.title(title)
        plt.legend(loc='best')
        plt.show()

    def run_benchmark(self, X, y, feature_names, classifier_name="RandomForest", n_features_to_test=2):
        """
        Tests combinations of features to find the highest accuracy for a chosen model.
        """
        if classifier_name not in self.classifiers:
            raise ValueError(f"Classifier {classifier_name} not found.")

        clf = self.classifiers[classifier_name]
        
        # Create combinations of features (testing combinations of 'n_features_to_test' length)
        feature_indices = list(range(X.shape[1]))
        combs = list(combinations(feature_indices, n_features_to_test))
        
        # Limit to first 50 combinations for speed in this example
        combs = combs[:50] 
        
        best_accuracy = 0
        best_features = None
        best_model = None
        best_X_test, best_y_test = None, None

        print(f"--- Running {classifier_name} Benchmark ---")
        print(f"Testing {len(combs)} different {n_features_to_test}-feature combinations...\n")

        for feats in combs:
            X_subset = X[:, feats]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42)
            
            # Train and predict
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            
            # Evaluate
            acc = accuracy_score(y_test, predictions)
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_features = feats
                best_model = clf
                best_X_test, best_y_test = X_test, y_test

        print("=============================================")
        print(f"BEST RESULTS FOR: {classifier_name}")
        print(f"Optimal Features Used: {[feature_names[i] for i in best_features]}")
        print(f"Peak Accuracy: {best_accuracy * 100:.2f}%")
        print("=============================================\n")

        # Visualizations based on feature dimension
        if len(best_features) == 2:
            title = f"2D Scatter - {classifier_name} (Acc: {best_accuracy*100:.2f}%)"
            self.plot_2d_scatter(X[:, best_features], y, title, [feature_names[i] for i in best_features])
        
        # Plot ROC if classifier supports probability predictions
        if hasattr(best_model, "predict_proba"):
            roc_title = f"ROC Curve - {classifier_name}"
            self.plot_roc_curve(best_model, best_X_test, best_y_test, roc_title)

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    benchmark = MLBenchmarker()
    
    # Load dataset
    X_data, y_data, f_names = benchmark.load_medical_data(use_dummy=True)
    
    # Run Benchmark for Random Forest using combinations of 2 features (for 2D plotting)
    benchmark.run_benchmark(X_data, y_data, f_names, classifier_name="RandomForest", n_features_to_test=2)
    
    # Run Benchmark for k-NN
    benchmark.run_benchmark(X_data, y_data, f_names, classifier_name="KNN", n_features_to_test=2)
