# =========================================================
# Pattern Recognition & Machine Learning Pipeline
# Step 9: Multi-Classifier Evaluation & Feature Pairing (scikit-learn)
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import combinations
from sklearn.datasets import load_breast_cancer
import warnings

warnings.simplefilter('ignore')

# Classifiers
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

class SklearnFeatureEvaluator:
    """
    Evaluates multiple scikit-learn models across different combinations 
    of features to find the optimal predictive subset.
    """

    def __init__(self):
        self.classifiers = {
            "MDC": NearestCentroid(),
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "Bayesian": GaussianNB(),
            "LDA": LinearDiscriminantAnalysis(),
            "LogReg": LogisticRegression(class_weight='balanced', max_iter=500)
        }

    def load_data(self):
        """ 
        Loads dataset. Using Breast Cancer dataset as a standard medical benchmark.
        Replaces the custom CSV loading for out-of-the-box execution.
        """
        data = load_breast_cancer()
        X = data.data
        y = data.target
        fNames = np.array(data.feature_names)
        return X, y, fNames

    def plot_2d_scatter(self, X, y, title, fNames):
        """ Visualizes the best performing 2D feature combination. """
        plt.figure(figsize=(8, 6))
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

    def evaluate_feature_pairs(self, X, y, fNames, classifier_name="MDC"):
        """
        Tests all possible 2-feature combinations for a chosen classifier 
        to find the optimal predictive pair.
        """
        if classifier_name not in self.classifiers:
            raise ValueError(f"Classifier {classifier_name} not supported.")
            
        clf = self.classifiers[classifier_name]
        
        # Determine the maximum allowed features to test based on class sizes (Rule of thumb: N_min / 3)
        class_counts = [np.sum(y == c) for c in np.unique(y)]
        min_class = np.min(class_counts)
        max_allowed_features = min(int(np.round(min_class / 3)), 8)
        
        print(f"--- Evaluation for {classifier_name} ---")
        print(f"Smallest class size: {min_class}. Max safe feature dimension: {max_allowed_features}\n")

        # Generate combinations of 2 features (to allow 2D plotting)
        # Using only the first 8 features to limit the output in this example
        n_feats = min(len(fNames), 8)
        feat_combinations = list(combinations(range(n_feats), 2))
        
        max_acc = 0
        best_feats = None

        # Split data once to ensure consistent evaluation across feature sets
        X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        for feats in feat_combinations:
            feats_arr = list(feats)
            
            # Subset the features
            X_train = X_train_full[:, feats_arr]
            X_test = X_test_full[:, feats_arr]
            
            # Train and Predict
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            
            # Evaluate Accuracy
            acc = accuracy_score(y_test, predictions)
            
            if acc > max_acc:
                max_acc = acc
                best_feats = feats_arr

        print("=============================================")
        print(f"BEST RESULT FOR {classifier_name.upper()}")
        print(f"Optimal Feature Pair: {fNames[best_feats]}")
        print(f"Peak Accuracy: {max_acc * 100:.2f}%")
        print("=============================================\n")

        # Plot the winning combination
        title = f"{classifier_name} Scatter Diagram - Acc: {max_acc*100:.2f}%"
        self.plot_2d_scatter(X[:, best_feats], y, title, fNames[best_feats])

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    evaluator = SklearnFeatureEvaluator()
    
    # Load data
    X, y, fNames = evaluator.load_data()
    
    # Run evaluation for Bayesian and MDC classifiers
    evaluator.evaluate_feature_pairs(X, y, fNames, classifier_name="Bayesian")
    evaluator.evaluate_feature_pairs(X, y, fNames, classifier_name="MDC")
