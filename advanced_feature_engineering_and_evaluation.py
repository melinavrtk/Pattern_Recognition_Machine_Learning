# =========================================================
# Pattern Recognition & Machine Learning Pipeline
# Step 11: Advanced Feature Engineering & Model Evaluation
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, LeaveOneOut, ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_breast_cancer
import warnings

warnings.simplefilter('ignore')

class AdvancedMLEvaluator:
    """
    A robust suite for advanced Feature Engineering (PCA, RFE, Statistical Ranking)
    and robust Model Evaluation (K-Fold, LOO, Bootstrap).
    """

    def __init__(self):
        # Using Random Forest as the default robust classifier for evaluation
        self.classifier = RandomForestClassifier(n_estimators=20, random_state=42)

    @staticmethod
    def load_data():
        """ Loads standard dataset (Breast Cancer) for demonstration. """
        data = load_breast_cancer()
        return data.data, data.target, np.array(data.feature_names)

    # ---------------------------------------------------------
    # FEATURE REDUCTION & SELECTION METHODS
    # ---------------------------------------------------------
    def rank_by_ttest(self, X, y, f_names, p_value_threshold=0.05):
        """ Retains features that show statistically significant differences (t-test) between classes. """
        print(f"--- T-Test Feature Ranking (p <= {p_value_threshold}) ---")
        class_0 = X[y == 0]
        class_1 = X[y == 1]
        
        p_values = []
        for i in range(X.shape[1]):
            _, p = stats.ttest_ind(class_0[:, i], class_1[:, i], equal_var=False)
            p_values.append(p)
            
        p_values = np.array(p_values)
        valid_indices = np.where(p_values <= p_value_threshold)[0]
        
        # Sort by most significant (lowest p-value)
        sorted_indices = valid_indices[np.argsort(p_values[valid_indices])]
        
        print(f"Selected {len(sorted_indices)} features: {f_names[sorted_indices]}\n")
        return X[:, sorted_indices], f_names[sorted_indices]

    def reduce_by_pca(self, X, desired_variance=0.95):
        """ Applies Principal Component Analysis (PCA) to retain desired variance. """
        print(f"--- PCA Dimensionality Reduction (Target Variance: {desired_variance*100}%) ---")
        pca = PCA()
        pca.fit(X)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= desired_variance) + 1
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        pca_names = np.array([f"PC{i+1}" for i in range(n_components)])
        print(f"Reduced to {n_components} Principal Components.\n")
        return X_pca, pca_names

    def select_by_rfe(self, X, y, f_names, n_features_to_select=5):
        """ Uses Recursive Feature Elimination (RFE) with a Random Forest estimator. """
        print(f"--- Recursive Feature Elimination (RFE) - Top {n_features_to_select} ---")
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfe = RFE(estimator, n_features_to_select=n_features_to_select)
        rfe.fit(X, y)
        
        selected_indices = np.where(rfe.support_)[0]
        print(f"Selected Features: {f_names[selected_indices]}\n")
        return X[:, selected_indices], f_names[selected_indices]

    # ---------------------------------------------------------
    # CROSS-VALIDATION EVALUATION METHODS
    # ---------------------------------------------------------
    def evaluate_model(self, X, y, method="kfold"):
        """ Evaluates the model using the specified cross-validation strategy. """
        if method == "kfold":
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            method_name = "5-Fold Cross-Validation"
        elif method == "loo":
            cv = LeaveOneOut()
            method_name = "Leave-One-Out (LOO)"
        elif method == "bootstrap":
            cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
            method_name = "Bootstrap (10 Iterations, 80/20 Split)"
        else:
            raise ValueError("Unknown evaluation method.")

        print(f"--- Running {method_name} ---")
        
        y_true_all, y_pred_all = [], []
        
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            self.classifier.fit(X_train, y_train)
            preds = self.classifier.predict(X_test)
            
            y_true_all.extend(y_test)
            y_pred_all.extend(preds)

        # Calculate final accuracy and Truth Table (Confusion Matrix)
        acc = accuracy_score(y_true_all, y_pred_all)
        cm = confusion_matrix(y_true_all, y_pred_all)
        
        print(f"Overall Accuracy: {acc * 100:.2f}%")
        print("Truth Table (Confusion Matrix):")
        print(cm)
        print("=============================================\n")
        return acc, cm

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    evaluator = AdvancedMLEvaluator()
    
    # 1. Load Data
    X_raw, y, feature_names = evaluator.load_data()
    
    # 2. Apply Feature Engineering (Choose one to test)
    # Example A: PCA Reduction
    X_pca, pca_names = evaluator.reduce_by_pca(X_raw, desired_variance=0.90)
    
    # Example B: RFE Selection
    X_rfe, rfe_names = evaluator.select_by_rfe(X_raw, y, feature_names, n_features_to_select=3)
    
    # 3. Evaluate the Engineered Data
    print("Evaluating PCA Data:")
    evaluator.evaluate_model(X_pca, y, method="kfold")
    
    print("Evaluating RFE Data:")
    evaluator.evaluate_model(X_rfe, y, method="bootstrap")
