# =========================================================
# Pattern Recognition & Machine Learning Pipeline
# Step 7: Feature Selection via Minimum Distance Classifier (MDC)
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from itertools import combinations
import warnings

# Ignore benign warnings for cleaner output
warnings.simplefilter('ignore')

class FeatureSelectorMDC:
    """
    Evaluates different combinations of features using a Minimum Distance 
    Classifier (Nearest Centroid) to identify the most discriminative feature pair.
    """

    @staticmethod
    def load_cervical_cancer_data():
        """
        Loads the Cervical Cancer (5-year survival) dataset.
        Features: Mean, Standard Deviation, Skewness, Kurtosis.
        """
        feats_names = ['Mean', 'Standard Deviation', 'Skewness', 'Kurtosis']
        
        # Class 1: Cervical cancer 5-years Non-survivors
        class1 = [
            [91.13, 22.355, 0.027635, 2.8102], [90.575, 29.5, 0.58353, 2.5318],
            [92.832, 28.751, 0.33067, 2.7629], [88.181, 28.642, 0.27494, 2.4152],
            [90.709, 30.939, 0.37855, 2.4851], [104.05, 35.792, 0.52814, 2.7435],
            [98.342, 36.193, 0.35172, 2.2441], [99.269, 26.342, 0.1492, 2.5312],
            [94.845, 31.323, 0.29206, 2.4237], [161.65, 17.314, 0.54258, 2.4952],
            [139.76, 19.887, 0.67346, 2.5673], [146.79, 20.749, 0.59332, 2.5536],
            [135.71, 21.629, 0.7011, 2.6016], [132.78, 19.188, 0.68376, 2.6227],
            [144.19, 19.955, 0.56923, 2.5648], [137.0, 22.193, 0.62745, 2.5139],
            [130.64, 20.973, 0.75887, 2.6656], [138.88, 20.66, 0.63738, 2.7213],
            [131.78, 22.323, 0.74514, 2.6985], [132.77, 22.599, 0.7028, 2.6367],
            [138.33, 21.06, 0.51725, 2.7121], [142.46, 26.016, 0.57662, 2.4987],
            [158.37, 22.355, 0.46325, 2.5201], [132.62, 22.586, 0.72381, 2.7383],
            [154.72, 21.404, 0.54958, 2.4043], [140.22, 26.547, 0.54553, 2.4363],
            [140.45, 25.769, 0.47033, 2.5039], [138.71, 25.652, 0.58671, 2.5203],
            [138.46, 25.098, 0.59268, 2.5743]
        ]
        
        # Class 2: Cervical cancer 5-years survivors
        class2 = [
            [131.35, 38.58, 0.23894, 2.9015], [164.11, 21.972, -0.0013328, 2.7228],
            [150.05, 22.108, 0.085606, 2.8163], [169.52, 22.394, 0.10823, 2.6467],
            [159.44, 23.22, -0.0020959, 2.585], [145.39, 16.517, 0.12471, 2.3972],
            [108.82, 24.038, 0.2161, 2.3497], [127.11, 22.186, 0.18473, 2.2985],
            [130.45, 17.907, 0.42612, 2.5721], [118.8, 25.907, 0.29538, 2.6881],
            [138.8, 25.896, 0.36752, 2.5331], [151.18, 18.881, 0.41188, 2.8039],
            [156.28, 24.799, 0.13559, 2.4917], [153.32, 29.878, 0.11923, 2.3113],
            [140.85, 27.791, 0.27059, 2.3632], [140.92, 29.667, 0.16173, 2.3855],
            [137.01, 29.869, 0.18743, 2.397], [147.71, 28.388, 0.054598, 2.4727],
            [113.36, 29.952, 0.39364, 2.5637], [123.92, 30.482, 0.23871, 2.4988],
            [136.87, 23.505, 0.22786, 2.4613], [136.36, 24.655, 0.19578, 2.4035],
            [130.26, 27.7, 0.17439, 2.505], [143.38, 26.278, 0.10052, 2.4845],
            [138.7, 26.622, 0.31597, 2.3737], [149.64, 24.757, 0.20055, 2.3103]
        ]
        
        X = np.concatenate((class1, class2), axis=0)
        y = np.concatenate((np.zeros(len(class1), dtype=int), np.ones(len(class2), dtype=int)), axis=0)
        feats_names = np.array(feats_names)
        
        return X, y, feats_names

    @staticmethod
    def plot_2d_scatter(X, y, title, feature_names):
        """ Visualizes the 2D feature space of the best performing combination. """
        plt.figure(figsize=(8, 6))
        n_classes = len(np.unique(y))
        colors = ['b', 'r']
        
        for i_class in range(n_classes):
            idx = np.where(y == i_class)
            plt.scatter(X[idx, 0], X[idx, 1], c=colors[i_class])
            
        plt.grid(True)
        plt.title(title)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.legend(['Non-Survivors (Class 1)', 'Survivors (Class 2)'], loc='best')
        plt.show()

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    selector = FeatureSelectorMDC()
    
    # 1. Load Data
    X, y, fNames = selector.load_cervical_cancer_data()
    
    # 2. Setup Classifier
    mdc_model = NearestCentroid()
    
    # Generate all pairs of features automatically (e.g., [0,1], [0,2], ... [2,3])
    feature_combinations = list(combinations(range(len(fNames)), 2))
    
    max_acc = 0
    best_feats = None

    print("--- Evaluating Feature Combinations using MDC ---\n")

    # 3. Iterate over all feature pairs to find the most accurate
    for feats in feature_combinations:
        feats_arr = list(feats)
        
        print(f"Testing Features: {fNames[feats_arr]}")
        
        # Subset the training data to only these 2 features
        X_subset = X[:, feats_arr]
        
        # Note: Testing on the training data directly (as per original code logic)
        # In a real-world scenario, cross-validation or train_test_split is preferred.
        mdc_model.fit(X_subset, y)
        predictions = mdc_model.predict(X_subset)
        
        # Calculate accuracy
        correct = np.sum(predictions == y)
        acc_percentage = (correct / len(X)) * 100
        
        print(f"Correctly classified: {correct} out of {len(X)}")
        print(f"Percent Accuracy: {acc_percentage:.2f}%\n")
        
        if acc_percentage > max_acc:
            max_acc = acc_percentage
            best_feats = feats_arr

    # 4. Display Results
    print("=============================================")
    print(f"BEST COMBINATION: {fNames[best_feats]}")
    print(f"PEAK ACCURACY: {max_acc:.2f}%")
    print("=============================================")

    # 5. Plot the Best Combination
    title = f"MDC Scatter Diagram (Best Features) - Accuracy: {max_acc:.2f}%"
    selector.plot_2d_scatter(X[:, best_feats], y, title, fNames[best_feats])
