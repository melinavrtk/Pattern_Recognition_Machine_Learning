# =========================================================
# Pattern Recognition & Machine Learning Pipeline
# Step 13: Advanced Clustering Suite & Stability Evaluation
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import warnings

# Clustering Algorithms
from sklearn.cluster import (KMeans, Birch, AgglomerativeClustering, 
                             SpectralClustering, AffinityPropagation, 
                             DBSCAN, MeanShift, estimate_bandwidth)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

warnings.simplefilter('ignore')

class AdvancedClusteringSuite:
    """
    Evaluates multiple Unsupervised Learning algorithms (Clustering).
    Assesses model stability over multiple epochs and visualizes cluster assignments.
    """

    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.algorithms = {
            "K-Means": KMeans(n_clusters=n_clusters, n_init='auto', algorithm='elkan', random_state=None),
            "Birch": Birch(threshold=0.05, n_clusters=n_clusters),
            "Hierarchical": AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean'),
            "Gaussian Mixture": GaussianMixture(n_components=n_clusters),
            "Spectral": SpectralClustering(n_clusters=n_clusters, assign_labels='discretize'),
            "Affinity Propagation": AffinityPropagation(preference=-10, damping=0.9),
            "DBSCAN": DBSCAN(eps=0.29, min_samples=40),
            "MeanShift": MeanShift()
        }

    @staticmethod
    def load_cervical_cancer_data():
        """ Loads a subset of the Cervical Cancer dataset (Survivors vs Non-Survivors). """
        # Class 1: Non-survivors
        class1 = [
            [91.13, 22.355], [90.575, 29.5], [92.832, 28.751], [88.181, 28.642],
            [90.709, 30.939], [104.05, 35.792], [98.342, 36.193], [99.269, 26.342],
            [94.845, 31.323], [161.65, 17.314], [139.76, 19.887], [146.79, 20.749],
            [135.71, 21.629], [132.78, 19.188], [144.19, 19.955], [137.0, 22.193],
            [130.64, 20.973], [138.88, 20.66], [131.78, 22.323], [132.77, 22.599],
            [138.33, 21.06], [142.46, 26.016], [158.37, 22.355], [132.62, 22.586],
            [154.72, 21.404], [140.22, 26.547], [140.45, 25.769], [138.71, 25.652],
            [138.46, 25.098]
        ]
        
        # Class 2: Survivors
        class2 = [
            [131.35, 38.58], [164.11, 21.972], [150.05, 22.108], [169.52, 22.394],
            [159.44, 23.22], [145.39, 16.517], [108.82, 24.038], [127.11, 22.186],
            [130.45, 17.907], [118.8, 25.907], [138.8, 25.896], [151.18, 18.881],
            [156.28, 24.799], [153.32, 29.878], [140.85, 27.791], [140.92, 29.667],
            [137.01, 29.869], [147.71, 28.388], [113.36, 29.952], [123.92, 30.482],
            [136.87, 23.505], [136.36, 24.655], [130.26, 27.7], [143.38, 26.278],
            [138.7, 26.622], [149.64, 24.757]
        ]
        
        X = np.concatenate((class1, class2), axis=0)
        y = np.concatenate((np.zeros(len(class1), dtype=int), np.ones(len(class2), dtype=int)), axis=0)
        
        # Normalize features along columns
        X_norm = normalize(X, axis=0)
        return X_norm, y, ["Mean", "Standard Deviation"]

    def run_clustering(self, X, algorithm_name):
        """ Fits the specified clustering algorithm and returns predicted labels. """
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not found.")
            
        model = self.algorithms[algorithm_name]
        
        # MeanShift requires dynamic bandwidth estimation based on the input data
        if algorithm_name == "MeanShift":
            bandwidth = estimate_bandwidth(X, quantile=0.62, n_samples=len(X))
            model = MeanShift(bandwidth=bandwidth)
            
        if hasattr(model, 'fit_predict'):
            y_pred = model.fit_predict(X)
        else:
            model.fit(X)
            y_pred = model.predict(X)
            
        return y_pred

    def map_clusters_to_ground_truth(self, y_true, y_pred):
        """ Aligns arbitrary cluster labels (0, 1) to the actual ground truth to calculate accuracy. """
        acc = accuracy_score(y_true, y_pred)
        # If accuracy is below 50% in a binary classification, the cluster labels are likely flipped
        if acc < 0.5:
            y_pred_flipped = 1 - y_pred
            return y_pred_flipped, accuracy_score(y_true, y_pred_flipped)
        return y_pred, acc

    def evaluate_stability(self, X, y, algorithm_name, epochs=10):
        """ Runs the algorithm for multiple epochs to evaluate initialization stability. """
        print(f"\n--- Evaluating Stability: {algorithm_name} ({epochs} Epochs) ---")
        
        accuracies = []
        best_y_pred = None
        max_acc = 0
        
        for epoch in range(epochs):
            y_pred = self.run_clustering(X, algorithm_name)
            y_pred_aligned, acc = self.map_clusters_to_ground_truth(y, y_pred)
            
            accuracies.append(acc * 100)
            
            if acc > max_acc:
                max_acc = acc
                best_y_pred = y_pred_aligned

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"Epoch Accuracies: {np.round(accuracies, 2)}")
        print(f"Mean Accuracy: {mean_acc:.2f}% | Std Dev: {std_acc:.2f}%")
        
        return best_y_pred

    @staticmethod
    def plot_comparison(X, y_true, y_pred, title, f_names):
        """ Plots Ground Truth vs. Clustering Results side-by-side. """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Ground Truth
        for i_class in np.unique(y_true):
            idx = np.where(y_true == i_class)
            ax1.scatter(X[idx, 0], X[idx, 1], label=f'Class {i_class}')
        ax1.set_title('Original Data (Ground Truth)')
        ax1.set_xlabel(f_names[0])
        ax1.set_ylabel(f_names[1])
        ax1.grid(True)
        ax1.legend(loc='best')
        
        # Plot 2: Clustering Result
        for i_cluster in np.unique(y_pred):
            idx = np.where(y_pred == i_cluster)
            ax2.scatter(X[idx, 0], X[idx, 1], label=f'Cluster {i_cluster}')
        ax2.set_title(f'Clustering Results ({title})')
        ax2.set_xlabel(f_names[0])
        ax2.set_ylabel(f_names[1])
        ax2.grid(True)
        ax2.legend(loc='best')
        
        plt.tight_layout()
        plt.show()

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    suite = AdvancedClusteringSuite(n_clusters=2)
    
    # Load and prepare data
    X, y, fNames = suite.load_cervical_cancer_data()
    
    # Choose clustering algorithm to test
    target_algorithm = "Gaussian Mixture"
    
    # Evaluate stability over 10 epochs
    best_predictions = suite.evaluate_stability(X, y, target_algorithm, epochs=10)
    
    # Visualize the best epoch result vs Ground Truth
    suite.plot_comparison(X, y, best_predictions, target_algorithm, fNames)
