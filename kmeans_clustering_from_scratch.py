# =========================================================
# Pattern Recognition & Machine Learning Pipeline
# Step 12: Unsupervised Learning - k-Means Clustering (From Scratch)
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    """
    Implements the k-Means clustering algorithm from scratch.
    Demonstrates unsupervised learning by iteratively updating centroids 
    and reassigning clusters until convergence.
    """

    def __init__(self, k=2, max_iters=50):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def initialize_labels(self, n_samples):
        """ Arbitrarily splits the initial dataset into k clusters. """
        labels = np.zeros(n_samples, dtype=int)
        split_size = n_samples // self.k
        for i in range(1, self.k):
            labels[i * split_size:] = i
        return labels

    def update_centroids(self, data, labels):
        """ Calculates the new centroids as the mean of the assigned data points. """
        centroids = np.zeros((self.k, data.shape[1]))
        for i in range(self.k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # If a cluster is empty, assign a random data point as its centroid
                centroids[i] = data[np.random.choice(data.shape[0])]
        return centroids

    def update_labels(self, data, centroids):
        """ Reassigns labels based on the closest Euclidean distance to a centroid. """
        # Calculate Euclidean distances from each point to each centroid using broadcasting
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        # Assign the label of the closest centroid
        new_labels = np.argmin(distances, axis=1)
        return new_labels

    def fit_and_visualize(self, data, feature_names=["Feature 1", "Feature 2"]):
        """ Runs the iterative k-Means process and visualizes each step. """
        n_samples = data.shape[0]
        self.labels = self.initialize_labels(n_samples)
        
        for iteration in range(self.max_iters):
            print(f"\n--- Iteration: {iteration + 1} ---")
            
            # Step 1: Find Centroids
            self.centroids = self.update_centroids(data, self.labels)
            print("Centroids:\n", np.round(self.centroids, 2))
            
            # Step 2: Visualize Current State
            title = f'k-Means Convergence (Iteration {iteration + 1})'
            self.plot_clusters(data, self.labels, self.centroids, title, feature_names)
            
            # Step 3: Reassign Labels
            new_labels = self.update_labels(data, self.centroids)
            print(f"Cluster Labels: {new_labels.tolist()}")
            
            # Step 4: Check for Convergence (if labels don't change, we're done)
            if np.array_equal(self.labels, new_labels):
                print(f"\n=> Algorithm converged successfully after {iteration + 1} iterations.")
                break
                
            self.labels = new_labels

    @staticmethod
    def plot_clusters(X, labels, centroids, title, fNames):
        """ Visualizes the 2D clusters and their centroids. """
        plt.figure(figsize=(7, 5))
        n_classes = len(np.unique(labels))
        colors = ['b', 'r', 'g', 'm']
        
        # Plot data points
        for i_class in range(n_classes):
            idx = np.where(labels == i_class)
            plt.scatter(X[idx, 0], X[idx, 1], s=80, c=colors[i_class % len(colors)], label=f'Cluster {i_class + 1}')
            
        # Plot centroids
        for i_class in range(len(centroids)):
            plt.plot(centroids[i_class, 0], centroids[i_class, 1], '*', markersize=20, 
                     markerfacecolor=colors[i_class % len(colors)], markeredgecolor='k', 
                     markeredgewidth=1.5, label=f'Centroid {i_class + 1}')
        
        plt.grid(True)
        plt.title(title)
        plt.xlabel(fNames[0])
        plt.ylabel(fNames[1])
        
        # Ensure unique legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')
        
        plt.show(block=False)
        plt.pause(1.5) # Pause to create an animation effect across iterations
        plt.close()

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    # Choose Data Mode: 0 for fixed dataset, 1 for random dataset
    choose_data = 0
    
    if choose_data == 0:
        dataset = np.array([[5, 3], [3, 7], [5, 2], [4, 2]], dtype=float)
    else:
        # Generate 100 random 2D points between 0 and 50
        np.random.seed(42) # For reproducibility
        dataset = np.round(np.random.rand(100, 2) * 50)
        
    print("--- Unsupervised Learning: k-Means Clustering ---")
    
    kmeans_model = KMeansClustering(k=2, max_iters=10)
    kmeans_model.fit_and_visualize(dataset)
