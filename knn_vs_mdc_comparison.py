# =========================================================
# Pattern Recognition & Machine Learning Pipeline
# Step 8: Algorithmic Comparison - k-NN vs MDC
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter('ignore')

class ClassifierComparator:
    """
    Directly compares the decision logic of a Minimum Distance Classifier (MDC) 
    and a k-Nearest Neighbors (k-NN) classifier for a given unknown pattern.
    """

    @staticmethod
    def mdc_distance(x_pattern, class_data):
        """ 
        Calculates the MDC discriminant function g(x).
        Higher value indicates closer proximity to the class mean.
        """
        class_mean = np.mean(class_data, axis=0)
        return np.dot(class_mean, x_pattern) - 0.5 * np.dot(class_mean, class_mean)

    @staticmethod
    def knn_distance(x_pattern, class_data, k):
        """ 
        Calculates the sum of squared Euclidean distances to the top K nearest 
        neighbors within a specific class. Lower value means closer proximity.
        """
        # Vectorized calculation of squared distances from x_pattern to all points
        distances = np.sum((class_data - x_pattern)**2, axis=1)
        
        # Sort and sum the distances of the K nearest neighbors
        sorted_distances = np.sort(distances)
        return np.sum(sorted_distances[:k])

    @staticmethod
    def run_comparison(class1, class2, x_pattern, k=1):
        """ Runs both classifiers and outputs the comparative results. """
        print(f"--- Evaluating Pattern X = {x_pattern.tolist()} ---")
        
        # --- MDC Classifier ---
        print("\n[1] Minimum Distance Classifier (MDC)")
        d1_mdc = ClassifierComparator.mdc_distance(x_pattern, class1)
        d2_mdc = ClassifierComparator.mdc_distance(x_pattern, class2)
        print(f"Discriminant d1 (Class 1): {d1_mdc:.2f}")
        print(f"Discriminant d2 (Class 2): {d2_mdc:.2f}")
        
        classified_mdc = 1 if d1_mdc > d2_mdc else 2
        print(f"-> MDC Prediction: Class {classified_mdc}")

        # --- k-NN Classifier ---
        print(f"\n[2] k-Nearest Neighbors (k={k})")
        d1_knn = ClassifierComparator.knn_distance(x_pattern, class1, k)
        d2_knn = ClassifierComparator.knn_distance(x_pattern, class2, k)
        print(f"Min Distance Sum d1 (Class 1): {d1_knn:.2f}")
        print(f"Min Distance Sum d2 (Class 2): {d2_knn:.2f}")
        
        # Note: For this KNN distance metric, smaller sum means closer
        classified_knn = 1 if d1_knn < d2_knn else 2
        print(f"-> k-NN Prediction: Class {classified_knn}\n")
        
        return classified_mdc, classified_knn

    @staticmethod
    def visualize(class1, class2, x_pattern):
        """ Simple scatter plot to visually verify the classification. """
        plt.figure(figsize=(7, 5))
        plt.plot(class1[:, 0], class1[:, 1], 'bo', markersize=8, label='Class 1')
        plt.plot(class2[:, 0], class2[:, 1], 'rx', markersize=8, label='Class 2')
        plt.plot(x_pattern[0], x_pattern[1], 'k*', markersize=14, label='Unknown Pattern (X)')
        
        # Show class means for MDC logic
        plt.plot(np.mean(class1[:, 0]), np.mean(class1[:, 1]), 'bD', markersize=10, alpha=0.5)
        plt.plot(np.mean(class2[:, 0]), np.mean(class2[:, 1]), 'rD', markersize=10, alpha=0.5)

        plt.grid(True)
        plt.title('Classifier Comparison Space')
        plt.legend(loc='best')
        plt.show()

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    # Define training classes
    c1 = np.array([[6, 7], [5, 8], [6, 6]], dtype=float)
    c2 = np.array([[3, 4], [2, 7], [3, 5]], dtype=float)
    
    # Define unknown pattern
    x_unknown = np.array([3, 8], dtype=float)
    
    # Run the comparison
    comparator = ClassifierComparator()
    comparator.run_comparison(c1, c2, x_unknown, k=1)
    
    # Visualize the problem space
    comparator.visualize(c1, c2, x_unknown)
