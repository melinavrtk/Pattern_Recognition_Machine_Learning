# =========================================================
# Pattern Recognition & Machine Learning Pipeline
# Step 5: Minimum Distance Classifier (MDC) Visualization
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

class MDCVisualizer:
    """ 
    A visualizer for the Minimum Distance Classifier (MDC).
    Calculates discriminant functions and visually plots the distances 
    between an unknown pattern and the class centroids.
    """

    @staticmethod
    def mdc_distance(x_pattern, class_data):
        """ 
        Computes the discriminant function g(x) for the Minimum Distance Classifier.
        Uses vectorized numpy operations for efficiency.
        """
        # Calculate the mean vector for the class across columns (features)
        class_mean = np.mean(class_data, axis=0)
        
        # Discriminant function: g(x) = (μ * x) - 0.5 * (μ * μ)
        sum_a = np.dot(class_mean, x_pattern)
        sum_b = np.dot(class_mean, class_mean)
        
        return sum_a - 0.5 * sum_b

    @staticmethod
    def classify_and_visualize(class1, class2, x_pattern):
        """
        Classifies the unknown pattern and generates a 2D scatter plot
        with geometric distance lines to class centroids.
        """
        # 1. Calculate class means
        mean1 = np.mean(class1, axis=0)
        mean2 = np.mean(class2, axis=0)

        # 2. Classification
        d1 = MDCVisualizer.mdc_distance(x_pattern, class1)
        d2 = MDCVisualizer.mdc_distance(x_pattern, class2)

        print(f"--- Classification Results ---")
        print(f"Discriminant Value (Class 1): d1 = {d1:.2f}")
        print(f"Discriminant Value (Class 2): d2 = {d2:.2f}")
        
        classified = 1 if d1 > d2 else 2
        print(f"Pattern X={x_pattern.tolist()} is classified to: Class {classified}")

        # 3. Visualization
        plt.figure(figsize=(8, 6))
        
        # Plot class points
        plt.plot(class1[:, 0], class1[:, 1], 'bo', markersize=8, label='Class 1')
        plt.plot(class2[:, 0], class2[:, 1], 'rx', markersize=8, label='Class 2')
        
        # Plot class centroids (means)
        plt.plot(mean1[0], mean1[1], 'bD', markersize=10, label='Mean Class 1')
        plt.plot(mean2[0], mean2[1], 'rD', markersize=10, label='Mean Class 2')
        
        # Plot unknown pattern
        plt.plot(x_pattern[0], x_pattern[1], 'k*', markersize=14, label='Unknown Pattern')

        # Draw dashed distance lines from the unknown pattern to the means
        plt.plot([x_pattern[0], mean1[0]], [x_pattern[1], mean1[1]], 'b--', alpha=0.6)
        plt.plot([x_pattern[0], mean2[0]], [x_pattern[1], mean2[1]], 'r--', alpha=0.6)

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("MDC Geometric Visualization")
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    # Define 2D feature patterns for two classes
    c1 = np.array([[5, 7], [4, 8], [6, 7]], dtype=float)
    c2 = np.array([[2, 5], [2, 8], [2, 6]], dtype=float)
    
    # Unknown pattern to classify
    x_p = np.array([3, 8], dtype=float)
    
    # Run visualization
    MDCVisualizer.classify_and_visualize(c1, c2, x_p)
