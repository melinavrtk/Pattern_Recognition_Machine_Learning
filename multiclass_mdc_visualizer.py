# =========================================================
# Pattern Recognition & Machine Learning Pipeline
# Step 6: Multi-Class Minimum Distance Classifier (MDC)
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

class MultiClassMDCVisualizer:
    """
    A multi-class visualizer for the Minimum Distance Classifier (MDC).
    Handles N classes dynamically and visually plots geometric distances.
    """

    @staticmethod
    def mdc_distance(x_pattern, class_data):
        """ 
        Computes the discriminant function g(x) for the MDC.
        Uses vectorized numpy operations for efficiency.
        """
        class_mean = np.mean(class_data, axis=0)
        
        # Discriminant function: g(x) = (μ * x) - 0.5 * (μ * μ)
        sum_a = np.dot(class_mean, x_pattern)
        sum_b = np.dot(class_mean, class_mean)
        
        return sum_a - 0.5 * sum_b

    @staticmethod
    def mdc_classifier(x_pattern, classes_list):
        """
        Dynamically classifies an unknown pattern across multiple classes.
        Returns the predicted class (1-indexed based) and the distances.
        """
        distances = []
        for i, class_data in enumerate(classes_list):
            d = MultiClassMDCVisualizer.mdc_distance(x_pattern, class_data)
            distances.append(d)
            
        print(f"Distances from each class: {np.round(distances, 2)}")
        
        # Find the class with the highest discriminant value (minimum distance)
        # Adding 1 because class indexing starts at 1 in our output
        classified = np.argmax(distances) + 1
        return classified, distances

    @staticmethod
    def classify_and_visualize(classes_list, x_pattern, feature_names=["Feature 1", "Feature 2"]):
        """
        Classifies the unknown pattern and generates a 2D scatter plot
        with geometric distance lines to all class centroids dynamically.
        """
        # 1. Classification
        classified, _ = MultiClassMDCVisualizer.mdc_classifier(x_pattern, classes_list)
        print(f"Pattern X={x_pattern.tolist()} is classified to: Class {classified}")

        # 2. Visualization Setup
        plt.figure(figsize=(7, 7))
        colors = ['b', 'r', 'g', 'm', 'c']
        markers = ['o', 'x', 's', 'v', '^']
        
        # List to keep track of legends
        legend_labels = []

        # 3. Plot each class and draw lines from the unknown pattern to their means
        for i, class_data in enumerate(classes_list):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Plot the class points
            plt.plot(class_data[:, 0], class_data[:, 1], marker, color=color, markersize=8)
            legend_labels.append(f'Class {i + 1}')
            
            # Calculate and plot the class mean
            class_mean = np.mean(class_data, axis=0)
            plt.plot(class_mean[0], class_mean[1], marker='D', color=color, markersize=10)
            
            # Draw dashed distance line from the unknown pattern to the mean
            plt.plot([x_pattern[0], class_mean[0]], [x_pattern[1], class_mean[1]], color='k', linestyle='--', alpha=0.5)

        # 4. Plot the unknown pattern
        plt.plot(x_pattern[0], x_pattern[1], 'k*', markersize=14)
        legend_labels.append('Unknown Pattern')

        # Formatting
        plt.grid(True)
        plt.legend(legend_labels, loc='best')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(f'Multi-Class MDC Scatter Diagram: {feature_names[0]} vs {feature_names[1]}')
        plt.show()

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    # Define 3 classes
    c1 = np.array([[1, 3], [-2, 4], [-1, 5]], dtype=float)
    c2 = np.array([[4, 5], [4, 8], [3, 6]], dtype=float)
    c3 = np.array([[6, 5], [7, 8], [5, 6]], dtype=float)
    
    # Pack classes into a list for dynamic processing
    all_classes = [c1, c2, c3]
    
    # Unknown pattern
    x_unknown = np.array([4, 9], dtype=float)
    
    # Run the visualizer
    MultiClassMDCVisualizer.classify_and_visualize(all_classes, x_unknown, feature_names=["feat1", "feat2"])
