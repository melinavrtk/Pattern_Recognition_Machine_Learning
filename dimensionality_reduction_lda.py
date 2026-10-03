# =========================================================
# Pattern Recognition & Machine Learning Pipeline
# Step 4: Dimensionality Reduction (Linear Discriminant Analysis - LDA)
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

class LDADimensionalityReduction:
    """
    Implements Linear Discriminant Analysis (LDA) for 2 classes.
    Projects high-dimensional data (e.g., 4D texture features) into 1D, 
    maximizing class separability based on the Fisher Criterion.
    """

    @staticmethod
    def linear_discriminant_analysis(X1, X2):
        """
        Equivalent to the MATLAB function: 
        [V, J, Y1, Y2] = lineardiscriminantanalysis(X1, X2)
        
        Inputs:
        X1, X2: Matrices for Class 1 and Class 2 (patterns x features)
        
        Outputs:
        V: Transformation vector
        J: Fisher criterion value
        Y1, Y2: Transformed 1D data
        """
        # Calculate class means
        m1 = np.mean(X1, axis=0)
        m2 = np.mean(X2, axis=0)
        
        # Calculate within-class scatter matrices (S1, S2) and total Sw
        S1 = np.dot((X1 - m1).T, (X1 - m1))
        S2 = np.dot((X2 - m2).T, (X2 - m2))
        Sw = S1 + S2
        
        # Calculate the transformation vector V = Sw^-1 * (m1 - m2)
        # Using pseudo-inverse for numerical stability
        Sw_inv = np.linalg.pinv(Sw)
        V = np.dot(Sw_inv, (m1 - m2))
        
        # Calculate the Fisher Criterion J = (V^T * Sb * V) / (V^T * Sw * V)
        Sb = np.outer((m1 - m2), (m1 - m2)) # Between-class scatter matrix
        J_numerator = np.dot(V.T, np.dot(Sb, V))
        J_denominator = np.dot(V.T, np.dot(Sw, V))
        J = J_numerator / J_denominator if J_denominator != 0 else 0
        
        # Transform the original data into the new 1D subspace
        Y1 = np.dot(X1, V)
        Y2 = np.dot(X2, V)
        
        return V, J, Y1, Y2

    @staticmethod
    def plot_lda_results(X1, X2, Y1, Y2):
        """
        Replicates the visual output of the assignment, plotting the 
        original feature space and the transformed 1D space.
        """
        plt.figure(figsize=(10, 12))
        
        # Plot 1: Original Data (Feature 1 vs Feature 2) -> Mean vs Std
        plt.subplot(4, 1, 1)
        plt.plot(X1[:, 0], X1[:, 1], 'ro', label='Class 1')
        plt.plot(X2[:, 0], X2[:, 1], 'bs', label='Class 2')
        plt.title('Original Data (Mean vs Std)')
        plt.grid(True)
        plt.legend()
        
        # Plot 2: Original Data (Feature 3 vs Feature 4) -> Skewness vs Kurtosis
        # Only plots if data has at least 4 features
        plt.subplot(4, 1, 2)
        if X1.shape[1] >= 4 and X2.shape[1] >= 4:
            plt.plot(X1[:, 2], X1[:, 3], 'ro', label='Class 1')
            plt.plot(X2[:, 2], X2[:, 3], 'bs', label='Class 2')
            plt.title('Original Data (Skewness vs Kurtosis)')
        else:
            plt.title('Not enough features for Plot 2')
        plt.grid(True)
        plt.legend()
        
        # Plot 3: Transformed Data (Scatter mapping)
        plt.subplot(4, 1, 3)
        # Assuming Y1, Y2 are 1D, we pair them against their indices for spread visualization
        plt.plot(Y1, np.arange(len(Y1)), 'rh', markersize=8, label='Class 1 (Transformed)')
        plt.plot(Y2, np.arange(len(Y2)), 'bp', markersize=8, label='Class 2 (Transformed)')
        plt.title('Transformed Data (Spread)')
        plt.grid(True)
        plt.legend()
        
        # Plot 4: Transformed Data (Projected perfectly onto a 1D line y=0)
        plt.subplot(4, 1, 4)
        plt.plot(Y1, np.zeros(len(Y1)), 'rh', markersize=8, label='Class 1')
        plt.plot(Y2, np.zeros(len(Y2)), 'bp', markersize=8, label='Class 2')
        plt.title('Transformed Data (Best Variable / 1D Projection)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# =========================================================
# MAIN EXECUTION (Example Usage)
# =========================================================
if __name__ == "__main__":
    # Example mock data representing 4 texture features (Mean, Std, Skewness, Kurtosis)
    # Class 1 Mock Data (5 patterns, 4 features)
    class1_mock = np.array([
        [45.1, 22.3, 0.4, 2.1], 
        [42.3, 20.1, 0.3, 1.9], 
        [46.8, 23.5, 0.5, 2.3], 
        [43.5, 21.0, 0.4, 2.0], 
        [44.0, 22.0, 0.4, 2.2]
    ])
    
    # Class 2 Mock Data (5 patterns, 4 features)
    class2_mock = np.array([
        [85.1, 42.3, -0.4, 3.1], 
        [82.3, 40.1, -0.3, 2.9], 
        [86.8, 43.5, -0.5, 3.3], 
        [83.5, 41.0, -0.4, 3.0], 
        [84.0, 42.0, -0.4, 3.2]
    ])
    
    lda_tool = LDADimensionalityReduction()
    
    # Run LDA Transformation
    V_matrix, Fisher_J, Y1_trans, Y2_trans = lda_tool.linear_discriminant_analysis(class1_mock, class2_mock)
    
    print("--- LDA Results ---")
    print(f"Fisher Criterion (J): {Fisher_J:.4f}")
    print(f"Transformation Vector (V): \n{V_matrix}")
    
    # Plot the results
    lda_tool.plot_lda_results(class1_mock, class2_mock, Y1_trans, Y2_trans)
