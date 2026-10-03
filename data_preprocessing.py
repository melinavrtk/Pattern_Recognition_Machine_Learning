# =========================================================
# Pattern Recognition & Machine Learning Pipeline
# Step 2: Data Preprocessing & Normalization
# =========================================================

import numpy as np

class DataPreprocessor:
    """
    A robust toolkit for merging datasets, creating labels, and applying 
    mathematical normalization (Min-Max & Standardization) from scratch.
    """
    
    def __init__(self):
        # Parameters stored during 'fit' to be applied to unknown test patterns
        self.min_vals = None
        self.max_vals = None
        self.mean_vals = None
        self.std_vals = None

    @staticmethod
    def join_classes(class_a, class_b):
        """
        Combines two class matrices into a single dataset X and generates 
        a label vector Y (1 for Class A, 2 for Class B).
        """
        X = np.vstack((class_a, class_b))
        
        # Create label vector: ones for A, twos for B
        labels_a = np.ones(len(class_a))
        labels_b = np.ones(len(class_b)) * 2
        Y = np.concatenate((labels_a, labels_b))
        
        return X, Y

    @staticmethod
    def split_classes(X, Y):
        """
        Splits a combined dataset X back into Class A and Class B 
        based on the label vector Y.
        """
        class_a = X[Y == 1]
        class_b = X[Y == 2]
        return class_a, class_b

    def fit_transform_minmax(self, X):
        """
        Normalizes the dataset X using Min-Max scaling [0, 1] and saves 
        the parameters for future test data.
        """
        self.min_vals = np.min(X, axis=0)
        self.max_vals = np.max(X, axis=0)
        
        # Prevent division by zero if a feature has no variance
        denominator = np.where((self.max_vals - self.min_vals) == 0, 1e-10, self.max_vals - self.min_vals)
        
        X_norm = (X - self.min_vals) / denominator
        return X_norm

    def transform_minmax(self, x_unknown):
        """
        Applies previously learned Min-Max parameters to an unknown pattern.
        """
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Min-Max parameters not found. Call 'fit_transform_minmax' first.")
            
        denominator = np.where((self.max_vals - self.min_vals) == 0, 1e-10, self.max_vals - self.min_vals)
        return (x_unknown - self.min_vals) / denominator

    def fit_transform_standardization(self, X):
        """
        Normalizes the dataset X using Z-score Standardization (Mean=0, Std=1) 
        and saves the parameters.
        """
        self.mean_vals = np.mean(X, axis=0)
        self.std_vals = np.std(X, axis=0, ddof=1) # ddof=1 matches MATLAB's default standard deviation
        
        denominator = np.where(self.std_vals == 0, 1e-10, self.std_vals)
        
        X_norm = (X - self.mean_vals) / denominator
        return X_norm

    def transform_standardization(self, x_unknown):
        """
        Applies previously learned Standardization parameters to an unknown pattern.
        """
        if self.mean_vals is None or self.std_vals is None:
            raise ValueError("Standardization parameters not found. Call 'fit_transform_standardization' first.")
            
        denominator = np.where(self.std_vals == 0, 1e-10, self.std_vals)
        return (x_unknown - self.mean_vals) / denominator

# =========================================================
# MAIN EXECUTION (Example Usage)
# =========================================================
if __name__ == "__main__":
    # Dummy data representing extracted features for Class A and Class B
    # 3 patterns per class, 2 features per pattern
    class_a_mock = np.array([[10, 2], [12, 3], [11, 2]])
    class_b_mock = np.array([[100, 20], [110, 22], [105, 21]])
    
    preprocessor = DataPreprocessor()
    
    # 1. Join classes
    X_combined, Y_labels = preprocessor.join_classes(class_a_mock, class_b_mock)
    print("Combined X (Original):\n", X_combined)
    
    # 2. Normalize (Standardization)
    X_standardized = preprocessor.fit_transform_standardization(X_combined)
    print("\nCombined X (Standardized):\n", np.round(X_standardized, 4))
    
    # 3. Process an unknown pattern using the saved parameters
    unknown_pattern = np.array([50, 10])
    unknown_standardized = preprocessor.transform_standardization(unknown_pattern)
    print("\nUnknown Pattern (Standardized):", np.round(unknown_standardized, 4))
    
    # 4. Split classes back for training
    A_norm, B_norm = preprocessor.split_classes(X_standardized, Y_labels)
    print("\nClass A (Standardized):\n", np.round(A_norm, 4))
