# =========================================================
# Pattern Recognition & Machine Learning Pipeline
# Step 3: Classifiers (Minimum Distance, Bayesian, k-NN)
# =========================================================

import numpy as np

class MinimumDistanceClassifier:
    """ Minimum Distance (MD) Classifier using class centroids. """
    
    def __init__(self):
        self.mean_a = None
        self.mean_b = None

    def fit(self, class_a, class_b):
        """ Calculates the centroid (mean vector) for each class. """
        self.mean_a = np.mean(class_a, axis=0)
        self.mean_b = np.mean(class_b, axis=0)

    def predict(self, x):
        """ Classifies an unknown pattern x based on the minimum distance discriminant function. """
        if self.mean_a is None or self.mean_b is None:
            raise ValueError("Classifier is not trained. Call 'fit' first.")
            
        # Discriminant function: g(x) = μ*x - 0.5*(μ*μ)
        g_a = np.dot(self.mean_a, x) - 0.5 * np.dot(self.mean_a, self.mean_a)
        g_b = np.dot(self.mean_b, x) - 0.5 * np.dot(self.mean_b, self.mean_b)
        
        if g_a > g_b:
            return 1
        elif g_b > g_a:
            return 2
        return 0


class BayesianClassifier:
    """ Gaussian Naive Bayes Classifier using covariance matrices. """
    
    def __init__(self, prior_a=0.5, prior_b=0.5):
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.mean_a = None
        self.mean_b = None
        self.cov_a = None
        self.cov_b = None

    def fit(self, class_a, class_b):
        """ Calculates the mean vectors and covariance matrices for each class. """
        self.mean_a = np.mean(class_a, axis=0)
        self.mean_b = np.mean(class_b, axis=0)
        
        # rowvar=False because rows are observations, columns are features
        self.cov_a = np.cov(class_a, rowvar=False)
        self.cov_b = np.cov(class_b, rowvar=False)

    def predict(self, x):
        """ Classifies an unknown pattern x using the Gaussian Bayes discriminant function. """
        if self.cov_a is None or self.cov_b is None:
            raise ValueError("Classifier is not trained. Call 'fit' first.")
            
        def discriminant(x, mean_vec, cov_mat, prior):
            # g(x) = ln(P) - 0.5*ln(|S|) - 0.5*(x-μ)^T * S^-1 * (x-μ)
            inv_cov = np.linalg.inv(cov_mat)
            det_cov = np.linalg.det(cov_mat)
            
            term1 = np.log(prior)
            term2 = -0.5 * np.log(det_cov)
            diff = x - mean_vec
            term3 = -0.5 * np.dot(np.dot(diff.T, inv_cov), diff)
            
            return term1 + term2 + term3

        g_a = discriminant(x, self.mean_a, self.cov_a, self.prior_a)
        g_b = discriminant(x, self.mean_b, self.cov_b, self.prior_b)
        
        if g_a > g_b:
            return 1
        elif g_b > g_a:
            return 2
        return 0


class KNNClassifier:
    """ k-Nearest Neighbors (k-NN) Classifier. """
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.Y_train = None

    def fit(self, X, Y):
        """ Stores the training data and labels (Lazy learning). """
        self.X_train = np.array(X)
        self.Y_train = np.array(Y)

    def predict(self, x):
        """ Classifies an unknown pattern x by finding the k-nearest neighbors. """
        if self.X_train is None:
            raise ValueError("Classifier is not fitted. Call 'fit' first.")
            
        # Calculate Euclidean distances from x to all training patterns
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        
        # Get indices of the k smallest distances
        k_nearest_indices = np.argsort(distances)[:self.k]
        
        # Get the classes of those k neighbors
        k_nearest_classes = self.Y_train[k_nearest_indices]
        
        # Count votes
        count_class_1 = np.sum(k_nearest_classes == 1)
        count_class_2 = np.sum(k_nearest_classes == 2)
        
        if count_class_1 > count_class_2:
            return 1
        elif count_class_2 > count_class_1:
            return 2
        return 0

# =========================================================
# MAIN EXECUTION (Example Usage)
# =========================================================
if __name__ == "__main__":
    # Dummy normalized data for demonstration
    class_a_train = np.array([[0.1, 0.2], [0.15, 0.18], [0.08, 0.22]])
    class_b_train = np.array([[0.8, 0.9], [0.85, 0.88], [0.92, 0.81]])
    
    # Unknown pattern to classify
    x_unknown = np.array([0.12, 0.20])
    
    print(f"Unknown Pattern: {x_unknown}")
    
    # 1. Minimum Distance
    md_clf = MinimumDistanceClassifier()
    md_clf.fit(class_a_train, class_b_train)
    print(f"Minimum Distance Prediction: Class {md_clf.predict(x_unknown)}")
    
    # 2. Bayesian
    bayes_clf = BayesianClassifier()
    bayes_clf.fit(class_a_train, class_b_train)
    print(f"Bayesian Prediction: Class {bayes_clf.predict(x_unknown)}")
    
    # 3. k-NN
    knn_clf = KNNClassifier(k=3)
    X_combined = np.vstack((class_a_train, class_b_train))
    Y_labels = np.array([1, 1, 1, 2, 2, 2])
    knn_clf.fit(X_combined, Y_labels)
    print(f"k-NN (k=3) Prediction: Class {knn_clf.predict(x_unknown)}")
