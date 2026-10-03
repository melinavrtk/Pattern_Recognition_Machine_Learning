# =========================================================
# Pattern Recognition & Machine Learning Pipeline
# Step 14: Deep Learning - Multi-Layer Perceptron (TensorFlow/Keras)
# =========================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split

# Suppress verbose TensorFlow warnings for a cleaner console output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class NeuralNetworkEvaluator:
    """
    Evaluates different feature combinations using a Deep Learning approach.
    Implements a Multi-Layer Perceptron (MLP) via TensorFlow/Keras to find 
    the optimal feature pair for classification.
    """

    @staticmethod
    def generate_synthetic_data(n_samples_per_class=200, n_classes=2, n_feats=4, dist=0.5):
        """
        Generates Gaussian-distributed synthetic data for testing the neural network.
        Uses fast vectorized numpy operations instead of iterative loops.
        """
        print(f"--- Generating Synthetic Data ({n_classes} Classes, {n_feats} Features) ---")
        X_list, y_list = [], []
        
        for i in range(n_classes):
            # Shift the mean based on the distance multiplier to separate classes
            mean = i * dist
            std = 0.25
            
            # Generate feature matrix for the class
            class_data = np.random.normal(loc=mean, scale=std, size=(n_samples_per_class, n_feats))
            class_labels = np.full(n_samples_per_class, i, dtype=int)
            
            X_list.append(class_data)
            y_list.append(class_labels)
            
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        feature_names = np.array([f"Feat_{i}" for i in range(n_feats)])
        
        return X, y, feature_names

    @staticmethod
    def build_mlp_model(input_dim, num_classes):
        """ Builds and compiles a Sequential Keras MLP model. """
        model = Sequential([
            Dense(input_dim * 6, activation='relu', kernel_initializer='he_normal', input_shape=(input_dim,)),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model

    def evaluate_feature_pairs(self, X, y, feature_names):
        """
        Iterates over all 2D feature combinations, trains a Neural Network 
        on each, and identifies the best performing pair.
        """
        n_classes = len(np.unique(y))
        feature_pairs = list(combinations(range(len(feature_names)), 2))
        
        best_accuracy = -1
        best_indices = None
        best_model = None

        print("\n--- Training MLP Neural Network across Feature Combinations ---")
        
        for indices in feature_pairs:
            idx_list = list(indices)
            X_sub = X[:, idx_list]
            f_names_sub = feature_names[idx_list]
            
            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(X_sub, y, test_size=0.33, random_state=42)
            
            # Build and Train Model
            model = self.build_mlp_model(input_dim=2, num_classes=n_classes)
            # verbose=0 keeps the console clean during epochs
            model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
            
            # Evaluate Model
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"Combination {f_names_sub}: Accuracy {accuracy * 100:.2f}%")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_indices = idx_list
                best_model = model

        print("=============================================")
        print(f"BEST COMBINATION: {feature_names[best_indices]}")
        print(f"PEAK ACCURACY: {best_accuracy * 100:.2f}%")
        print("=============================================\n")
        
        return best_indices, best_accuracy

    @staticmethod
    def plot_best_combination(X, y, best_indices, feature_names, accuracy):
        """ Visualizes the 2D feature space that yielded the highest NN accuracy. """
        X_best = X[:, best_indices]
        names_best = feature_names[best_indices]
        n_classes = len(np.unique(y))
        
        plt.figure(figsize=(8, 6))
        colors = ['b', 'r', 'g', 'm']
        
        for i_class in range(n_classes):
            idx = np.where(y == i_class)
            plt.scatter(X_best[idx, 0], X_best[idx, 1], c=colors[i_class % len(colors)], label=f'Class {i_class}')
            
        title = f"MLP Neural Network Feature Space\nBest Accuracy: {accuracy * 100:.2f}%"
        plt.title(title)
        plt.xlabel(names_best[0])
        plt.ylabel(names_best[1])
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    nn_evaluator = NeuralNetworkEvaluator()
    
    # 1. Generate Fake Data
    X_data, y_data, f_names = nn_evaluator.generate_synthetic_data(
        n_samples_per_class=200, 
        n_classes=2, 
        n_feats=4, 
        dist=0.5
    )
    
    # 2. Train NNs and Find Best Features
    best_feats, peak_acc = nn_evaluator.evaluate_feature_pairs(X_data, y_data, f_names)
    
    # 3. Visualize
    nn_evaluator.plot_best_combination(X_data, y_data, best_feats, f_names, peak_acc)
