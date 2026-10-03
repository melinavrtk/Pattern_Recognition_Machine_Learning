# =========================================================
# Pattern Recognition & Machine Learning Pipeline
# Step 1: Texture Feature Extraction (1st and 2nd Order)
# =========================================================

import numpy as np
import cv2
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops

class TextureFeatureExtractor:
    """
    A class to extract 1st-order and 2nd-order (GLCM) texture features from image ROIs.
    Matches MATLAB's default statistical behavior and GLCM properties.
    """
    
    def __init__(self, glcm_levels=32, glcm_distances=[1]):
        """
        Initialize the extractor with GLCM parameters.
        Angles used: 0, 45, 90, and 135 degrees (in radians).
        """
        self.glcm_levels = glcm_levels
        self.glcm_distances = glcm_distances
        self.glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    def extract_first_order(self, roi):
        """
        Extracts 1st-order statistical features: Mean, Standard Deviation, Skewness, Kurtosis.
        """
        roi_flat = roi.flatten().astype(float)
        
        # ddof=1 for sample standard deviation (MATLAB default)
        f_mean = np.mean(roi_flat)
        f_std = np.std(roi_flat, ddof=1) 
        
        # bias=False matches MATLAB's statistical formulas for skewness and kurtosis
        f_skew = skew(roi_flat, bias=False)
        f_kurt = kurtosis(roi_flat, bias=False, fisher=False) # fisher=False -> Pearson kurtosis
        
        return [f_mean, f_std, f_skew, f_kurt]

    def extract_second_order_glcm(self, roi):
        """
        Extracts 2nd-order GLCM features: Contrast, Correlation, Energy, Homogeneity.
        Computes the GLCM for 4 angles and returns the averaged properties.
        """
        # Quantize image to the specified number of gray levels (e.g., 256 -> 32)
        bins = np.linspace(0, 256, self.glcm_levels + 1)
        roi_quantized = np.digitize(roi, bins) - 1
        
        # Compute the GLCM matrix (symmetric and normalized as required for probabilities)
        glcm = graycomatrix(roi_quantized, distances=self.glcm_distances, 
                            angles=self.glcm_angles, levels=self.glcm_levels, 
                            symmetric=True, normed=True)
        
        # Extract properties and average them across all 4 angles for rotation invariance
        contrast = np.mean(graycoprops(glcm, 'contrast'))
        correlation = np.mean(graycoprops(glcm, 'correlation'))
        energy = np.mean(graycoprops(glcm, 'energy'))
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
        
        return [contrast, correlation, energy, homogeneity]

    def process_roi(self, roi):
        """
        Processes an input Region of Interest (ROI) and returns the full 8-element feature vector.
        """
        first_order = self.extract_first_order(roi)
        second_order = self.extract_second_order_glcm(roi)
        
        # Combine into a single feature vector
        return first_order + second_order

# =========================================================
# MAIN EXECUTION (Example Usage)
# =========================================================
if __name__ == "__main__":
    # Create a dummy ROI (e.g., 20x20 pixel area) to simulate processing
    dummy_roi = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
    
    # Initialize the extractor
    extractor = TextureFeatureExtractor(glcm_levels=32)
    
    # Process the ROI
    features = extractor.process_roi(dummy_roi)
    
    print("--- Texture Features Extracted ---")
    print("1st Order (Mean, Std, Skewness, Kurtosis):")
    print([f"{val:.4f}" for val in features[0:4]])
    print("\n2nd Order GLCM (Contrast, Correlation, Energy, Homogeneity):")
    print([f"{val:.4f}" for val in features[4:8]])
