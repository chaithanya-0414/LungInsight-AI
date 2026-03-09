# models/radiomics_extractor.py
# Radiomics feature extraction from segmented regions

import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
import warnings
warnings.filterwarnings('ignore')


class RadiomicsExtractor:
    """
    Extract quantitative radiomics features from medical images
    Features include shape, intensity, and texture characteristics
    """
    
    def __init__(self, image, mask):
        """
        Args:
            image: Grayscale image (H, W), values in [0, 1]
            mask: Binary segmentation mask (H, W)
        """
        self.image = image
        self.mask = (mask > 0.5).astype(np.uint8)
        self.features = {}
    
    def extract_all_features(self):
        """Extract all radiomics features"""
        self.extract_shape_features()
        self.extract_intensity_features()
        self.extract_texture_features()
        
        return self.features
    
    def extract_shape_features(self):
        """Extract shape-based features"""
        if not self.mask.any():
            self.features.update({
                'area_pixels': 0,
                'perimeter': 0,
                'compactness': 0,
                'eccentricity': 0,
                'solidity': 0,
                'extent': 0,
                'major_axis_length': 0,
                'minor_axis_length': 0,
                'aspect_ratio': 0
            })
            return
        
        # Get region properties
        props = measure.regionprops(self.mask)[0]
        
        # Area
        area = props.area
        
        # Perimeter
        perimeter = props.perimeter
        
        # Compactness (circularity)
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Eccentricity
        eccentricity = props.eccentricity
        
        # Solidity
        solidity = props.solidity
        
        # Extent
        extent = props.extent
        
        # Axis lengths
        major_axis = props.major_axis_length
        minor_axis = props.minor_axis_length
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
        
        self.features.update({
            'area_pixels': float(area),
            'perimeter': float(perimeter),
            'compactness': float(compactness),
            'eccentricity': float(eccentricity),
            'solidity': float(solidity),
            'extent': float(extent),
            'major_axis_length': float(major_axis),
            'minor_axis_length': float(minor_axis),
            'aspect_ratio': float(aspect_ratio)
        })
    
    def extract_intensity_features(self):
        """Extract intensity-based features"""
        # Get pixels within mask
        masked_pixels = self.image[self.mask > 0]
        
        if len(masked_pixels) == 0:
            self.features.update({
                'mean_intensity': 0,
                'std_intensity': 0,
                'min_intensity': 0,
                'max_intensity': 0,
                'median_intensity': 0,
                'intensity_range': 0,
                'skewness': 0,
                'kurtosis': 0,
                'energy': 0,
                'entropy': 0
            })
            return
        
        # Basic statistics
        mean_int = np.mean(masked_pixels)
        std_int = np.std(masked_pixels)
        min_int = np.min(masked_pixels)
        max_int = np.max(masked_pixels)
        median_int = np.median(masked_pixels)
        intensity_range = max_int - min_int
        
        # Higher-order statistics
        skewness = self._compute_skewness(masked_pixels)
        kurtosis = self._compute_kurtosis(masked_pixels)
        
        # Energy
        energy = np.sum(masked_pixels ** 2)
        
        # Entropy
        entropy = self._compute_entropy(masked_pixels)
        
        self.features.update({
            'mean_intensity': float(mean_int),
            'std_intensity': float(std_int),
            'min_intensity': float(min_int),
            'max_intensity': float(max_int),
            'median_intensity': float(median_int),
            'intensity_range': float(intensity_range),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'energy': float(energy),
            'entropy': float(entropy)
        })
    
    def extract_texture_features(self):
        """Extract texture features using GLCM (Gray Level Co-occurrence Matrix)"""
        # Convert image to uint8 for GLCM
        image_uint8 = (self.image * 255).astype(np.uint8)
        
        # Apply mask
        masked_image = image_uint8 * self.mask
        
        if not self.mask.any():
            self.features.update({
                'glcm_contrast': 0,
                'glcm_dissimilarity': 0,
                'glcm_homogeneity': 0,
                'glcm_energy': 0,
                'glcm_correlation': 0,
                'glcm_asm': 0
            })
            return
        
        # Compute GLCM
        # distances: pixel pair distance offsets
        # angles: pixel pair angles (0, 45, 90, 135 degrees)
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        try:
            glcm = graycomatrix(masked_image, distances=distances, angles=angles, 
                               levels=256, symmetric=True, normed=True)
            
            # Extract GLCM properties
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            asm = graycoprops(glcm, 'ASM').mean()
            
            self.features.update({
                'glcm_contrast': float(contrast),
                'glcm_dissimilarity': float(dissimilarity),
                'glcm_homogeneity': float(homogeneity),
                'glcm_energy': float(energy),
                'glcm_correlation': float(correlation),
                'glcm_asm': float(asm)
            })
        except Exception as e:
            print(f"Warning: GLCM computation failed: {e}")
            self.features.update({
                'glcm_contrast': 0,
                'glcm_dissimilarity': 0,
                'glcm_homogeneity': 0,
                'glcm_energy': 0,
                'glcm_correlation': 0,
                'glcm_asm': 0
            })
    
    def _compute_skewness(self, data):
        """Compute skewness"""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data):
        """Compute kurtosis"""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _compute_entropy(self, data):
        """Compute Shannon entropy"""
        if len(data) == 0:
            return 0.0
        
        # Create histogram
        hist, _ = np.histogram(data, bins=256, range=(0, 1))
        
        # Normalize
        hist = hist / hist.sum()
        
        # Remove zeros
        hist = hist[hist > 0]
        
        # Compute entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        return float(entropy)


def compute_radiomics_features(image, mask):
    """
    Convenience function to extract all radiomics features
    
    Args:
        image: Grayscale image (H, W), values in [0, 1]
        mask: Binary segmentation mask (H, W)
        
    Returns:
        features: Dictionary of radiomics features
    """
    extractor = RadiomicsExtractor(image, mask)
    features = extractor.extract_all_features()
    
    return features


def format_radiomics_report(features):
    """
    Format radiomics features into a readable report
    
    Args:
        features: Dictionary of radiomics features
        
    Returns:
        report: Formatted string report
    """
    report = "=== RADIOMICS FEATURES REPORT ===\n\n"
    
    # Shape features
    report += "SHAPE FEATURES:\n"
    report += f"  Area: {features.get('area_pixels', 0):.2f} pixels\n"
    report += f"  Perimeter: {features.get('perimeter', 0):.2f} pixels\n"
    report += f"  Compactness: {features.get('compactness', 0):.4f}\n"
    report += f"  Eccentricity: {features.get('eccentricity', 0):.4f}\n"
    report += f"  Solidity: {features.get('solidity', 0):.4f}\n"
    report += f"  Extent: {features.get('extent', 0):.4f}\n"
    report += f"  Major Axis: {features.get('major_axis_length', 0):.2f} pixels\n"
    report += f"  Minor Axis: {features.get('minor_axis_length', 0):.2f} pixels\n"
    report += f"  Aspect Ratio: {features.get('aspect_ratio', 0):.4f}\n\n"
    
    # Intensity features
    report += "INTENSITY FEATURES:\n"
    report += f"  Mean: {features.get('mean_intensity', 0):.4f}\n"
    report += f"  Std Dev: {features.get('std_intensity', 0):.4f}\n"
    report += f"  Min: {features.get('min_intensity', 0):.4f}\n"
    report += f"  Max: {features.get('max_intensity', 0):.4f}\n"
    report += f"  Median: {features.get('median_intensity', 0):.4f}\n"
    report += f"  Range: {features.get('intensity_range', 0):.4f}\n"
    report += f"  Skewness: {features.get('skewness', 0):.4f}\n"
    report += f"  Kurtosis: {features.get('kurtosis', 0):.4f}\n"
    report += f"  Energy: {features.get('energy', 0):.2f}\n"
    report += f"  Entropy: {features.get('entropy', 0):.4f}\n\n"
    
    # Texture features
    report += "TEXTURE FEATURES (GLCM):\n"
    report += f"  Contrast: {features.get('glcm_contrast', 0):.4f}\n"
    report += f"  Dissimilarity: {features.get('glcm_dissimilarity', 0):.4f}\n"
    report += f"  Homogeneity: {features.get('glcm_homogeneity', 0):.4f}\n"
    report += f"  Energy: {features.get('glcm_energy', 0):.4f}\n"
    report += f"  Correlation: {features.get('glcm_correlation', 0):.4f}\n"
    report += f"  ASM: {features.get('glcm_asm', 0):.4f}\n"
    
    return report
