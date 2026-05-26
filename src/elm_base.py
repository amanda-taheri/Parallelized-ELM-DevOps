"""
Implementation of Parallelized Extreme Learning Machine (P-ELM)
Target: Online Data Classification & DevOps Anomaly Detection

Author: Amanda Taheri
Based on the research paper:
Title: Parallelized Extreme Learning Machine for Online Data Classification
https://doi.org/10.1007/s10489-022-03308-7
Copyright (c) 2026. All rights reserved.

"""


import numpy as np
from scipy.linalg import pinv

class ELMBase:
    """
    Base Extreme Learning Machine class tailored for DevOps data.
    This implementation follows the fundamental ELM theory (H * Beta = T).
    """
    def __init__(self, input_size, hidden_size, activation='sigmoid'):
        # Initialize basic architecture parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_type = activation
        
        # Weights and biases (Phase 1: Random, Phase 2: SVD-based)
        # We use a seed for reproducibility during our DevOps experiments
        np.random.seed(42)
        self.input_weights = np.random.normal(size=(self.input_size, self.hidden_size))
        self.biases = np.random.normal(size=(self.hidden_size))
        
        # Output weights (Beta) will be computed during training
        self.output_weights = None

    def _activate(self, x):
        """
        Activation function: In DevOps anomaly detection, sigmoid or relu 
        are common to capture non-linear resource spikes.
        """
        if self.activation_type == 'sigmoid':
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        elif self.activation_type == 'relu':
            return np.maximum(0, x)
        return x

    def fit(self, X, y):
        """
        Train the ELM using the Moore-Penrose pseudo-inverse.
        Formula: Beta = pinv(H) * T
        """
        # Step 1: Calculate the Hidden Layer Matrix (H)
        # H = G(W * X + b)
        H = self._activate(np.dot(X, self.input_weights) + self.biases)
        
        # Step 2: Solve for output weights (Beta)
        # Using pinv (Singular Value Decomposition based pseudo-inverse)
        self.output_weights = np.dot(pinv(H), y)
        print(f"Model fitted. Output weights shape: {self.output_weights.shape}")

    def predict(self, X):
        """
        Generate predictions for new DevOps monitoring samples.
        """
        H = self._activate(np.dot(X, self.input_weights) + self.biases)
        return np.dot(H, self.output_weights)
