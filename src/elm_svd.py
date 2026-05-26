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
from scipy.linalg import pinv, svd

class ELMSVD:
    def __init__(self, input_size, hidden_size, activation='sigmoid'):
        """
        ELM with SVD-based initialization as proposed in the paper.
        Phase 2: Enhancing stability and accuracy using Singular Value Decomposition.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        
        # Weights and biases to be initialized during fit
        self.input_weights = None
        self.biases = None
        self.output_weights = None

    def _svd_init(self, X):
        """
        Implements SVD-based weight initialization (refer to page 4 of the paper).
        Extracts principal components of the input data to set initial weights.
        """
        # Singular Value Decomposition: X = U * S * Vh
        # Vh contains the principal axes of the data
        _, _, Vh = svd(X, full_matrices=False)
        
        # Selection of weights from the Vh matrix (eigenvectors)
        if self.hidden_size <= self.input_size:
            # Use top principal components if hidden_size is small
            weights = Vh[:self.hidden_size, :].T
        else:
            # Tile/Repeat patterns if hidden_size is larger than input features
            repeats = int(np.ceil(self.hidden_size / self.input_size))
            weights = np.tile(Vh.T, (1, repeats))[:, :self.hidden_size]
            
        # Initialize biases randomly (or could be zeros)
        biases = np.random.randn(1, self.hidden_size)
        return weights, biases

    def _activate(self, x):
        """Hidden layer activation function."""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        return x

    def fit(self, X, y):
        """
        Trains the ELM model using SVD for input weights and 
        Moore-Penrose pseudo-inverse for output weights.
        """
        # Phase 2 Key: Use SVD instead of random initialization
        self.input_weights, self.biases = self._svd_init(X)

        # Calculate Hidden Layer Matrix (H)
        H = self._activate(np.dot(X, self.input_weights) + self.biases)
        
        # Calculate Output Weights (Beta) using pseudo-inverse
        # Solve H * Beta = T => Beta = H+ * T
        self.output_weights = np.dot(pinv(H), y.reshape(-1, 1))

    def predict(self, X):
        """Forward pass to generate predictions."""
        H = self._activate(np.dot(X, self.input_weights) + self.biases)
        return np.dot(H, self.output_weights).flatten()
