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
        self.classes_ = None

    def _svd_init(self, X):
        """
        Implements SVD-based weight initialization (refer to page 4 of the paper).
        Extracts principal components of the augmented input data to set initial
        weights and biases from V_r.
        """
        # Singular Value Decomposition: [X, 1] = U * S * Vh
        # Vh contains the principal axes for both input weights and bias.
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        _, _, Vh = svd(X_aug, full_matrices=False)
        
        # Selection of hidden-node parameters from Vh (eigenvectors).
        n_components = Vh.shape[0]
        if self.hidden_size <= n_components:
            # Use top principal components if hidden_size is small
            params = Vh[:self.hidden_size, :].T
        else:
            # Tile/Repeat patterns if hidden_size is larger than input features
            repeats = int(np.ceil(self.hidden_size / n_components))
            params = np.tile(Vh.T, (1, repeats))[:, :self.hidden_size]
            
        weights = params[:-1, :]
        biases = params[-1:, :]
        return weights, biases

    def _prepare_targets(self, y):
        y = np.asarray(y)
        if y.ndim > 1:
            return y

        classes = np.unique(y)
        self.classes_ = classes
        if len(classes) <= 2:
            if len(classes) == 2:
                class_to_index = {label: idx for idx, label in enumerate(classes)}
                return np.array([class_to_index[label] for label in y], dtype=float).reshape(-1, 1)
            return y.reshape(-1, 1)

        target = np.zeros((len(y), len(classes)))
        class_to_index = {label: idx for idx, label in enumerate(classes)}
        for row, label in enumerate(y):
            target[row, class_to_index[label]] = 1.0
        return target

    def _activate(self, x):
        """Hidden layer activation function."""
        if self.activation == 'sigmoid':
            x = np.clip(x, -500, 500)
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
        self.output_weights = np.dot(pinv(H), self._prepare_targets(y))

    def predict_scores(self, X):
        """Forward pass to generate predictions."""
        H = self._activate(np.dot(X, self.input_weights) + self.biases)
        scores = np.dot(H, self.output_weights)
        return scores.flatten() if scores.shape[1] == 1 else scores

    def predict(self, X):
        """Forward pass with class labels for multi-class targets."""
        scores = self.predict_scores(X)
        if scores.ndim == 1 or self.classes_ is None or len(self.classes_) <= 2:
            return scores
        return self.classes_[np.argmax(scores, axis=1)]
