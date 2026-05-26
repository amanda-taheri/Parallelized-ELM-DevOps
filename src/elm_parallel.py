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
from joblib import Parallel, delayed

class ParallelELM:
    def __init__(self, input_size, hidden_size, n_workers=2, activation='sigmoid'):
        """
        Phase 3: Parallelized ELM with Weight Synthesizer.
        As per the paper architecture: 
        Parallel Modules -> Weight Synthesizer -> Knowledge Base.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_workers = n_workers
        self.activation = activation
        
        # Shared initial weights (using SVD logic)
        self.input_weights = None
        self.biases = None
        # Final synthesized output weights
        self.output_weights = None
        self.classes_ = None

    def _svd_init(self, X):
        """Standard SVD initialization to ensure all parallel modules start correctly."""
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        _, _, Vh = svd(X_aug, full_matrices=False)
        n_components = Vh.shape[0]
        if self.hidden_size <= n_components:
            params = Vh[:self.hidden_size, :].T
        else:
            repeats = int(np.ceil(self.hidden_size / n_components))
            params = np.tile(Vh.T, (1, repeats))[:, :self.hidden_size]
        return params[:-1, :], params[-1:, :]

    def _prepare_targets(self, y):
        y = np.asarray(y)
        if y.ndim > 1:
            return y

        classes = np.unique(y) if self.classes_ is None else self.classes_
        self.classes_ = classes
        if len(classes) <= 2:
            if len(classes) == 2:
                class_to_index = {label: idx for idx, label in enumerate(classes)}
                return np.array([class_to_index[label] for label in y], dtype=float).reshape(-1, 1)
            return y.reshape(-1, 1)

        target = np.zeros((len(y), len(classes)))
        class_to_index = {label: idx for idx, label in enumerate(classes)}
        for row, label in enumerate(y):
            if label in class_to_index:
                target[row, class_to_index[label]] = 1.0
        return target

    def _activate(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        return np.maximum(0, x)

    def _train_single_module(self, X_block, y_block):
        """Trains a single ELM module on a data block."""
        H = self._activate(np.dot(X_block, self.input_weights) + self.biases)
        # Beta_i = H+ * T
        beta_i = np.dot(pinv(H), self._prepare_targets(y_block))
        return beta_i

    def fit(self, X, y):
        """
        Parallel Training Process:
        1. Global SVD Initialization.
        2. Partition data into blocks.
        3. Train modules in parallel.
        4. Synthesize weights (Weight Synthesizer).
        """
        # 1. Global Init (using a sample to set directions)
        sample_size = min(len(X), 1000)
        self.input_weights, self.biases = self._svd_init(X[:sample_size])

        # 2. Partition Data
        target = self._prepare_targets(y)
        X_blocks = np.array_split(X, self.n_workers)
        y_blocks = np.array_split(target, self.n_workers)

        # 3. Parallel Execution (Multi-core)
        # Each worker calculates its own Beta_i
        beta_list = Parallel(n_jobs=self.n_workers)(
            delayed(self._train_single_module)(X_blocks[i], y_blocks[i]) 
            for i in range(self.n_workers)
        )

        # 4. Weight Synthesizer (Simple Averaging for global knowledge)
        # The paper suggests combining local weights into the Knowledge Base
        self.output_weights = np.mean(beta_list, axis=0)

    def predict_scores(self, X):
        """Forward pass using synthesized global weights."""
        H = self._activate(np.dot(X, self.input_weights) + self.biases)
        scores = np.dot(H, self.output_weights)
        return scores.flatten() if scores.shape[1] == 1 else scores

    def predict(self, X):
        """Forward pass with class labels for multi-class targets."""
        scores = self.predict_scores(X)
        if scores.ndim == 1 or self.classes_ is None or len(self.classes_) <= 2:
            return scores
        return self.classes_[np.argmax(scores, axis=1)]
