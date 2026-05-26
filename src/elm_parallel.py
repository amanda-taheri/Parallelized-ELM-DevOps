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

    def _svd_init(self, X):
        """Standard SVD initialization to ensure all parallel modules start correctly."""
        _, _, Vh = svd(X, full_matrices=False)
        if self.hidden_size <= self.input_size:
            weights = Vh[:self.hidden_size, :].T
        else:
            repeats = int(np.ceil(self.hidden_size / self.input_size))
            weights = np.tile(Vh.T, (1, repeats))[:, :self.hidden_size]
        biases = np.random.randn(1, self.hidden_size)
        return weights, biases

    def _activate(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        return np.maximum(0, x)

    def _train_single_module(self, X_block, y_block):
        """Trains a single ELM module on a data block."""
        H = self._activate(np.dot(X_block, self.input_weights) + self.biases)
        # Beta_i = H+ * T
        beta_i = np.dot(pinv(H), y_block.reshape(-1, 1))
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
        X_blocks = np.array_split(X, self.n_workers)
        y_blocks = np.array_split(y, self.n_workers)

        # 3. Parallel Execution (Multi-core)
        # Each worker calculates its own Beta_i
        beta_list = Parallel(n_jobs=self.n_workers)(
            delayed(self._train_single_module)(X_blocks[i], y_blocks[i]) 
            for i in range(self.n_workers)
        )

        # 4. Weight Synthesizer (Simple Averaging for global knowledge)
        # The paper suggests combining local weights into the Knowledge Base
        self.output_weights = np.mean(beta_list, axis=0)

    def predict(self, X):
        """Forward pass using synthesized global weights."""
        H = self._activate(np.dot(X, self.input_weights) + self.biases)
        return np.dot(H, self.output_weights).flatten()
