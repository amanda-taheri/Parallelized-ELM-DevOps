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
from .weight_synthesizer import WeightSynthesizer

class OnlineParallelELM:
    def __init__(self, input_size, hidden_size, n_workers=2, activation='sigmoid'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_workers = n_workers
        self.activation = activation
        
        self.synthesizer = WeightSynthesizer()
        self.input_weights = None
        self.biases = None

    def _svd_init(self, X):
        _, _, Vh = svd(X, full_matrices=False)
        weights = Vh[:self.hidden_size, :].T if self.hidden_size <= self.input_size else \
                  np.tile(Vh.T, (1, int(np.ceil(self.hidden_size / self.input_size))))[:, :self.hidden_size]
        return weights, np.random.randn(1, self.hidden_size)

    def _train_block(self, X_block, y_block):
        H = 1 / (1 + np.exp(-(np.dot(X_block, self.input_weights) + self.biases)))
        return np.dot(pinv(H), y_block.reshape(-1, 1))

    def learn_batch(self, X_batch, y_batch):
        """Processes a new batch and updates the global Knowledge Base."""
        if self.input_weights is None:
            self.input_weights, self.biases = self._svd_init(X_batch[:1000])

        # 1. Parallel local learning
        X_blocks = np.array_split(X_batch, self.n_workers)
        y_blocks = np.array_split(y_batch, self.n_workers)
        
        local_betas = Parallel(n_jobs=self.n_workers)(
            delayed(self._train_block)(X_blocks[i], y_blocks[i]) for i in range(self.n_workers)
        )

        # 2. Synthesis & KB Update
        batch_beta = self.synthesizer.synthesize(local_betas)
        self.synthesizer.update_knowledge_base(batch_beta)

    def predict(self, X):
        H = 1 / (1 + np.exp(-(np.dot(X, self.input_weights) + self.biases)))
        return np.dot(H, self.synthesizer.global_beta).flatten()
