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
    def __init__(
        self,
        input_size,
        hidden_size,
        n_workers=2,
        activation='sigmoid',
        kb_size=20,
        kb_distance_threshold=2.0,
        min_reliability=None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_workers = n_workers
        self.activation = activation
        
        self.synthesizer = WeightSynthesizer(
            max_size=kb_size,
            distance_threshold=kb_distance_threshold,
            min_reliability=min_reliability,
        )
        self.input_weights = None
        self.biases = None
        self.classes_ = None

    def _svd_init(self, X):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        _, _, Vh = svd(X_aug, full_matrices=False)
        n_components = Vh.shape[0]
        if self.hidden_size <= n_components:
            params = Vh[:self.hidden_size, :].T
        else:
            repeats = int(np.ceil(self.hidden_size / n_components))
            params = np.tile(Vh.T, (1, repeats))[:, :self.hidden_size]
        return params[:-1, :], params[-1:, :]

    def _activate(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        return x

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

    def _train_block(self, X_block, y_block):
        H = self._activate(np.dot(X_block, self.input_weights) + self.biases)
        return np.dot(pinv(H), self._prepare_targets(y_block))

    def _evaluate_beta(self, beta, X_eval, y_eval):
        if X_eval is None or y_eval is None:
            return None

        scores = self._predict_with_beta(X_eval, beta)
        y_eval = np.asarray(y_eval)
        if scores.ndim == 1 or scores.shape[1] == 1:
            encoded = (scores.flatten() > 0.5).astype(int)
            if self.classes_ is not None and len(self.classes_) == 2:
                labels = self.classes_[encoded]
            else:
                labels = encoded.astype(y_eval.dtype)
        else:
            labels = self.classes_[np.argmax(scores, axis=1)]
        return float(np.mean(labels == y_eval))

    def _predict_with_beta(self, X, beta):
        H = self._activate(np.dot(X, self.input_weights) + self.biases)
        scores = np.dot(H, beta)
        return scores.flatten() if scores.shape[1] == 1 else scores

    def learn_batch(self, X_batch, y_batch, X_eval=None, y_eval=None):
        """Processes a new batch and updates the global Knowledge Base."""
        if self.input_weights is None:
            self.input_weights, self.biases = self._svd_init(X_batch[:1000])

        target_batch = self._prepare_targets(y_batch)

        # 1. Parallel local learning
        X_blocks = np.array_split(X_batch, self.n_workers)
        y_blocks = np.array_split(target_batch, self.n_workers)
        
        local_betas = Parallel(n_jobs=self.n_workers)(
            delayed(self._train_block)(X_blocks[i], y_blocks[i]) for i in range(self.n_workers)
        )

        # 2. Synthesis & KB Update
        batch_beta = self.synthesizer.synthesize(local_betas)
        reliability = self._evaluate_beta(batch_beta, X_eval, y_eval)
        self.synthesizer.update_knowledge_base(batch_beta, reliability=reliability)

    def predict_scores(self, X):
        return self._predict_with_beta(X, self.synthesizer.global_beta)

    def predict(self, X):
        scores = self.predict_scores(X)
        if scores.ndim == 1 or self.classes_ is None or len(self.classes_) <= 2:
            return scores
        return self.classes_[np.argmax(scores, axis=1)]
