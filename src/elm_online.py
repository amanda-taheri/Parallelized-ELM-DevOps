"""
Paper-aligned Online Parallel ELM.

Key alignment points:
- Hidden nodes are expected to be set as min(batch_size, feature_dimension) by the runner.
- Parallelism is applied across independent mini-batches, not by splitting one mini-batch.
- A fixed-length Knowledge Base stores accepted beta vectors.
- An evaluator score can be used when deciding whether a beta enters the Knowledge Base.
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
        activation="sigmoid",
        kb_size=20,
        kb_distance_threshold=2.0,
        min_reliability=None,
        classes=None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_workers = n_workers
        self.activation = activation
        self.classes_ = None if classes is None else np.asarray(classes)

        self.synthesizer = WeightSynthesizer(
            max_size=kb_size,
            distance_threshold=kb_distance_threshold,
            min_reliability=min_reliability,
        )
        self.input_weights = None
        self.biases = None

    def set_classes(self, classes):
        """Freeze the label space before online mini-batch training starts."""
        self.classes_ = np.asarray(classes)

    def _svd_init(self, X):
        """SVD initialization for [w; b] using the augmented input matrix."""
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        _, _, Vh = svd(X_aug, full_matrices=False)

        n_components = Vh.shape[0]
        if self.hidden_size <= n_components:
            params = Vh[: self.hidden_size, :].T
        else:
            repeats = int(np.ceil(self.hidden_size / n_components))
            params = np.tile(Vh.T, (1, repeats))[:, : self.hidden_size]

        weights = params[:-1, :]
        biases = params[-1, :]
        return weights, biases

    def _activate(self, x):
        if self.activation == "sigmoid":
            x = np.clip(x, -500, 500)
            return 1.0 / (1.0 + np.exp(-x))
        if self.activation == "relu":
            return np.maximum(0, x)
        return x

    def _prepare_targets(self, y):
        """Convert labels to fixed one-hot targets for multi-class ELM."""
        y = np.asarray(y)
        if y.ndim > 1:
            return y.astype(float)

        if self.classes_ is None:
            self.classes_ = np.unique(y)

        if len(self.classes_) == 2:
            class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
            return np.array([class_to_index[label] for label in y], dtype=float).reshape(-1, 1)

        class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        target = np.zeros((len(y), len(self.classes_)), dtype=float)
        for row, label in enumerate(y):
            if label not in class_to_index:
                raise ValueError(f"Unknown class in online batch: {label}")
            target[row, class_to_index[label]] = 1.0
        return target

    def _train_block(self, X_block, y_block):
        H = self._activate(np.dot(X_block, self.input_weights) + self.biases)
        T = self._prepare_targets(y_block)
        return np.dot(pinv(H), T)

    def _predict_with_beta(self, X, beta):
        H = self._activate(np.dot(X, self.input_weights) + self.biases)
        scores = np.dot(H, beta)
        if scores.ndim == 2 and scores.shape[1] == 1:
            return scores.ravel()
        return scores

    def _labels_from_scores(self, scores):
        if scores.ndim == 1:
            encoded = (scores > 0.5).astype(int)
            if self.classes_ is not None and len(self.classes_) == 2:
                return self.classes_[encoded]
            return encoded
        return self.classes_[np.argmax(scores, axis=1)]

    def _evaluate_beta(self, beta, X_eval=None, y_eval=None):
        if X_eval is None or y_eval is None:
            return None
        scores = self._predict_with_beta(X_eval, beta)
        labels = self._labels_from_scores(scores)
        return float(np.mean(labels == np.asarray(y_eval)))

    def learn_batches(self, X_batches, y_batches, X_eval=None, y_eval=None):
        """
        Process a group of independent mini-batches in parallel.
        Each mini-batch produces one beta vector; accepted beta vectors enter KB.
        """
        if len(X_batches) != len(y_batches):
            raise ValueError("X_batches and y_batches must have the same length.")
        if not X_batches:
            return

        if self.input_weights is None:
            self.input_weights, self.biases = self._svd_init(X_batches[0])

        local_betas = Parallel(n_jobs=min(self.n_workers, len(X_batches)), prefer="threads")(
            delayed(self._train_block)(X_batches[i], y_batches[i])
            for i in range(len(X_batches))
        )

        # Paper-style KB: store/update beta vectors from individual parallel ELMs.
        for beta in local_betas:
            reliability = self._evaluate_beta(beta, X_eval, y_eval)
            self.synthesizer.update_knowledge_base(beta, reliability=reliability)

    def learn_batch(self, X_batch, y_batch, X_eval=None, y_eval=None):
        """Backward-compatible wrapper for processing a single mini-batch."""
        self.learn_batches([X_batch], [y_batch], X_eval=X_eval, y_eval=y_eval)

    def predict_scores(self, X):
        if self.synthesizer.global_beta is None:
            raise RuntimeError("The model has not learned any accepted beta yet.")
        return self._predict_with_beta(X, self.synthesizer.global_beta)

    def predict(self, X):
        return self._labels_from_scores(self.predict_scores(X))
