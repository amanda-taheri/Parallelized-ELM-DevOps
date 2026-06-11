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

class WeightSynthesizer:
    """
    As per Phase 4: Manages the 'Knowledge Base' by synthesizing 
    output weights (Beta) from multiple parallel modules or batches.
    """
    def __init__(self, max_size=20, distance_threshold=2.0, min_reliability=None):
        self.max_size = max_size
        self.distance_threshold = distance_threshold
        self.min_reliability = min_reliability
        self.knowledge_base = []
        self.reliabilities = []
        self.global_beta = None
        self.n_updates = 0

    def synthesize(self, beta_list):
        """Combines local weights from parallel workers in a single batch."""
        betas = np.asarray(beta_list)
        center = np.mean(betas, axis=0)

        if len(betas) <= 2:
            return center

        distances = np.array([np.linalg.norm(beta - center) for beta in betas])
        cutoff = distances.mean() + distances.std()
        central_betas = betas[distances <= cutoff]
        return np.mean(central_betas, axis=0)

    def _is_eligible(self, new_beta, reliability):
        if not self.knowledge_base:
            return True

        if self.min_reliability is not None and reliability is not None:
            if reliability < self.min_reliability:
                return False

        center = np.mean(self.knowledge_base, axis=0)
        distances = np.array([np.linalg.norm(beta - center) for beta in self.knowledge_base])
        new_distance = np.linalg.norm(new_beta - center)

        if len(distances) < 2:
            return True

        threshold = distances.mean() + self.distance_threshold * (distances.std() + 1e-12)
        return new_distance <= threshold

    def _refresh_global_beta(self):
        if not self.knowledge_base:
            self.global_beta = None
        else:
            self.global_beta = np.mean(self.knowledge_base, axis=0)
        return self.global_beta

    def update_knowledge_base(self, new_beta, reliability=None):
        """
        Updates the fixed-length KB if the new beta is close enough to the
        current central model, optionally using evaluator reliability.
        """
        accepted = self._is_eligible(new_beta, reliability)
        if accepted:
            self.knowledge_base.append(new_beta)
            self.reliabilities.append(reliability)

            if len(self.knowledge_base) > self.max_size:
                if any(score is not None for score in self.reliabilities):
                    scores = [
                        -np.inf if score is None else score
                        for score in self.reliabilities
                    ]
                    remove_index = int(np.argmin(scores))
                else:
                    center = np.mean(self.knowledge_base, axis=0)
                    distances = [
                        np.linalg.norm(beta - center)
                        for beta in self.knowledge_base
                    ]
                    remove_index = int(np.argmax(distances))

                self.knowledge_base.pop(remove_index)
                self.reliabilities.pop(remove_index)

        self.n_updates += 1
        return self._refresh_global_beta()


