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
    def __init__(self):
        self.global_beta = None
        self.n_updates = 0

    def synthesize(self, beta_list):
        """Combines local weights from parallel workers in a single batch."""
        return np.mean(beta_list, axis=0)

    def update_knowledge_base(self, new_beta):
        """
        Incrementally updates the global knowledge using the new synthesized beta.
        Formula: Moving Average to preserve history.
        """
        if self.global_beta is None:
            self.global_beta = new_beta
        else:
            # Weighted update based on history
            self.global_beta = (self.global_beta * self.n_updates + new_beta) / (self.n_updates + 1)
        
        self.n_updates += 1
        return self.global_beta



