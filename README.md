Parallelized-ELM-DevOps
A custom implementation of Parallelized Extreme Learning Machine (P-ELM) for real-time anomaly detection. This project focuses on high-speed stream processing and efficient weight synthesis for online data classification.

Why this project?
Standard ELM is fast, but when dealing with massive DevOps data streams (like KDD-99), single-core training becomes a bottleneck. This implementation solves that by:

Parallelizing the hidden layer matrix (H) calculation across multiple CPU cores.
Implementing a Weight Synthesizer to merge knowledge from different data batches without retraining from scratch.
Using SVD-based initialization to ensure the model doesn’t “explode” during online updates.
Quick Stats (Tested on MacBook Pro M2 Max)
Dataset: KDD Cup 99 (10% subset)
Total Samples: ~494,021
Training Time: ~5.8 seconds (for 345k samples)
Accuracy: 98%
Recall (Anomaly): 0.99 (Crucial for not missing attacks)
Project Architecture
The code is split into modular components to keep things clean:

src/elm_base.py: The core ELM logic.
src/elm_svd.py: Added stability using Singular Value Decomposition.
src/weight_synthesizer.py: The logic for combining weights from parallel workers.
src/elm_online.py: The main class for stream/batch learning.
main.py: Interactive CLI tool to run the whole pipeline and monitor system resources.
How to Run
Clone the repo and install requirements:
bash
   pip install -r requirements.txt
Put your dataset in the data/ folder.
Fire up the main tool:
bash
   python3 main.py
Credits & Reference
This work is based on the methodology described in:

“Parallelized Extreme Learning Machine for Online Data Classification”

https://doi.org/10.1007/s10489-022-03308-7

Developed by Amanda Taheri - 2026
Distributed under the MIT License. See LICENSE for more information.
