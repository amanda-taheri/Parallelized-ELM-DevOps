# Benchmark + README patch

Copy these files into your project root:

- `test.py`
- `src/elm_online.py`

Then run one of these commands:

```bash
python3 test.py --task multiclass --mode both --workers 2
```

For the faster DevOps binary benchmark:

```bash
python3 test.py --task binary --mode models --workers 2 --max-samples 120000
```

Outputs:

- `assets/benchmark_results.csv`
- `assets/batch_sweep_results.csv`
- `assets/accuracy.png`
- `assets/performance.png`
- `assets/resources.png`
- `assets/confusion_matrix.png`
- `assets/batch_sweep.png`
- updated README section between `<!-- BENCHMARK_RESULTS_START -->` and `<!-- BENCHMARK_RESULTS_END -->`
