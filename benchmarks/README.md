# SAP_LLM Performance Benchmark Suite

Comprehensive performance benchmarking infrastructure for validating enterprise-grade performance targets.

## Performance Targets

- **P95 Latency**: <600ms per document
- **Throughput**: ≥100,000 documents/minute
- **Classification Accuracy**: ≥99%
- **Extraction F1 Score**: ≥97%
- **Routing Accuracy**: ≥99.5%
- **Memory Usage**: <16GB per worker
- **GPU Utilization**: >80% during processing

## Directory Structure

```
benchmarks/
├── data/
│   ├── sample_documents/  # 1000+ diverse test documents
│   └── ground_truth/      # Manually validated labels
├── scripts/
│   ├── run_latency_benchmark.py      # Latency measurements
│   ├── run_throughput_benchmark.py   # Throughput tests
│   ├── run_accuracy_benchmark.py     # Accuracy validation
│   ├── run_load_test.py              # Load testing with Locust
│   └── generate_test_data.py         # Create test datasets
├── results/                           # Benchmark results (JSON/CSV)
├── notebooks/                         # Interactive analysis
└── README.md
```

## Installation

```bash
# Install benchmark dependencies
pip install -r benchmarks/requirements.txt

# Generate test data
python benchmarks/scripts/generate_test_data.py --num-docs 1000

# Run all benchmarks
python benchmarks/scripts/run_all_benchmarks.py
```

## Running Benchmarks

### Latency Benchmark
```bash
python benchmarks/scripts/run_latency_benchmark.py \
    --num-documents 1000 \
    --concurrent-workers 1,10,50,100 \
    --output benchmarks/results/latency_results.json
```

### Throughput Benchmark
```bash
python benchmarks/scripts/run_throughput_benchmark.py \
    --duration 600 \
    --workers 1,2,4,8 \
    --output benchmarks/results/throughput_results.json
```

### Accuracy Benchmark
```bash
python benchmarks/scripts/run_accuracy_benchmark.py \
    --test-dataset benchmarks/data/ground_truth/ \
    --output benchmarks/results/accuracy_results.json
```

### Load Testing
```bash
# Run Locust load test
locust -f benchmarks/scripts/run_load_test.py \
    --host http://localhost:8000 \
    --users 100 \
    --spawn-rate 10 \
    --run-time 10m \
    --headless \
    --csv benchmarks/results/load_test
```

## Performance Profiling

### CPU Profiling with py-spy
```bash
py-spy record -o benchmarks/results/profile.svg \
    -- python benchmarks/scripts/run_throughput_benchmark.py
```

### Memory Profiling
```bash
python -m memory_profiler benchmarks/scripts/run_throughput_benchmark.py
```

### GPU Monitoring
```bash
# Monitor GPU during benchmarks
watch -n 1 nvidia-smi

# Or use gpustat
gpustat -i 1
```

## Analyzing Results

### Jupyter Notebooks
```bash
jupyter lab benchmarks/notebooks/
```

Open `performance_analysis.ipynb` for interactive visualization and analysis.

## Continuous Benchmarking

Add to CI/CD pipeline:
```yaml
# .github/workflows/benchmarks.yml
- name: Run Performance Benchmarks
  run: |
    python benchmarks/scripts/run_all_benchmarks.py --quick
    python benchmarks/scripts/check_regressions.py
```

## Report Generation

Generate comprehensive performance report:
```bash
python benchmarks/scripts/generate_report.py \
    --results benchmarks/results/ \
    --output docs/PERFORMANCE_REPORT.md
```

## Acceptance Criteria

✅ P95 latency measured and documented
✅ Throughput at scale measured (100k+ docs/min)
✅ Accuracy validated on 1000+ documents
✅ Resource usage profiled under load
✅ Performance report published
✅ Bottlenecks identified with recommendations
✅ Continuous benchmarking in CI/CD

## Troubleshooting

### GPU Not Available
If CUDA is not available, benchmarks will run on CPU (slower but functional).

### Out of Memory
Reduce batch sizes or concurrent workers:
```bash
python benchmarks/scripts/run_throughput_benchmark.py --batch-size 8
```

### Network Errors
For load testing, ensure the API server is running:
```bash
uvicorn sap_llm.api.main:app --host 0.0.0.0 --port 8000
```
