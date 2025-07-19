# GPU Performance Benchmarks

This module contains CUDA implementations for GPU performance benchmarking, specifically focusing on memory bandwidth measurements and computational throughput analysis.

## Prerequisites

- NVIDIA GPU with CUDA capability
- CUDA toolkit installed (nvcc compiler)
- Access to SLURM cluster (for job submission)

## Files

- `memory-bandwidth-benchmark.cu` - CUDA implementation of memory bandwidth benchmark
- `performance-visualizer.py` - Python script to visualize performance results
- `benchmark-job.sh` - SLURM job script for cluster execution

## Building

Compile the CUDA code:
```bash
make
```

## Running

### Local execution:
```bash
./memory-bandwidth-benchmark
```

### Cluster execution:
```bash
sbatch benchmark-job.sh
```

## Performance Analysis

After running the benchmark, use the Python plotting script to visualize results:
```bash
python performance-visualizer.py
```

This will generate performance plots showing GFLOPS (Giga Floating Point Operations Per Second) metrics.