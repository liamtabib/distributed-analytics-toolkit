# Distributed Analytics Toolkit

Toolkit for Bayesian sports ranking and GPU performance benchmarking, using advanced probabilistic inference algorithms and high-performance CUDA implementations.

## Overview

This project provides two analytical modules: a Bayesian ranking system for sports team analysis and a GPU benchmarking suite for performance evaluation. The toolkit combines statistical inference methods with high-performance computing, making it suitable for researchers, data scientists, and engineers working with probabilistic models and GPU optimization.

## Features

### Bayesian Ranking Module
- **Gibbs Sampling**: Implementation of MCMC algorithms for Bayesian inference
- **Message Passing Inference**: Factor graph-based probabilistic reasoning for sports ranking
- **Covariance Modeling**: Advanced correlation analysis between team performances
- **Visualization Pipeline**: Data plots and performance ranking visualizations

### GPU Benchmarking Module
- **CUDA Memory Bandwidth Testing**: High-performance memory bandwidth benchmarking
- **Matrix Operations Benchmarking**: Linear algebra evaluation
- **SLURM Integration**: Cluster computing job scheduling and execution using SLURM
- **Performance Analysis**: Comprehensive analysis using metrics TFLOPS and throughput
- **Comparative Benchmarking**: GPU vs CPU vs cuBLAS performance comparison

## Quick Setup

### Prerequisites
- Python 3.8+ with NumPy, SciPy, Pandas, Matplotlib
- NVIDIA GPU with CUDA Toolkit (for GPU benchmarks)
- SLURM cluster access (optional, for distributed computing)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/distributed-analytics-toolkit.git
cd distributed-analytics-toolkit

# Install Python dependencies
pip install -r requirements.txt

# Build GPU benchmarks (requires CUDA)
cd src/gpu_benchmarks/core
make
```

## Usage

### Bayesian Sports Ranking

```bash
# Run full season team ranking analysis
python scripts/run_bayesian_ranking.py

# Analyze single match outcomes
python scripts/run_match_analysis.py

# Compare inference methods
python scripts/run_inference_comparison.py
```

### GPU Performance Benchmarking

```bash
# Run comprehensive GPU benchmark suite
python scripts/run_gpu_benchmark.py

# Manual benchmark execution
cd src/gpu_benchmarks/core
make run
```

## Project Structure

```
distributed-analytics-toolkit/
├── src/                          # Source packages
│   ├── bayesian_ranking/         # Bayesian ranking implementation
│   │   ├── core/                 # Core inference algorithms
│   │   │   ├── gibbs_sampler.py  # MCMC sampling implementation
│   │   │   └── message_passing.py # Factor graph inference
│   │   ├── models/               # Statistical models
│   │   │   ├── team_ranker.py    # Season-long ranking system
│   │   │   └── match_analyzer.py # Single match analysis
│   │   └── utils/                # Utility functions
│   └── gpu_benchmarks/           # GPU performance suite
│       ├── core/                 # CUDA implementations
│       │   ├── memory_benchmark.cu # Memory bandwidth tests
│       │   └── Makefile          # Build configuration
│       ├── analysis/             # Performance analysis
│       │   └── performance_analyzer.py # Results visualization
│       └── utils/                # Cluster utilities
├── scripts/                      # Executable entry points
├── data/                         # Dataset files
├── outputs/                      # Generated results
│   ├── plots/                    # Visualization outputs
│   ├── results/                  # Raw benchmark data
│   └── benchmarks/               # Performance metrics
├── docs/                         # Documentation
└── requirements.txt              # Python dependencies
```

## How It Works

### Bayesian Inference Pipeline

The Bayesian ranking system models team strength as latent variables in a probabilistic framework. Using Gibbs sampling and message passing algorithms, the system infers posterior distributions over team abilities from match outcome data. The pipeline handles:

- **Prior specification** for team strength parameters
- **Likelihood modeling** of match outcomes given team strengths
- **Posterior inference** using MCMC and variational methods
- **Uncertainty quantification** through posterior sampling

### GPU Benchmarking Architecture

The GPU benchmarking suite implements optimized CUDA kernels for memory bandwidth and computational throughput testing. Key components include:

- **Stream triad operations** for memory bandwidth measurement
- **Matrix-matrix multiplication** for computational performance
- **Multi-GPU scaling** analysis for distributed computing
- **Performance profiling** across different problem sizes

## Configuration

### Bayesian Model Parameters

Modify inference parameters in the model files:
- **Sample count**: Adjust `n_draws` for MCMC chain length
- **Prior parameters**: Set team strength priors in `init_priors()`
- **Covariance modeling**: Configure team correlation structures

### GPU Benchmark Settings

Configure benchmark parameters in `src/gpu_benchmarks/core/Makefile`:
- **Architecture target**: Set `-arch=sm_XX` for your GPU
- **Optimization level**: Adjust `-O3` compiler flags
- **Memory patterns**: Modify kernel parameters

## Dataset

The project includes Serie A 2018/19 season data (`data/serie_a.csv`) containing match results for demonstration. The dataset provides:
- Team matchups and scores
- Seasonal performance patterns
- Ground truth for ranking validation

## Results

### Bayesian Ranking Outputs
- Team strength posterior distributions
- Match outcome predictions with uncertainty
- Seasonal ranking evolution visualizations
- Inference method comparison plots

### GPU Benchmark Outputs
- Memory bandwidth measurements (GB/s)
- Computational throughput (TFLOPS)
- Scaling analysis across problem sizes
- Performance comparison charts

## Troubleshooting

**Import Errors**: Ensure all dependencies are installed and PYTHONPATH includes the src directory
**CUDA Compilation**: Verify CUDA toolkit installation and GPU architecture compatibility
**Memory Issues**: Reduce sample counts or problem sizes for limited memory systems
**Performance**: Use optimized BLAS libraries and appropriate compiler flags

## License

MIT License