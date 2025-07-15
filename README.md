# Course Projects

This repository contains coursework and implementations from two advanced computer science courses.

## Project Structure

```
course-projects/
├── gpu-benchmarks/           # GPU performance benchmarking
│   ├── README.md             # Build and usage instructions
│   ├── Makefile              # Build system
│   ├── memory-bandwidth-benchmark.cu  # Memory bandwidth benchmark
│   ├── performance-visualizer.py      # Performance visualization
│   ├── benchmark-job.sh              # SLURM job script
│   └── output/               # Generated results and plots
└── bayesian-ranking/         # Bayesian sports ranking
    ├── README.md             # Setup and usage guide
    ├── *.py                  # Python analysis scripts
    ├── serie_a.csv           # Serie A dataset
    ├── output/               # Generated plots and results
    └── requirements.txt      # Python dependencies
```

## Modules

### gpu-benchmarks
**Course**: Accelerator-based Programming  
**Focus**: GPU-parallelized code using CUDA programming language and SLURM job scheduling

- High-performance stream triad benchmarking
- CUDA kernel optimization
- Performance analysis and visualization
- Cluster computing with SLURM

### bayesian-ranking
**Course**: Advanced Probabilistic Machine Learning  
**Focus**: Bayesian statistical methods for sports team ranking using Serie A 2018/19 season data

- Gibbs sampling for Bayesian inference
- Message passing algorithms
- Probabilistic ranking systems
- Statistical visualization and analysis

## Getting Started

Each module contains its own README with specific setup instructions, dependencies, and usage examples. Navigate to the respective directories for specific information. 