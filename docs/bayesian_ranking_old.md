# Bayesian Team Ranking System

This module contains implementations of Bayesian ranking systems for sports teams, specifically analyzing the Serie A 2018/19 season using advanced probabilistic methods.
## Dependencies

Create a new virtual environment and install all the necessary Python packages:

```
python3 -m venv bayesian-ranking-env
source bayesian-ranking-env/bin/activate
pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Scripts

- `team-ranker.py` - Main script for full season team ranking analysis
- `single-match-analyzer.py` - Analysis of single match outcomes using Gibbs sampling
- `advanced-gibbs-sampler.py` - Advanced Gibbs sampling implementation with covariance
- `message-passing-inference.py` - Message passing inference algorithm
- `inference-comparison.py` - Comparison visualization between different inference methods
- `serie_a.csv` - Serie A 2018/19 season dataset

## Usage

Run the main team ranking analysis:
```bash
python team-ranker.py
```

Compare different inference methods:
```bash
python inference-comparison.py
```