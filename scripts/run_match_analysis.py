#!/usr/bin/env python3
"""
Script for running single match analysis using Gibbs sampling.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bayesian_ranking.models.match_analyzer import main

if __name__ == "__main__":
    main()