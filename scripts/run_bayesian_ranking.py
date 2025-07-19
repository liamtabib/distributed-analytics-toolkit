#!/usr/bin/env python3
"""
Main script for running Bayesian team ranking analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bayesian_ranking.models.team_ranker import main

if __name__ == "__main__":
    main()