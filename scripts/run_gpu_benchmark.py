#!/usr/bin/env python3
"""
Script for running GPU performance benchmarks and analysis.
"""

import sys
import os
import subprocess

def main():
    # Change to GPU benchmarks core directory
    core_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu_benchmarks', 'core')
    os.chdir(core_dir)
    
    print("Building GPU benchmark...")
    subprocess.run(['make'], check=True)
    
    print("Running GPU benchmark...")
    subprocess.run(['make', 'run-output'], check=True)
    
    print("Analyzing performance results...")
    analysis_dir = os.path.join('..', 'analysis')
    subprocess.run(['python', os.path.join(analysis_dir, 'performance_analyzer.py')], check=True)
    
    print("GPU benchmark complete! Check outputs/plots/ for results.")

if __name__ == "__main__":
    main()