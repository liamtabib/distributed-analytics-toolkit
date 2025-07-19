import numpy as np
import matplotlib.pyplot as plt
import os

max_size = 0

def init_plot(filename, plt, label):
    # Initialize empty lists to store data
    sizes = []
    bandwidth = []
    # Open the results file and parse data
    with open(filename, "r") as f:
        for line in f:
            # Filter out lines that contain the performance data
            if "STREAM triad of size" in line or 'Matrix-matrix' in line:
                parts = line.split()
                if 'x' in parts[4]:
                    sizes.append(int(parts[4].split('x')[1]))
                else:
                    sizes.append(int(parts[4]))

                bandwidth.append(float(parts[12])/1e6)

        # Convert lists to numpy arrays for better handling
        sizes = np.array(sizes)
        bandwidth = np.array(bandwidth)

        # Plotting
        plt.plot(sizes, bandwidth, 'o-', label=label) # Added label here

def main():
    """Main function to generate performance plots."""
    # Check if results files exist
    results_dir = "../../outputs/results"
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found. Please run benchmarks first.")
        return
    
    plt.figure(figsize=(5, 3))
    
    # Try to load data files, use defaults if not found
    data_files = [
        ('my_data', 'matrix-matrix GPU'),
        ('cublas_data', 'matrix-matrix cublas'),
        ('cpu_data', 'matrix-matrix cpu')
    ]
    
    for filename, label in data_files:
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            init_plot(filepath, plt, label)
        else:
            print(f"Warning: {filepath} not found, skipping {label}")
    
    plt.xlabel("Size")
    plt.ylabel("TFLOPS")
    plt.xscale('log')
    plt.yticks(np.arange(0,7,1))
    plt.title("TFLOPS vs. Size")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir = "../../outputs/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, "MM.png"), dpi=400)
    print(f"Performance plot saved to {output_dir}/MM.png")

if __name__ == "__main__":
    main()
