import numpy as np
import matplotlib.pyplot as plt

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

plt.figure(figsize=(5, 3))
init_plot('my_data', plt, 'matrix-matrix GPU')
init_plot('cublas_data', plt, 'matrix-matrix cublas')
init_plot('cpu_data', plt, 'matrix-matrix cpu')
plt.xlabel("Size")
plt.ylabel("TFLOPS")
plt.xscale('log')
plt.yticks(np.arange(0,7,1))
plt.title("TFLOPS vs. Size")
plt.grid(True)
plt.legend() # Uncommented this line to show the legend

plt.tight_layout()
plt.savefig("output/MM.png",dpi=400)
