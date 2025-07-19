# CUDA Makefile

# Compiler
NVCC = nvcc

# Target executable
TARGET = memory_benchmark

# Source files
SOURCES = memory_benchmark.cu

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_60 -lcudart

# Output directory
OUTPUT_DIR = ../../outputs/results

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCES)

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Clean build artifacts
clean:
	rm -f $(TARGET)
	rm -rf $(OUTPUT_DIR)/*

# Run the program
run: $(TARGET)
	./$(TARGET)

# Run with output redirection
run-output: $(TARGET) $(OUTPUT_DIR)
	./$(TARGET) > $(OUTPUT_DIR)/results.txt

.PHONY: all clean run run-output