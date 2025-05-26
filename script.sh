#!/bin/bash

# Thread scaling benchmark script for 4-core VM
echo "=== Thread Scaling Benchmark (Dual Dataset) ==="
echo "VM Configuration: 4 threads available"

# Compile optimized versions
echo "Compiling benchmarks with -O2 optimization..."
# Both versions should use same optimization for fair comparison
g++ -std=c++14 -O2 benchmark_serial.cpp -o benchmark_serial
g++ -std=c++14 -O2 -fopenmp benchmark_parallel.cpp -o benchmark_parallel

# Check if compilation was successful
if [ ! -f benchmark_serial ] || [ ! -f benchmark_parallel ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

# Run serial version once
echo "=== Running SERIAL benchmark (both datasets) ==="
./benchmark_serial

echo ""
echo "=== Testing PARALLEL version with different thread counts ==="
echo ""

# Test different thread counts (1, 2, 3, 4 for your 4-core VM)
for threads in 1 2 3 4; do
    echo "----------------------------------------"
    echo "Testing with $threads thread(s)"
    echo "----------------------------------------"
    
    export OMP_NUM_THREADS=$threads
    
    # Run parallel benchmark
    ./benchmark_parallel
    
    # Rename output file to avoid overwriting
    if [ -f benchmark_results_parallel.csv ]; then
        mv benchmark_results_parallel.csv "benchmark_results_parallel_${threads}threads.csv"
        echo "Results saved to: benchmark_results_parallel_${threads}threads.csv"
    else
        echo "WARNING: No output file generated for $threads threads"
    fi
    
    echo ""
done

# Create a summary of all generated files
echo "========================================="
echo "=== BENCHMARK COMPLETE ==="
echo "========================================="
echo ""
echo "Generated files:"
ls -la benchmark_results_*.csv | while read -r line; do
    echo "  $line"
done

echo ""
echo "File descriptions:"
echo "  benchmark_results_serial.csv                 - Serial results (both datasets)"
echo "  benchmark_results_parallel_1threads.csv      - Parallel with 1 thread (both datasets)"
echo "  benchmark_results_parallel_2threads.csv      - Parallel with 2 threads (both datasets)"
echo "  benchmark_results_parallel_3threads.csv      - Parallel with 3 threads (both datasets)"
echo "  benchmark_results_parallel_4threads.csv      - Parallel with 4 threads (both datasets)"
echo ""
echo "Each CSV contains results for both cancer and hmeq datasets."
echo "Use these files to create performance comparison graphs!"
echo ""
echo "Suggested graphs to create:"
echo "1. Training Time vs Tree Depth (separate lines for each thread count)"
echo "2. Speedup Ratio vs Tree Depth (serial_time / parallel_time)"
echo "3. Thread Scaling (time vs number of threads for different depths)"
echo "4. Dataset Comparison (cancer vs hmeq performance)"