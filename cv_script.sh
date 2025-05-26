#!/bin/bash

# Thread scaling benchmark script for Cross-Validation (4-core VM)
echo "=== Cross-Validation Thread Scaling Benchmark (Dual Dataset) ==="
echo "VM Configuration: 4 threads available"
echo "Testing: Serial Tree + Serial CV vs Serial Tree + Parallel CV"

# Compile optimized versions
echo "Compiling CV benchmarks with -O2 optimization..."
# Serial CV version (serial tree + serial CV)
g++ -std=c++14 -O2 cv_benchmark.cpp -o cv_benchmark_serial

# Parallel CV version (serial tree + parallel CV folds)
g++ -std=c++14 -O2 -fopenmp cv_parallel.cpp -o cv_benchmark_parallel

# Check if compilation was successful
if [ ! -f cv_benchmark_serial ] || [ ! -f cv_benchmark_parallel ]; then
    echo "ERROR: Compilation failed!"
    echo "Make sure cv_benchmark.cpp and cv_parallel.cpp exist in the current directory"
    echo "Also ensure src/cv.cpp and src/cv.hpp contain the serial CV implementation"
    exit 1
fi

echo "Compilation successful!"
echo ""

# Run serial CV version once
echo "=== Running SERIAL Cross-Validation benchmark (both datasets) ==="
echo "Configuration: Serial Tree + Serial CV Folds"
./cv_benchmark_serial

echo ""
echo "=== Testing PARALLEL Cross-Validation with different thread counts ==="
echo "Configuration: Serial Tree + Parallel CV Folds"
echo ""

# Test different thread counts (1, 2, 3, 4 for your 4-core VM)
for threads in 1 2 3 4; do
    echo "----------------------------------------"
    echo "Testing Parallel CV with $threads thread(s)"
    echo "----------------------------------------"
    
    export OMP_NUM_THREADS=$threads
    
    # Run parallel CV benchmark
    ./cv_benchmark_parallel
    
    # Rename output file to avoid overwriting
    if [ -f cv_results_parallel.csv ]; then
        mv cv_results_parallel.csv "cv_results_parallel_${threads}threads.csv"
        echo "Results saved to: cv_results_parallel_${threads}threads.csv"
    else
        echo "WARNING: No CV output file generated for $threads threads"
    fi
    
    echo ""
done

# Create a summary of all generated files
echo "========================================="
echo "=== CROSS-VALIDATION BENCHMARK COMPLETE ==="
echo "========================================="
echo ""
echo "Generated files:"
ls -la cv_results_*.csv 2>/dev/null | while read -r line; do
    echo "  $line"
done

echo ""
echo "File descriptions:"
echo "  cv_results_serial.csv                 - Serial CV results (both datasets)"
echo "  cv_results_parallel_1threads.csv      - Parallel CV with 1 thread (both datasets)"
echo "  cv_results_parallel_2threads.csv      - Parallel CV with 2 threads (both datasets)"
echo "  cv_results_parallel_3threads.csv      - Parallel CV with 3 threads (both datasets)"
echo "  cv_results_parallel_4threads.csv      - Parallel CV with 4 threads (both datasets)"
echo ""
echo "Each CSV contains cross-validation results for both cancer and hmeq datasets."
echo "Columns include: version, dataset, max_depth, cv_time_ms, mean_cv_accuracy, std_cv_accuracy, fold_scores"
echo ""
echo "Use these files to create performance comparison graphs!"
echo ""
echo "Suggested Cross-Validation Performance Graphs:"
echo "1. CV Time vs Tree Depth (separate lines for each thread count)"
echo "2. CV Speedup Ratio vs Tree Depth (serial_cv_time / parallel_cv_time)"
echo "3. Thread Scaling for CV (time vs number of threads for different depths)"
echo "4. Dataset Comparison (cancer vs hmeq CV performance)"
echo "5. CV Accuracy vs Tree Depth (to ensure parallel CV maintains same accuracy)"
echo "6. CV Time Distribution (boxplots of fold times for different thread counts)"
echo ""
echo "Performance Analysis Tips:"
echo "- Compare serial CV vs parallel CV with 1 thread to measure overhead"
echo "- Look for optimal thread count (may not always be 4 threads)"
echo "- Check if speedup varies by tree depth or dataset size"
echo "- Verify that parallel CV produces identical accuracies to serial CV"