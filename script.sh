#!/bin/bash

# Updated Tree Training Benchmark Script
echo "=== Tree Training Thread Scaling Benchmark (Dual Dataset) ==="
echo "VM Configuration: 4 threads available"
echo "Testing realistic tree depths (1-20) with improved timing"

# Compile optimized versions
echo "Compiling tree benchmarks with -O2 optimization..."
g++ -std=c++14 -O2 benchmark_serial.cpp -o benchmark_serial
g++ -std=c++14 -O2 -fopenmp benchmark_parallel.cpp -o benchmark_parallel

# Check if compilation was successful
if [ ! -f benchmark_serial ] || [ ! -f benchmark_parallel ]; then
    echo "ERROR: Tree benchmark compilation failed!"
    exit 1
fi

echo "Tree benchmark compilation successful!"
echo ""

# Run serial version once
echo "=== Running SERIAL tree benchmark (both datasets) ==="
./benchmark_serial

echo ""
echo "=== Testing PARALLEL tree version with different thread counts ==="
echo ""

# Test different thread counts (1, 2, 3, 4 for your 4-core VM)
for threads in 1 2 3 4; do
    echo "----------------------------------------"
    echo "Testing tree training with $threads thread(s)"
    echo "----------------------------------------"
    
    export OMP_NUM_THREADS=$threads
    
    # Run parallel benchmark
    ./benchmark_parallel
    
    # Rename output file to avoid overwriting
    if [ -f benchmark_results_parallel.csv ]; then
        mv benchmark_results_parallel.csv "benchmark_results_parallel_${threads}threads.csv"
        echo "Results saved to: benchmark_results_parallel_${threads}threads.csv"
    else
        echo "WARNING: No tree output file generated for $threads threads"
    fi
    
    echo ""
done

echo "========================================="
echo "=== TREE TRAINING BENCHMARK COMPLETE ==="
echo "========================================="

---

#!/bin/bash

# Updated Cross-Validation Benchmark Script
echo "=== Cross-Validation Thread Scaling Benchmark (Dual Dataset) ==="
echo "VM Configuration: 4 threads available"
echo "Testing: Serial Tree + Serial CV vs Serial Tree + Parallel CV"
echo "Testing realistic tree depths (1-20) with improved timing"

# Compile optimized versions
echo "Compiling CV benchmarks with -O2 optimization..."
g++ -std=c++14 -O2 cv_benchmark.cpp -o cv_benchmark_serial
g++ -std=c++14 -O2 -fopenmp cv_parallel.cpp -o cv_benchmark_parallel

# Check if compilation was successful
if [ ! -f cv_benchmark_serial ] || [ ! -f cv_benchmark_parallel ]; then
    echo "ERROR: CV benchmark compilation failed!"
    echo "Make sure cv_benchmark.cpp and cv_parallel.cpp exist in the current directory"
    echo "Also ensure src/cv.cpp and src/cv.hpp contain the serial CV implementation"
    exit 1
fi

echo "CV benchmark compilation successful!"
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

echo "========================================="
echo "=== CROSS-VALIDATION BENCHMARK COMPLETE ==="
echo "========================================="

---

#!/bin/bash

# Complete Benchmark Script - Runs Both Tree Training and CV
echo "=== COMPLETE Performance Benchmark Suite ==="
echo "VM Configuration: 4 threads available"
echo "Testing: Tree Training + Cross-Validation Performance"
echo "Realistic tree depths (1-20) with improved timing methodology"

# Create results directory
mkdir -p results
echo "Results will be saved in ./results/ directory"

# Part 1: Tree Training Benchmarks
echo ""
echo "PART 1: TREE TRAINING PERFORMANCE"
echo "=================================="

# Compile tree benchmarks
echo "Compiling tree benchmarks..."
g++ -std=c++14 -O2 benchmark_serial.cpp -o benchmark_serial
g++ -std=c++14 -O2 -fopenmp benchmark_parallel.cpp -o benchmark_parallel

if [ ! -f benchmark_serial ] || [ ! -f benchmark_parallel ]; then
    echo "ERROR: Tree benchmark compilation failed!"
    exit 1
fi

# Run serial tree benchmark
echo "Running serial tree benchmark..."
./benchmark_serial
mv benchmark_results_serial.csv results/

# Run parallel tree benchmarks
for threads in 1 2 3 4; do
    echo "Running parallel tree benchmark with $threads threads..."
    export OMP_NUM_THREADS=$threads
    ./benchmark_parallel
    mv benchmark_results_parallel.csv "results/benchmark_results_parallel_${threads}threads.csv"
done

# Part 2: Cross-Validation Benchmarks
echo ""
echo "PART 2: CROSS-VALIDATION PERFORMANCE"
echo "====================================="

# Compile CV benchmarks
echo "Compiling CV benchmarks..."
g++ -std=c++14 -O2 cv_benchmark.cpp -o cv_benchmark_serial
g++ -std=c++14 -O2 -fopenmp cv_parallel.cpp -o cv_benchmark_parallel

if [ ! -f cv_benchmark_serial ] || [ ! -f cv_benchmark_parallel ]; then
    echo "ERROR: CV benchmark compilation failed!"
    exit 1
fi

# Run serial CV benchmark
echo "Running serial CV benchmark..."
./cv_benchmark_serial
mv cv_results_serial.csv results/

# Run parallel CV benchmarks
for threads in 1 2 3 4; do
    echo "Running parallel CV benchmark with $threads threads..."
    export OMP_NUM_THREADS=$threads
    ./cv_benchmark_parallel
    mv cv_results_parallel.csv "results/cv_results_parallel_${threads}threads.csv"
done

# Cleanup executables
rm -f benchmark_serial benchmark_parallel cv_benchmark_serial cv_benchmark_parallel

# Final summary
echo ""
echo "========================================="
echo "=== COMPLETE BENCHMARK SUITE FINISHED ==="
echo "========================================="
echo ""
echo "Generated files in ./results/:"
ls -la results/*.csv | while read -r line; do
    echo "  $line"
done

echo ""
echo "Tree Training Results:"
echo "  benchmark_results_serial.csv                 - Serial tree training"
echo "  benchmark_results_parallel_1threads.csv      - Parallel tree with 1 thread"
echo "  benchmark_results_parallel_2threads.csv      - Parallel tree with 2 threads"
echo "  benchmark_results_parallel_3threads.csv      - Parallel tree with 3 threads"
echo "  benchmark_results_parallel_4threads.csv      - Parallel tree with 4 threads"
echo ""
echo "Cross-Validation Results:"
echo "  cv_results_serial.csv                 - Serial CV"
echo "  cv_results_parallel_1threads.csv      - Parallel CV with 1 thread"
echo "  cv_results_parallel_2threads.csv      - Parallel CV with 2 threads"
echo "  cv_results_parallel_3threads.csv      - Parallel CV with 3 threads"
echo "  cv_results_parallel_4threads.csv      - Parallel CV with 4 threads"
echo ""
echo "Expected Results Summary:"
echo "🌳 Tree Training:"
echo "   - Depths 1-20 (realistic range)"
echo "   - Times: 1ms to 2000ms (reasonable range)"
echo "   - Speedup: 1.5x to 3.5x for deeper trees"
echo ""
echo "🔄 Cross-Validation:"
echo "   - Depths 1-20 (realistic range)" 
echo "   - Times: 10ms to 8000ms (4x tree training time)"
echo "   - Speedup: 2x to 3.8x (approaching 4x with 4 threads)"
echo ""
echo "📊 Analysis Tips:"
echo "   - Use Python notebook to generate performance graphs"
echo "   - Compare speedup patterns between tree training and CV"
echo "   - Look for optimal thread counts for your workload"
echo "   - Verify accuracy remains consistent across implementations"