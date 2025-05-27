#!/bin/bash

# Performance Benchmark Suite
echo "=== Performance Benchmark Suite ==="
echo "VM Configuration: 4 threads available"
echo ""

# Create results and logs directories
mkdir -p results logs

# Part 1: Tree Training Benchmarks
echo "PART 1: TREE TRAINING BENCHMARKS"
echo "================================="

# Compile tree benchmarks
echo "Compiling tree benchmarks..."
g++ -std=c++14 -O2 benchmark_serial.cpp -o benchmark_serial 2>logs/compile.log
g++ -std=c++14 -O2 -fopenmp benchmark_parallel.cpp -o benchmark_parallel 2>>logs/compile.log

if [ ! -f benchmark_serial ] || [ ! -f benchmark_parallel ]; then
    echo "ERROR: Tree benchmark compilation failed!"
    cat logs/compile.log
    exit 1
fi

echo "✓ Tree benchmarks compiled successfully"
echo ""

# Run serial tree benchmark
echo "Running SERIAL tree benchmark..."
./benchmark_serial | tee logs/tree_serial.log
mv benchmark_results_serial.csv results/ 2>/dev/null
echo ""

# Run parallel tree benchmarks
echo "Running PARALLEL tree benchmarks..."
for threads in 2 4 6 8; do
    echo "  → Testing with $threads thread(s)..."
    export OMP_NUM_THREADS=$threads
    ./benchmark_parallel | tee logs/tree_parallel_${threads}t.log
    mv benchmark_results_parallel.csv "results/benchmark_results_parallel_${threads}threads.csv" 2>/dev/null
done

echo "✓ Tree training benchmarks complete"
echo ""

# Part 2: Cross-Validation Benchmarks
echo "PART 2: CROSS-VALIDATION BENCHMARKS"
echo "===================================="

# Compile CV benchmarks
echo "Compiling CV benchmarks..."
g++ -std=c++14 -O2 cv_benchmark.cpp -o cv_benchmark_serial 2>>logs/compile.log
g++ -std=c++14 -O2 -fopenmp cv_parallel.cpp -o cv_benchmark_parallel 2>>logs/compile.log

if [ ! -f cv_benchmark_serial ] || [ ! -f cv_benchmark_parallel ]; then
    echo "ERROR: CV benchmark compilation failed!"
    cat logs/compile.log
    exit 1
fi

echo "✓ CV benchmarks compiled successfully"
echo ""

# Run serial CV benchmark
echo "Running SERIAL cross-validation benchmark..."
./cv_benchmark_serial | tee logs/cv_serial.log
mv cv_results_serial.csv results/ 2>/dev/null
echo ""

# Run parallel CV benchmarks
echo "Running PARALLEL cross-validation benchmarks..."
for threads in 2 4 6 8; do
    echo "  → Testing with $threads thread(s)..."
    export OMP_NUM_THREADS=$threads
    ./cv_benchmark_parallel | tee logs/cv_parallel_${threads}t.log
    mv cv_results_parallel.csv "results/cv_results_parallel_${threads}threads.csv" 2>/dev/null
done

echo "✓ Cross-validation benchmarks complete"
echo ""

# Cleanup
rm -f benchmark_serial benchmark_parallel cv_benchmark_serial cv_benchmark_parallel

# Display results summary
echo "========================================="
echo "=== BENCHMARK SUITE COMPLETE ==="
echo "========================================="
echo ""
echo "Generated files in ./results/:"
for file in results/*.csv; do
    if [ -f "$file" ]; then
        echo "  $(basename "$file")"
    fi
done

echo ""
echo "Detailed logs saved in ./logs/:"
echo "  compile.log              - Compilation output"
echo "  tree_serial.log          - Serial tree training output"
echo "  tree_parallel_*t.log     - Parallel tree training output"
echo "  cv_serial.log            - Serial CV output"
echo "  cv_parallel_*t.log       - Parallel CV output"
echo ""
echo "Next steps:"
echo "  • Analyze results: python analyze_results.py"
echo "  • View specific logs: cat logs/[filename]"