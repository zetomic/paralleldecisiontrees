#!/bin/bash

# Clean Performance Benchmark Suite
echo "Performance Benchmark Suite - Starting..."

# Create results and logs directories
mkdir -p results logs

# Redirect verbose output to log file
exec 3>&1 4>&2
exec 1>logs/benchmark.log 2>&1

# Part 1: Tree Training Benchmarks
echo "Compiling tree benchmarks..." >&3
g++ -std=c++14 -O2 benchmark_serial.cpp -o benchmark_serial
g++ -std=c++14 -O2 -fopenmp benchmark_parallel.cpp -o benchmark_parallel

if [ ! -f benchmark_serial ] || [ ! -f benchmark_parallel ]; then
    echo "ERROR: Tree benchmark compilation failed!" >&3
    exit 1
fi

echo "✓ Tree benchmarks compiled" >&3

# Run serial tree benchmark
echo "Running tree training benchmarks..." >&3
./benchmark_serial
mv benchmark_results_serial.csv results/ 2>/dev/null

# Run parallel tree benchmarks
for threads in 1 2 3 4; do
    export OMP_NUM_THREADS=$threads
    ./benchmark_parallel
    mv benchmark_results_parallel.csv "results/benchmark_results_parallel_${threads}threads.csv" 2>/dev/null
done

echo "✓ Tree training benchmarks complete" >&3

# Part 2: Cross-Validation Benchmarks
echo "Compiling CV benchmarks..." >&3
g++ -std=c++14 -O2 cv_benchmark.cpp -o cv_benchmark_serial
g++ -std=c++14 -O2 -fopenmp cv_parallel.cpp -o cv_benchmark_parallel

if [ ! -f cv_benchmark_serial ] || [ ! -f cv_benchmark_parallel ]; then
    echo "ERROR: CV benchmark compilation failed!" >&3
    exit 1
fi

echo "✓ CV benchmarks compiled" >&3

# Run serial CV benchmark
echo "Running cross-validation benchmarks..." >&3
./cv_benchmark_serial
mv cv_results_serial.csv results/ 2>/dev/null

# Run parallel CV benchmarks
for threads in 1 2 3 4; do
    export OMP_NUM_THREADS=$threads
    ./cv_benchmark_parallel
    mv cv_results_parallel.csv "results/cv_results_parallel_${threads}threads.csv" 2>/dev/null
done

echo "✓ Cross-validation benchmarks complete" >&3

# Cleanup
rm -f benchmark_serial benchmark_parallel cv_benchmark_serial cv_benchmark_parallel

# Restore stdout/stderr
exec 1>&3 2>&4

# Display results summary
echo ""
echo "=== BENCHMARK RESULTS ==="
echo ""
echo "Generated files:"
for file in results/*.csv; do
    if [ -f "$file" ]; then
        echo "  $(basename "$file")"
    fi
done

echo ""
echo "Detailed output saved to: logs/benchmark.log"
echo ""
echo "Next steps:"
echo "  • Analyze results with: python analyze_results.py"
echo "  • View detailed logs: cat logs/benchmark.log"