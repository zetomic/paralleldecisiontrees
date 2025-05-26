#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <iomanip>

// Serial implementation includes
#include "src/datasets.cpp"
#include "src/losses.cpp"
#include "src/metrics.cpp"
#include "src/tree_node.cpp"
#include "src/decision_tree.cpp"

struct BenchmarkResult {
    int max_depth;
    double train_time_ms;
    double train_accuracy;
    double test_accuracy;
    int tree_size;
    int tree_height;
};

void writeResultsToCSV(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    
    // Write header
    file << "version,max_depth,train_time_ms,train_accuracy,test_accuracy,tree_size,tree_height\n";
    
    // Write data
    for (const auto& r : results) {
        file << "serial,"
             << r.max_depth << ","
             << std::fixed << std::setprecision(4) << r.train_time_ms << ","
             << std::fixed << std::setprecision(4) << r.train_accuracy << ","
             << std::fixed << std::setprecision(4) << r.test_accuracy << ","
             << r.tree_size << ","
             << r.tree_height << "\n";
    }
    
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

BenchmarkResult benchmarkDecisionTree(DataFrame& train_data, DataFrame& test_data, int max_depth) {
    
    std::cout << "Testing SERIAL version with max_depth=" << max_depth << "..." << std::flush;
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Train decision tree
    DecisionTree tree(train_data, 
                     false,           // classification (not regression)
                     "gini_impurity", // loss function
                     -1,              // mtry (-1 = use all features)
                     max_depth,       // max_height
                     -1,              // max_leaves (no limit)
                     1,               // min_obs (minimum 1 observation per leaf)
                     -1,              // max_prop (no limit)
                     42);             // seed for reproducibility
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double train_time_ms = duration.count() / 1000.0;
    
    // Make predictions
    DataVector train_predictions = tree.predict(&train_data);
    DataVector test_predictions = tree.predict(&test_data);
    
    // Calculate accuracies
    double train_acc = accuracy(train_data.col(-1), train_predictions);
    double test_acc = accuracy(test_data.col(-1), test_predictions);
    
    std::cout << " Done! (" << std::fixed << std::setprecision(2) << train_time_ms << "ms)" << std::endl;
    
    return {max_depth, train_time_ms, train_acc, test_acc, 
            tree.getSize(), tree.getHeight()};
}

int main() {
    std::cout << "=== SERIAL Decision Tree Performance Benchmark ===" << std::endl;
    std::cout << "Loading cancer dataset..." << std::endl;
    
    // Load dataset
    DataLoader loader("data/cancer_clean.csv");
    DataFrame df = loader.load();
    
    std::cout << "Dataset loaded: " << df.length() << " rows, " << df.width() << " columns" << std::endl;
    
    // Create train/test split (80/20)
    std::vector<DataFrame> split_data = df.train_test_split(0.2, 42);
    DataFrame train_data = split_data[0];
    DataFrame test_data = split_data[1];
    
    std::cout << "Train set: " << train_data.length() << " rows" << std::endl;
    std::cout << "Test set: " << test_data.length() << " rows" << std::endl;
    std::cout << std::endl;
    
    // Define tree depths to test
    std::vector<int> depths = {1, 2, 3, 4, 5}; // -1 = no limit
    
    std::vector<BenchmarkResult> results;
    
    std::cout << "Running SERIAL version tests..." << std::endl;
    
    // Run benchmarks
    for (int depth : depths) {
        try {
            BenchmarkResult result = benchmarkDecisionTree(train_data, test_data, depth);
            results.push_back(result);
            
            // Print summary
            std::cout << "  Depth=" << (depth == -1 ? "unlimited" : std::to_string(depth))
                      << ", Time=" << std::fixed << std::setprecision(2) << result.train_time_ms << "ms"
                      << ", Train Acc=" << std::fixed << std::setprecision(3) << result.train_accuracy
                      << ", Test Acc=" << std::fixed << std::setprecision(3) << result.test_accuracy
                      << ", Tree Size=" << result.tree_size
                      << ", Tree Height=" << result.tree_height << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Error with depth " << depth << ": " << e.what() << std::endl;
        }
    }
    
    std::cout << std::endl;
    
    // Save results to CSV
    writeResultsToCSV(results, "benchmark_results_serial.csv");
    
    // Print summary statistics
    std::cout << "\n=== SERIAL Summary Statistics ===" << std::endl;
    double total_time = 0;
    double max_time = 0;
    double min_time = results.empty() ? 0 : results[0].train_time_ms;
    
    for (const auto& r : results) {
        total_time += r.train_time_ms;
        max_time = std::max(max_time, r.train_time_ms);
        min_time = std::min(min_time, r.train_time_ms);
    }
    
    std::cout << "Total tests: " << results.size() << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << "ms" << std::endl;
    std::cout << "Average time: " << std::fixed << std::setprecision(2) << total_time/results.size() << "ms" << std::endl;
    std::cout << "Min time: " << std::fixed << std::setprecision(2) << min_time << "ms" << std::endl;
    std::cout << "Max time: " << std::fixed << std::setprecision(2) << max_time << "ms" << std::endl;
    
    std::cout << "\nSERIAL benchmark completed! Results saved to benchmark_results_serial.csv" << std::endl;
    
    return 0;
}