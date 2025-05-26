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
    std::string dataset;
    int max_depth;
    double train_time_ms;
    double train_accuracy;
    double test_accuracy;
    int tree_size;
    int tree_height;
    int warmup_runs;
    int measurement_runs;
};

void writeResultsToCSV(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    
    // Write header
    file << "version,dataset,max_depth,train_time_ms,train_accuracy,test_accuracy,tree_size,tree_height,warmup_runs,measurement_runs\n";
    
    // Write data
    for (const auto& r : results) {
        file << "serial,"
             << r.dataset << ","
             << r.max_depth << ","
             << std::fixed << std::setprecision(4) << r.train_time_ms << ","
             << std::fixed << std::setprecision(4) << r.train_accuracy << ","
             << std::fixed << std::setprecision(4) << r.test_accuracy << ","
             << r.tree_size << ","
             << r.tree_height << ","
             << r.warmup_runs << ","
             << r.measurement_runs << "\n";
    }
    
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

double measureTrainingTime(const DataFrame& train_data, int depth, int warmup_runs = 2, int measurement_runs = 3) {
    /**
     * Measure training time with warmup runs to avoid cold start effects
     */
    
    // Warmup runs (not timed)
    for (int i = 0; i < warmup_runs; i++) {
        DecisionTree warmup_tree(train_data, false, "gini_impurity", -1, depth, -1, 1, -1, 42 + i);
    }
    
    // Measurement runs (timed)
    std::vector<double> times;
    for (int i = 0; i < measurement_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        DecisionTree tree(train_data, false, "gini_impurity", -1, depth, -1, 1, -1, 42 + warmup_runs + i);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        times.push_back(duration.count() / 1000.0);
    }
    
    // Return median time (more robust than mean)
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

std::vector<BenchmarkResult> testDataset(const std::string& dataset_path, const std::string& dataset_name) {
    std::cout << "\n=== Testing " << dataset_name << " Dataset ===" << std::endl;
    
    // Load dataset
    DataLoader loader(dataset_path);
    DataFrame df = loader.load();
    
    std::cout << "Dataset loaded: " << df.length() << " rows, " << df.width() << " columns" << std::endl;
    
    // Create train/test split (80/20)
    std::vector<DataFrame> split_data = df.train_test_split(0.2, 42);
    DataFrame train_data = split_data[0];
    DataFrame test_data = split_data[1];
    
    std::cout << "Train set: " << train_data.length() << " rows" << std::endl;
    std::cout << "Test set: " << test_data.length() << " rows" << std::endl;
    
    // FIXED: Realistic tree depths to test (1 to 20)
    std::vector<int> depths = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20};
    
    std::vector<BenchmarkResult> results;
    const int warmup_runs = 2;
    const int measurement_runs = 3;
    
    // Run benchmarks
    for (int depth : depths) {
        try {
            std::cout << "Testing SERIAL with depth=" << depth << "..." << std::flush;
            
            // Measure training time with warmup
            double train_time_ms = measureTrainingTime(train_data, depth, warmup_runs, measurement_runs);
            
            // Train final tree for accuracy measurement
            DecisionTree tree(train_data, false, "gini_impurity", -1, depth, -1, 1, -1, 42);
            
            // Make predictions
            DataVector train_predictions = tree.predict(&train_data);
            DataVector test_predictions = tree.predict(&test_data);
            
            // Calculate accuracies
            double train_acc = accuracy(train_data.col(-1), train_predictions);
            double test_acc = accuracy(test_data.col(-1), test_predictions);
            
            // Validate reasonable ranges
            if (train_time_ms < 0.01 || train_time_ms > 300000) {  // 0.01ms to 5 minutes
                std::cout << " WARNING: Suspicious timing: " << train_time_ms << "ms" << std::endl;
            }
            
            std::cout << " Done! (" << std::fixed << std::setprecision(2) << train_time_ms << "ms)" << std::endl;
            
            BenchmarkResult result = {dataset_name, depth, train_time_ms, train_acc, test_acc, 
                                    tree.getSize(), tree.getHeight(), warmup_runs, measurement_runs};
            results.push_back(result);
            
            // Print summary
            std::cout << "  Depth=" << depth
                      << ", Time=" << std::fixed << std::setprecision(2) << result.train_time_ms << "ms"
                      << ", Train Acc=" << std::fixed << std::setprecision(3) << result.train_accuracy
                      << ", Test Acc=" << std::fixed << std::setprecision(3) << result.test_accuracy
                      << ", Tree Size=" << result.tree_size
                      << ", Tree Height=" << result.tree_height << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Error with depth " << depth << ": " << e.what() << std::endl;
        }
    }
    
    return results;
}

int main() {
    std::cout << "=== SERIAL Decision Tree Performance Benchmark (Dual Dataset) ===" << std::endl;
    std::cout << "Testing realistic tree depths (1-20) with improved timing methodology" << std::endl;
    
    std::vector<BenchmarkResult> all_results;
    
    // Test Cancer dataset
    std::vector<BenchmarkResult> cancer_results = testDataset("data/cancer_clean.csv", "cancer");
    all_results.insert(all_results.end(), cancer_results.begin(), cancer_results.end());
    
    // Test HMEQ dataset
    std::vector<BenchmarkResult> hmeq_results = testDataset("data/hmeq_clean.csv", "hmeq");
    all_results.insert(all_results.end(), hmeq_results.begin(), hmeq_results.end());
    
    // Save combined results
    writeResultsToCSV(all_results, "benchmark_results_serial.csv");
    
    // Print summary statistics
    std::cout << "\n=== SERIAL Overall Summary ===" << std::endl;
    std::cout << "Total tests: " << all_results.size() << std::endl;
    
    // Separate summaries by dataset
    for (const std::string& dataset : {"cancer", "hmeq"}) {
        std::vector<BenchmarkResult> dataset_results;
        for (const auto& r : all_results) {
            if (r.dataset == dataset) {
                dataset_results.push_back(r);
            }
        }
        
        if (!dataset_results.empty()) {
            double total_time = 0;
            double max_time = 0;
            double min_time = dataset_results[0].train_time_ms;
            
            for (const auto& r : dataset_results) {
                total_time += r.train_time_ms;
                max_time = std::max(max_time, r.train_time_ms);
                min_time = std::min(min_time, r.train_time_ms);
            }
            
            std::cout << "\n" << dataset << " dataset:" << std::endl;
            std::cout << "  Tests: " << dataset_results.size() << std::endl;
            std::cout << "  Total time: " << std::fixed << std::setprecision(2) << total_time << "ms" << std::endl;
            std::cout << "  Average time: " << std::fixed << std::setprecision(2) << total_time/dataset_results.size() << "ms" << std::endl;
            std::cout << "  Min time: " << std::fixed << std::setprecision(2) << min_time << "ms" << std::endl;
            std::cout << "  Max time: " << std::fixed << std::setprecision(2) << max_time << "ms" << std::endl;
            
            // Data quality checks
            if (max_time / min_time > 1000) {
                std::cout << "  ⚠️  WARNING: Large time variance detected" << std::endl;
            }
        }
    }
    
    std::cout << "\nSERIAL benchmark completed! Results saved to benchmark_results_serial.csv" << std::endl;
    std::cout << "Expected time ranges:" << std::endl;
    std::cout << "  Depth 1-5: 1-50ms" << std::endl;
    std::cout << "  Depth 6-12: 10-500ms" << std::endl;
    std::cout << "  Depth 15-20: 50-2000ms" << std::endl;
    
    return 0;
}