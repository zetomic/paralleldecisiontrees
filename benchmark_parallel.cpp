#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <iomanip>

// Parallel implementation includes
#include "src-openmp/datasets.cpp"
#include "src-openmp/losses.cpp"
#include "src-openmp/metrics.cpp"
#include "src-openmp/tree_node.cpp"
#include "src-openmp/decision_tree.cpp"

struct BenchmarkResult {
    std::string dataset;
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
    file << "version,dataset,max_depth,train_time_ms,train_accuracy,test_accuracy,tree_size,tree_height\n";
    
    // Write data
    for (const auto& r : results) {
        file << "parallel,"
             << r.dataset << ","
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
    
    // Define tree depths to test
    std::vector<int> depths = {1, 2, 3, 4, 5, 10, 15, 20, 50, 100, 200, 500};
    
    std::vector<BenchmarkResult> results;
    
    // Run benchmarks
    for (int depth : depths) {
        try {
            std::cout << "Testing PARALLEL with depth=" << depth << "..." << std::flush;
            
            // Start timing
            auto start = std::chrono::high_resolution_clock::now();
            
            // Train decision tree
            DecisionTree tree(train_data, 
                             false,           // classification (not regression)
                             "gini_impurity", // loss function
                             -1,              // mtry (-1 = use all features)
                             depth,           // max_height
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
            
            BenchmarkResult result = {dataset_name, depth, train_time_ms, train_acc, test_acc, 
                                    tree.getSize(), tree.getHeight()};
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
    std::cout << "=== PARALLEL Decision Tree Performance Benchmark (Dual Dataset) ===" << std::endl;
    
    std::vector<BenchmarkResult> all_results;
    
    // Test Cancer dataset
    std::vector<BenchmarkResult> cancer_results = testDataset("data/cancer_clean.csv", "cancer");
    all_results.insert(all_results.end(), cancer_results.begin(), cancer_results.end());
    
    // Test HMEQ dataset
    std::vector<BenchmarkResult> hmeq_results = testDataset("data/hmeq_clean.csv", "hmeq");
    all_results.insert(all_results.end(), hmeq_results.begin(), hmeq_results.end());
    
    // Save combined results
    writeResultsToCSV(all_results, "benchmark_results_parallel.csv");
    
    // Print summary statistics
    std::cout << "\n=== PARALLEL Overall Summary ===" << std::endl;
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
        }
    }
    
    std::cout << "\nPARALLEL benchmark completed! Results saved to benchmark_results_parallel.csv" << std::endl;
    
    return 0;
}