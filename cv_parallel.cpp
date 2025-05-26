#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <iomanip>

// Include the PARALLEL modules (decision tree + CV)
#include "src-openmp/datasets.cpp"
#include "src-openmp/losses.cpp"
#include "src-openmp/metrics.cpp"
#include "src-openmp/tree_node.cpp"
#include "src-openmp/decision_tree.cpp"
#include "src-openmp/cv.cpp"  // Include the parallel CV module

void writeCVResultsToCSV(const std::vector<CVResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    
    // Write header - UPDATED with additional columns
    file << "version,dataset,max_depth,cv_time_ms,mean_cv_accuracy,std_cv_accuracy,fold1_acc,fold2_acc,fold3_acc,fold4_acc,warmup_runs,measurement_runs\n";
    
    // Write data
    for (const auto& r : results) {
        file << "parallel_cv,"
             << r.dataset << ","
             << r.max_depth << ","
             << std::fixed << std::setprecision(4) << r.cv_time_ms << ","
             << std::fixed << std::setprecision(4) << r.mean_cv_accuracy << ","
             << std::fixed << std::setprecision(4) << r.std_cv_accuracy;
        
        // Write individual fold scores
        for (size_t i = 0; i < r.fold_scores.size() && i < 4; i++) {
            file << "," << std::fixed << std::setprecision(4) << r.fold_scores[i];
        }
        
        // Fill remaining columns if less than 4 folds
        for (size_t i = r.fold_scores.size(); i < 4; i++) {
            file << ",";
        }
        
        // Add warmup and measurement run info
        file << ",1,1";  // 1 warmup, 1 measurement for CV
        
        file << "\n";
    }
    
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

std::vector<CVResult> testDatasetCV(const std::string& dataset_path, const std::string& dataset_name) {
    std::cout << "\n=== Testing " << dataset_name << " Dataset with Parallel Cross-Validation ===" << std::endl;
    
    // Load dataset
    DataLoader loader(dataset_path);
    DataFrame df = loader.load();
    
    std::cout << "Dataset loaded: " << df.length() << " rows, " << df.width() << " columns" << std::endl;
    
    // Create CrossValidator with 4 folds (perfect for 4 threads)
    CrossValidator cv(df, 4, 42, false);  // 4 folds, seed=42, classification
    
    // FIXED: Realistic tree depths to test (1 to 20)
    std::vector<int> depths = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20};
        
    std::vector<CVResult> results;
    
    // Run cross-validation benchmarks
    for (int depth : depths) {
        try {
            std::cout << "Testing PARALLEL CV with depth=" << depth << "..." << std::flush;
            
            // Warmup run (not timed)
            cv.validateDepth(depth, dataset_name);
            
            // Start timing for actual measurement
            auto start = std::chrono::high_resolution_clock::now();
            
            // Perform 4-fold cross-validation using PARALLEL CV (each fold on separate thread)
            CVResult cv_result = cv.validateDepth(depth, dataset_name);
            
            // End timing
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            cv_result.cv_time_ms = duration.count() / 1000.0;
            
            // Validate reasonable ranges
            if (cv_result.cv_time_ms < 1.0 || cv_result.cv_time_ms > 1200000) {  // 1ms to 20 minutes
                std::cout << " WARNING: Suspicious CV timing: " << cv_result.cv_time_ms << "ms" << std::endl;
            }
            
            // Check for reasonable accuracy range
            if (cv_result.mean_cv_accuracy < 0.3 || cv_result.mean_cv_accuracy > 1.0) {
                std::cout << " WARNING: Suspicious CV accuracy: " << cv_result.mean_cv_accuracy << std::endl;
            }
            
            std::cout << " Done! (" << std::fixed << std::setprecision(2) << cv_result.cv_time_ms << "ms)" << std::endl;
            
            results.push_back(cv_result);
            
            // Print summary
            std::cout << "  Depth=" << depth
                      << ", Time=" << std::fixed << std::setprecision(2) << cv_result.cv_time_ms << "ms"
                      << ", Mean CV Acc=" << std::fixed << std::setprecision(3) << cv_result.mean_cv_accuracy
                      << ", Std=" << std::fixed << std::setprecision(3) << cv_result.std_cv_accuracy
                      << ", Folds=[";
            for (size_t i = 0; i < cv_result.fold_scores.size(); i++) {
                std::cout << std::fixed << std::setprecision(3) << cv_result.fold_scores[i];
                if (i < cv_result.fold_scores.size() - 1) std::cout << ",";
            }
            std::cout << "]" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Error with depth " << depth << ": " << e.what() << std::endl;
        }
    }
    
    return results;
}

int main() {
    std::cout << "=== PARALLEL Cross-Validation Benchmark (Dual Dataset) ===" << std::endl;
    std::cout << "Using parallel fold execution with 4-fold cross-validation" << std::endl;
    std::cout << "Configuration: Parallel Tree + Parallel CV Folds" << std::endl;
    std::cout << "Testing realistic tree depths (1-20) with improved timing methodology" << std::endl;
    
    std::vector<CVResult> all_results;
    
    // Test Cancer dataset
    std::vector<CVResult> cancer_results = testDatasetCV("data/cancer_clean.csv", "cancer");
    all_results.insert(all_results.end(), cancer_results.begin(), cancer_results.end());
    
    // Test HMEQ dataset
    std::vector<CVResult> hmeq_results = testDatasetCV("data/hmeq_clean.csv", "hmeq");
    all_results.insert(all_results.end(), hmeq_results.begin(), hmeq_results.end());
    
    // Save combined results
    writeCVResultsToCSV(all_results, "cv_results_parallel.csv");
    
    // Print summary statistics
    std::cout << "\n=== PARALLEL Cross-Validation Overall Summary ===" << std::endl;
    std::cout << "Total CV tests: " << all_results.size() << std::endl;
    
    // Separate summaries by dataset
    for (const std::string& dataset : {"cancer", "hmeq"}) {
        std::vector<CVResult> dataset_results;
        for (const auto& r : all_results) {
            if (r.dataset == dataset) {
                dataset_results.push_back(r);
            }
        }
        
        if (!dataset_results.empty()) {
            double total_time = 0;
            double max_time = 0;
            double min_time = dataset_results[0].cv_time_ms;
            double best_cv_acc = 0;
            int best_depth = 0;
            
            for (const auto& r : dataset_results) {
                total_time += r.cv_time_ms;
                max_time = std::max(max_time, r.cv_time_ms);
                min_time = std::min(min_time, r.cv_time_ms);
                
                if (r.mean_cv_accuracy > best_cv_acc) {
                    best_cv_acc = r.mean_cv_accuracy;
                    best_depth = r.max_depth;
                }
            }
            
            std::cout << "\n" << dataset << " dataset:" << std::endl;
            std::cout << "  CV tests: " << dataset_results.size() << std::endl;
            std::cout << "  Total CV time: " << std::fixed << std::setprecision(2) << total_time << "ms" << std::endl;
            std::cout << "  Average CV time: " << std::fixed << std::setprecision(2) << total_time/dataset_results.size() << "ms" << std::endl;
            std::cout << "  Min CV time: " << std::fixed << std::setprecision(2) << min_time << "ms" << std::endl;
            std::cout << "  Max CV time: " << std::fixed << std::setprecision(2) << max_time << "ms" << std::endl;
            std::cout << "  Best CV accuracy: " << std::fixed << std::setprecision(3) << best_cv_acc 
                      << " (depth=" << best_depth << ")" << std::endl;
            
            // Data quality checks
            if (max_time / min_time > 1000) {
                std::cout << "  ⚠️  WARNING: Large CV time variance detected" << std::endl;
            }
        }
    }
    
    std::cout << "\nPARALLEL Cross-Validation benchmark completed!" << std::endl;
    std::cout << "Results saved to cv_results_parallel.csv" << std::endl;
    std::cout << "Expected speedup vs serial CV: 2x to 3.8x (approaching 4x)" << std::endl;
    std::cout << "Expected CV time ranges:" << std::endl;
    std::cout << "  Depth 1-5: 3-60ms (faster than serial)" << std::endl;
    std::cout << "  Depth 6-12: 15-600ms (faster than serial)" << std::endl;
    std::cout << "  Depth 15-20: 60-2500ms (faster than serial)" << std::endl;
    std::cout << "Compare with cv_results_serial.csv to see fold-parallelization speedup!" << std::endl;
    
    return 0;
}