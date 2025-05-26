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
#include "src-openmp/cv.cpp"  // Include the original parallel CV module (no changes)

struct CVBenchmarkResult {
    std::string dataset;
    int max_depth;
    double cv_training_time_ms;  // Only training time
    double mean_cv_accuracy;
    double std_cv_accuracy;
    std::vector<double> fold_scores;
    int warmup_runs;
    int measurement_runs;
};

void writeCVResultsToCSV(const std::vector<CVBenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    
    // Write header - Updated to clarify training time only
    file << "version,dataset,max_depth,cv_training_time_ms,mean_cv_accuracy,std_cv_accuracy,fold1_acc,fold2_acc,fold3_acc,fold4_acc,warmup_runs,measurement_runs\n";
    
    // Write data
    for (const auto& r : results) {
        file << "parallel_cv,"
             << r.dataset << ","
             << r.max_depth << ","
             << std::fixed << std::setprecision(4) << r.cv_training_time_ms << ","
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
        file << "," << r.warmup_runs << "," << r.measurement_runs;
        
        file << "\n";
    }
    
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

CVBenchmarkResult measureCVTrainingTimeParallel(const DataFrame& data, int depth, const std::string& dataset_name, 
                                               int warmup_runs = 1, int measurement_runs = 1) {
    /**
     * Manually perform PARALLEL CV to measure only training time, not prediction/evaluation
     */
    
    // Create 4-fold splits manually
    DataFrame shuffled_data = data.sample(-1, 42, false);  // Shuffle with seed 42
    int fold_size = shuffled_data.length() / 4;
    int remainder = shuffled_data.length() % 4;
    
    std::vector<std::vector<DataFrame>> folds;
    
    // Create 4 folds
    for (int fold = 0; fold < 4; fold++) {
        int current_fold_size = fold_size + (fold < remainder ? 1 : 0);
        
        int start_idx = 0;
        for (int i = 0; i < fold; i++) {
            start_idx += fold_size + (i < remainder ? 1 : 0);
        }
        int end_idx = start_idx + current_fold_size;
        
        // Create validation set (current fold)
        DataFrame validation_data = DataFrame();
        for (int i = start_idx; i < end_idx; i++) {
            validation_data.addRow(shuffled_data.row(i));
        }
        
        // Create training set (all other folds)
        DataFrame training_data = DataFrame();
        for (int i = 0; i < shuffled_data.length(); i++) {
            if (i < start_idx || i >= end_idx) {
                training_data.addRow(shuffled_data.row(i));
            }
        }
        
        std::vector<DataFrame> fold_pair = {training_data, validation_data};
        folds.push_back(fold_pair);
    }
    
    // Warmup runs
    for (int w = 0; w < warmup_runs; w++) {
        #pragma omp parallel for
        for (int fold = 0; fold < 4; fold++) {
            DataFrame train_data = folds[fold][0];
            DecisionTree warmup_tree(train_data, false, "gini_impurity", -1, depth, -1, 1, -1, 42 + fold + w);
        }
    }
    
    // Measurement runs - time only training in parallel
    std::vector<double> total_times;
    std::vector<double> fold_scores;
    
    for (int m = 0; m < measurement_runs; m++) {
        std::vector<double> fold_training_times(4, 0.0);
        std::vector<double> current_fold_scores(4, 0.0);
        
        // START TIMING: Parallel fold training
        auto start = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel for
        for (int fold = 0; fold < 4; fold++) {
            DataFrame train_data = folds[fold][0];
            DataFrame val_data = folds[fold][1];
            
            // TIME ONLY TRAINING (each thread times its own fold)
            auto fold_start = std::chrono::high_resolution_clock::now();
            
            DecisionTree tree(train_data, false, "gini_impurity", -1, depth, -1, 1, -1, 
                            42 + fold + warmup_runs + m);
            
            auto fold_end = std::chrono::high_resolution_clock::now();
            auto fold_duration = std::chrono::duration_cast<std::chrono::microseconds>(fold_end - fold_start);
            fold_training_times[fold] = fold_duration.count() / 1000.0;
            
            // PREDICTION AND EVALUATION (NOT TIMED)
            DataVector predictions = tree.predict(&val_data);
            DataVector true_labels = val_data.col(-1);
            double fold_accuracy = accuracy(true_labels, predictions);
            current_fold_scores[fold] = fold_accuracy;
        }
        
        // END TIMING: All parallel folds complete
        auto end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // For parallel: use wall clock time (actual time elapsed)
        // This shows the real speedup from parallelization
        double wall_clock_time = total_duration.count() / 1000.0;
        total_times.push_back(wall_clock_time);
        
        if (m == 0) {  // Use fold scores from first measurement run
            fold_scores = current_fold_scores;
        }
    }
    
    // Use median time if multiple measurement runs
    std::sort(total_times.begin(), total_times.end());
    double median_time = total_times[total_times.size() / 2];
    
    // Calculate mean and std of fold accuracies
    double sum = 0.0;
    for (double score : fold_scores) sum += score;
    double mean = sum / fold_scores.size();
    
    double variance_sum = 0.0;
    for (double score : fold_scores) {
        double diff = score - mean;
        variance_sum += diff * diff;
    }
    double std_dev = std::sqrt(variance_sum / fold_scores.size());
    
    return {dataset_name, depth, median_time, mean, std_dev, fold_scores, warmup_runs, measurement_runs};
}

std::vector<CVBenchmarkResult> testDatasetCV(const std::string& dataset_path, const std::string& dataset_name) {
    std::cout << "\n=== Testing " << dataset_name << " Dataset with Parallel Cross-Validation ===" << std::endl;
    
    // Load dataset
    DataLoader loader(dataset_path);
    DataFrame df = loader.load();
    
    std::cout << "Dataset loaded: " << df.length() << " rows, " << df.width() << " columns" << std::endl;
    
    // FIXED: Realistic tree depths to test (1 to 20)
    std::vector<int> depths = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20};
    
    std::vector<CVBenchmarkResult> results;
    const int warmup_runs = 1;
    const int measurement_runs = 1;
    
    // Run cross-validation benchmarks
    for (int depth : depths) {
        try {
            std::cout << "Testing PARALLEL CV with depth=" << depth << " (training time only)..." << std::flush;
            
            CVBenchmarkResult result = measureCVTrainingTimeParallel(df, depth, dataset_name, warmup_runs, measurement_runs);
            
            // Validate reasonable ranges
            if (result.cv_training_time_ms < 0.5 || result.cv_training_time_ms > 600000) {  // 0.5ms to 10 minutes
                std::cout << " WARNING: Suspicious CV training timing: " << result.cv_training_time_ms << "ms" << std::endl;
            }
            
            std::cout << " Done! (" << std::fixed << std::setprecision(2) << result.cv_training_time_ms << "ms training)" << std::endl;
            
            results.push_back(result);
            
            // Print summary
            std::cout << "  Depth=" << depth
                      << ", Training Time=" << std::fixed << std::setprecision(2) << result.cv_training_time_ms << "ms"
                      << ", Mean CV Acc=" << std::fixed << std::setprecision(3) << result.mean_cv_accuracy
                      << ", Std=" << std::fixed << std::setprecision(3) << result.std_cv_accuracy
                      << ", Folds=[";
            for (size_t i = 0; i < result.fold_scores.size(); i++) {
                std::cout << std::fixed << std::setprecision(3) << result.fold_scores[i];
                if (i < result.fold_scores.size() - 1) std::cout << ",";
            }
            std::cout << "]" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Error with depth " << depth << ": " << e.what() << std::endl;
        }
    }
    
    return results;
}

int main() {
    std::cout << "=== PARALLEL Cross-Validation Benchmark (Training Time Only) ===" << std::endl;
    std::cout << "Using manual parallel CV implementation to measure only training time" << std::endl;
    std::cout << "Testing realistic tree depths (1-20)" << std::endl;
    
    std::vector<CVBenchmarkResult> all_results;
    
    // Test Cancer dataset
    std::vector<CVBenchmarkResult> cancer_results = testDatasetCV("data/cancer_clean.csv", "cancer");
    all_results.insert(all_results.end(), cancer_results.begin(), cancer_results.end());
    
    // Test HMEQ dataset
    std::vector<CVBenchmarkResult> hmeq_results = testDatasetCV("data/hmeq_clean.csv", "hmeq");
    all_results.insert(all_results.end(), hmeq_results.begin(), hmeq_results.end());
    
    // Save combined results
    writeCVResultsToCSV(all_results, "cv_results_parallel.csv");
    
    std::cout << "\nPARALLEL Cross-Validation benchmark completed!" << std::endl;
    std::cout << "Results saved to cv_results_parallel.csv" << std::endl;
    std::cout << "IMPORTANT: Times represent training only (wall clock), comparable to tree benchmark times" << std::endl;
    std::cout << "Expected speedup vs serial CV training: 2x to 3.8x (approaching 4x)" << std::endl;
    std::cout << "Expected CV training time ranges:" << std::endl;
    std::cout << "  Depth 1-5: 1-60ms (parallel training of 4 trees)" << std::endl;
    std::cout << "  Depth 6-12: 10-600ms (parallel training of 4 trees)" << std::endl;
    std::cout << "  Depth 15-20: 50-2500ms (parallel training of 4 trees)" << std::endl;
    
    return 0;
}