#include "cv.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <omp.h>

CrossValidator::CrossValidator(DataFrame data, int k_folds, int seed, bool regression) 
    : data_(data), k_folds_(k_folds), random_seed_(seed), regression_(regression) {
    
    assert(k_folds > 1);
    assert(data.length() >= k_folds);
}

std::vector<std::vector<DataFrame>> CrossValidator::createKFolds(const DataFrame& data, int k, int seed) const {
    /**
     * Create k-fold splits of the data.
     * Returns vector of k pairs, each containing (train_data, validation_data)
     */
    
    // Shuffle the data first
    DataFrame shuffled_data = data.sample(-1, seed, false);  // Sample all rows without replacement
    
    int fold_size = shuffled_data.length() / k;
    int remainder = shuffled_data.length() % k;
    
    std::vector<std::vector<DataFrame>> folds;
    
    for (int fold = 0; fold < k; fold++) {
        // Calculate this fold's size (distribute remainder among first folds)
        int current_fold_size = fold_size + (fold < remainder ? 1 : 0);
        
        // Calculate start and end indices for validation set
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
    
    return folds;
}

std::pair<double, double> CrossValidator::calculateMeanStd(const std::vector<double>& scores) const {
    /**
     * Calculate mean and standard deviation of cross-validation scores.
     */
    if (scores.empty()) return {0.0, 0.0};
    
    // Calculate mean
    double sum = 0.0;
    for (double score : scores) {
        sum += score;
    }
    double mean = sum / scores.size();
    
    // Calculate standard deviation
    double variance_sum = 0.0;
    for (double score : scores) {
        double diff = score - mean;
        variance_sum += diff * diff;
    }
    double std_dev = std::sqrt(variance_sum / scores.size());
    
    return {mean, std_dev};
}

CVResult CrossValidator::validateSingleHyperparameter(const HyperparameterSet& params, const std::string& dataset_name) const {
    /**
     * Perform k-fold cross-validation for a single set of hyperparameters.
     * PARALLEL VERSION: Each fold runs on a separate thread
     */
    
    // Create k-fold splits
    std::vector<std::vector<DataFrame>> folds = createKFolds(data_, k_folds_, random_seed_);
    
    // Pre-allocate fold scores vector for thread safety
    std::vector<double> fold_scores(k_folds_, 0.0);
    
    // PARALLEL FOLDS: Each fold trains on a separate thread
    int fold;
    #pragma omp parallel for private(fold) shared(folds, fold_scores, params)
    for (fold = 0; fold < k_folds_; fold++) {
        DataFrame train_data = folds[fold][0];
        DataFrame val_data = folds[fold][1];
        
        // Train decision tree on training fold (uses SERIAL tree, but parallel tree construction if available)
        DecisionTree tree(train_data,
                         regression_,
                         params.loss,
                         -1,                    // mtry (use all features)
                         params.max_depth,     // max_height
                         -1,                    // max_leaves (no limit)
                         params.min_obs,       // min_obs
                         -1,                    // max_prop (no limit)
                         random_seed_ + fold); // Different seed for each fold
        
        // Make predictions on validation fold
        DataVector predictions = tree.predict(&val_data);
        DataVector true_labels = val_data.col(-1);
        
        // Calculate accuracy for this fold
        double fold_accuracy = accuracy(true_labels, predictions);
        fold_scores[fold] = fold_accuracy;
        
        // Optional: Print progress from master thread only
        #pragma omp master
        {
            static int completed_folds = 0;
            completed_folds++;
            if (completed_folds == 1) {
                std::cout << " [Parallel CV: " << omp_get_num_threads() << " threads]" << std::flush;
            }
        }
    }
    
    // Calculate mean and standard deviation
    auto mean_std = calculateMeanStd(fold_scores);
    
    return CVResult(dataset_name, params.max_depth, 0.0, mean_std.first, mean_std.second, fold_scores);
}

CVResult CrossValidator::validateDepth(int max_depth, const std::string& dataset_name) const {
    /**
     * Convenience method to validate a single depth with default parameters.
     * Uses parallel fold validation.
     */
    HyperparameterSet params(max_depth);
    return validateSingleHyperparameter(params, dataset_name);
}

std::vector<CVResult> CrossValidator::gridSearchCV(const std::vector<HyperparameterSet>& param_grid, const std::string& dataset_name) const {
    /**
     * Perform cross-validation for multiple hyperparameter combinations.
     * Each parameter set uses parallel folds.
     */
    std::vector<CVResult> results;
    
    for (const auto& params : param_grid) {
        CVResult result = validateSingleHyperparameter(params, dataset_name);
        results.push_back(result);
    }
    
    return results;
}

std::vector<CVResult> CrossValidator::validateDepths(const std::vector<int>& depths, const std::string& dataset_name) const {
    /**
     * Convenience method to validate only tree depths with default other parameters.
     * Each depth uses parallel fold validation.
     */
    std::vector<HyperparameterSet> param_grid;
    for (int depth : depths) {
        param_grid.push_back(HyperparameterSet(depth));
    }
    
    return gridSearchCV(param_grid, dataset_name);
}

HyperparameterSet CrossValidator::getBestParams(const std::vector<CVResult>& cv_results) const {
    /**
     * Find the hyperparameters that achieved the best cross-validation score.
     */
    if (cv_results.empty()) {
        return HyperparameterSet(); // Return default params
    }
    
    double best_score = cv_results[0].mean_cv_accuracy;
    int best_depth = cv_results[0].max_depth;
    
    for (const auto& result : cv_results) {
        if (result.mean_cv_accuracy > best_score) {
            best_score = result.mean_cv_accuracy;
            best_depth = result.max_depth;
        }
    }
    
    std::cout << "\nBest hyperparameters found:" << std::endl;
    std::cout << "  Max depth: " << best_depth << std::endl;
    std::cout << "  CV accuracy: " << std::fixed << std::setprecision(3) << best_score << std::endl;
    
    return HyperparameterSet(best_depth);
}