#ifndef CV_HPP
#define CV_HPP

#include "decision_tree.hpp"
#include "datasets.hpp"
#include "metrics.hpp"
#include <vector>
#include <string>
#include <utility>

struct CVResult {
    std::string dataset;
    int max_depth;
    double cv_time_ms;
    double mean_cv_accuracy;
    double std_cv_accuracy;
    std::vector<double> fold_scores;
    
    // Constructor for easy creation
    CVResult(const std::string& dataset_name, int depth, double time_ms, 
             double mean_acc, double std_acc, const std::vector<double>& scores)
        : dataset(dataset_name), max_depth(depth), cv_time_ms(time_ms), 
          mean_cv_accuracy(mean_acc), std_cv_accuracy(std_acc), fold_scores(scores) {}
    
    // Default constructor
    CVResult() : dataset(""), max_depth(0), cv_time_ms(0.0), 
                 mean_cv_accuracy(0.0), std_cv_accuracy(0.0) {}
};

struct HyperparameterSet {
    int max_depth;
    int min_obs;
    std::string loss;
    
    // Constructor for easy creation
    HyperparameterSet(int depth = 5, int min_obs_val = 1, std::string loss_func = "gini_impurity") 
        : max_depth(depth), min_obs(min_obs_val), loss(loss_func) {}
};

class CrossValidator {
private:
    DataFrame data_;
    int k_folds_;
    int random_seed_;
    bool regression_;
    
    // Helper function to create k-fold splits
    std::vector<std::vector<DataFrame>> createKFolds(const DataFrame& data, int k, int seed) const;
    
    // Helper function to calculate mean and standard deviation
    std::pair<double, double> calculateMeanStd(const std::vector<double>& scores) const;

public:
    // Constructor
    CrossValidator(DataFrame data, int k_folds = 4, int seed = 42, bool regression = false);
    
    // Single hyperparameter cross-validation
    CVResult validateSingleHyperparameter(const HyperparameterSet& params, const std::string& dataset_name = "") const;
    
    // Convenience method for depth-only validation
    CVResult validateDepth(int max_depth, const std::string& dataset_name = "") const;
    
    // Multiple hyperparameter grid search with cross-validation
    std::vector<CVResult> gridSearchCV(const std::vector<HyperparameterSet>& param_grid, const std::string& dataset_name = "") const;
    
    // Convenience method to validate multiple depths
    std::vector<CVResult> validateDepths(const std::vector<int>& depths, const std::string& dataset_name = "") const;
    
    // Get the best hyperparameters from CV results
    HyperparameterSet getBestParams(const std::vector<CVResult>& cv_results) const;
    
    // Getters
    int getKFolds() const { return k_folds_; }
    int getSeed() const { return random_seed_; }
    bool isRegression() const { return regression_; }
};

#endif