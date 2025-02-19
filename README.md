# SIGIR-2025-Submission-2117
SIGIR-2025-Submission-2117

SIGIR-2025-Submission-2117 🔍
Show Image
Show Image
Query Performance Prediction Pipeline
This repository contains scripts for evaluating search queries using BM25 ranking and training Query Performance Prediction (QPP) models. The pipeline integrates both traditional text-based features and EEG signals for advanced query performance prediction.

📋 Pipeline Overview

BM25 QPP Evaluation: Generate retrieval metrics for queries
QPP Model Training: Train models using the generated metrics


🚀 Getting Started
Requirements
bashCopypip install -r requirements.txt
Step 1: BM25 QPP Evaluation
Generate retrieval metrics for your queries using the BM25 evaluation script:
bashCopypython BM25_QPP_Labels.py --input /path/to/your/dataset.pkl --k 10
Arguments
ArgumentDescriptionRequired--input, -iPath to input dataset pickle file✅--k, -kNumber of top documents to consider (default: 10)❌--debug, -dEnable debug output❌--output, -oCustom output path for results❌
Step 2: Training QPP Models
Train the QPP models using the generated retrieval metrics:
bashCopypython run_machine_learning_models.py --dataset /path/to/evaluation_results.pkl --results-dir /path/to/output_directory
Arguments
ArgumentDescriptionRequiredDefault--dataset, -dPath to evaluation dataset file✅---results-dir, -rDirectory for saving results✅---metricMetric to use ('ndcg@10' or 'mrr@10')❌ndcg@10--eegInclude EEG features❌False--textInclude text features❌False--modelModel type (RandomForestRegressor/LGBMRegressor)❌RandomForestRegressor--loaderData loader to use❌load_eeg_averaged_across_subjects--evaluatorEvaluator to use❌group_kfold_evaluator

📝 Example Usage
bashCopy# Step 1: Generate retrieval metrics
python BM25_QPP_Labels.py --input data/my_dataset.pkl --k 10

# Step 2: Train QPP model with both EEG and text features
python run_machine_learning_models.py \
    --dataset data/my_dataset_evaluation_k10.pkl \
    --results-dir results/ \
    --eeg \
    --text \
    --model LGBMRegressor
📊 Output

BM25 evaluation results will be saved as a pickle file with the suffix _evaluation_k{k}.pkl
ML model results will be saved in the specified results directory as JSON files


📬 Contact
For questions about the code, please open an issue in this repository.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
