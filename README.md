# SIGIR-2025-Submission-2117 🔍

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Query Performance Prediction Pipeline

This repository contains scripts for evaluating search queries using BM25 ranking and training Query Performance Prediction (QPP) models.

## Pipeline Overview

1. First, run the BM25 QPP evaluation script to generate retrieval metrics
2. Then, use these metrics to train the QPP machine learning models

## Step 1: BM25 QPP Evaluation
python BM25_QPP_Labels.py --input /path/to/your/dataset.pkl --k 10

## Arguments:
--input, -i:     Path to the input dataset pickle file (required) \
--k, -k:         Number of top documents to consider (default: 10) \
--debug, -d:     Enable debug output (optional) \
--output, -o:    Custom output path for results (optional, default: input_path_evaluation_k{k}.pkl) 

## Step 2: Training QPP Models
python run_machine_learning_models.py --dataset /path/to/evaluation_results.pkl --results-dir /path/to/output_directory 

## Arguments:
--dataset, -d:     Path to the evaluation dataset file (required) \
--results-dir, -r: Directory for saving results (required) \
--metric:          Metric to use ('ndcg@10' or 'mrr@10', default: 'ndcg@10') \
--eeg:             Include EEG features \
--text:            Include text features \
--model:           Model to use ('RandomForestRegressor' or 'LGBMRegressor', default: 'RandomForestRegressor') \
--loader:          Data loader to use (default: 'load_eeg_averaged_across_subjects') \
--evaluator:       Evaluator to use (default: 'group_kfold_evaluator') 

## Example Usage
# Step 1: Generate retrieval metrics
python BM25_QPP_Labels.py --input data/my_dataset.pkl --k 10 \

# Step 2: Train QPP model with both EEG and text features
python run_machine_learning_models.py \
    --dataset data/my_dataset_evaluation_k10.pkl \
    --results-dir results/ \
    --eeg \
    --text \
    --model LGBMRegressor

