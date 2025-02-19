import argparse
import json
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from utilities import MachineLearningEvaluator
from transformers import BertTokenizer
from feature_loader import EEGDataLoader
import pickle
import numpy as np
import os

MODEL_MAPPING = {
    "RandomForestRegressor": RandomForestRegressor,
    "LGBMRegressor": LGBMRegressor,
}

def parse_args():
    parser = argparse.ArgumentParser(description='Run ML experiments')

    # Required arguments
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='Path to the evaluation dataset file')
    parser.add_argument('--results-dir', '-r', type=str, required=True,
                        help='Directory for results')

    # Optional arguments
    parser.add_argument('--metric', type=str, choices=['ndcg@10', 'mrr@10'],
                        default='ndcg@10', help='Metric to use')
    parser.add_argument('--eeg', action='store_true', help='Use EEG features')
    parser.add_argument('--text', action='store_true', help='Use text features')
    parser.add_argument('--model', type=str, choices=list(MODEL_MAPPING.keys()),
                        default='RandomForestRegressor', help='Model to use')
    parser.add_argument('--loader', type=str, default='load_eeg_averaged_across_subjects',
                        help='Data loader to use')
    parser.add_argument('--evaluator', type=str, default='group_kfold_evaluator',
                        help='Evaluator to use')

    args = parser.parse_args()

    args.dataset = os.path.expanduser(args.dataset.strip('"\''))
    args.results_dir = os.path.expanduser(args.results_dir.strip('"\''))

    return args

if __name__ == "__main__":
    args = parse_args()

    print(f"Loading scores from: {args.dataset}")
    with open(args.dataset, 'rb') as f:
        scores_loaded = pickle.load(f)

    # Generate output filename
    filename_parts = ['ml_regressor_results']
    filename_parts.append(args.model.lower().replace('regressor', ''))

    if args.text and args.eeg:
        filename_parts.append('eeg_text_combined')
    elif args.text:
        filename_parts.append('text_only')
    elif args.eeg:
        filename_parts.append('eeg_only')

    filename_parts.append(args.metric.replace('@', '_'))
    filename_parts.append(args.evaluator.replace('_evaluator', ''))
    filename_parts.append(args.loader.replace('load_', '').replace('_', '-'))

    results_file_path = os.path.join(args.results_dir, '_'.join(filename_parts) + '.json')

    # Initialize components
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    loader = EEGDataLoader(
        score_data=scores_loaded,
        select_eeg_features=args.eeg,
        select_deep_learning_model=False,
        selected_metric=args.metric,
        tokenizer=tokenizer
    )

    evaluator = MachineLearningEvaluator(
        select_eeg_features=args.eeg,
        select_text_features=args.text
    )

    # Load and prepare data
    data_and_targets = loader.choose_loader(args.loader)
    features = None
    targets = None
    groups = None

    if args.loader != 'load_eeg_per_subject':
        targets = data_and_targets['Targets']
        groups = data_and_targets['Groups']

        if args.eeg and args.text:
            text_features = data_and_targets['Queries']
            eeg_features = data_and_targets['EEG']
            print(f"EEG features shape: {eeg_features.shape}")
            features = np.concatenate((eeg_features, text_features), axis=1)
            print(f"Combined features shape: {features.shape}")
        elif args.text:
            features = data_and_targets['Queries']
        elif args.eeg:
            features = data_and_targets['EEG']
    else:
        features, targets = evaluator.prepare_data_for_LOO_evaluation(data_and_targets)

    # Get parameters and train model
    param_grid = evaluator.get_param_grid(args.model)
    model_config = evaluator.optimize_hyperparameters(
        features, targets,
        param_grid,
        args.evaluator,
        groups=groups,
        model_name=args.model
    )

    print(f"Best Parameters: {model_config['best_params']}")
    print(f"Best Score (MSE): {model_config['best_score']}")

    # Train final model with best parameters
    model_class = MODEL_MAPPING[args.model]
    best_model = model_class(**model_config['best_params'])

    results = evaluator.choose_machine_learning_evaluator(
        args.evaluator,
        best_model,
        features,
        targets,
        groups=groups
    )

    # Save results
    with open(results_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to: {results_file_path}")