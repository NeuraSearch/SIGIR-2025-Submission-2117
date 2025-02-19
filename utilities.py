import numpy as np
from sklearn.model_selection import GroupKFold
import tensorflow as tf
from deep_learning_models import SingleModalityRegressionModel, DualModalityRegressionModel
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from sklearn.ensemble import RandomForestRegressor
from itertools import product
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

# Dictionary mapping model names to their classes
MODEL_MAPPING = {
    "RandomForestRegressor": RandomForestRegressor,
    "SVR": SVR,
    "LGBMRegressor": LGBMRegressor
}
class DeepLearningEvaluator:
    def __init__(self, select_eeg_features, select_text_features, select_single_modality, n_splits=5, random_state=42,
                 epochs=10, batch_size=32, conv_filters=32, gru_units=32, dense_units=32,
                 projection_units=128, use_projections=False):
        self.select_eeg_features = select_eeg_features
        self.select_text_features = select_text_features
        self.select_single_modality = select_single_modality
        self.n_splits = n_splits
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.conv_filters = conv_filters
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.projection_units = projection_units
        self.use_projections = use_projections
    @staticmethod
    def combine_features_targets_dataset(features, targets, batch_size):
        """
        Combines features and targets into a tf.data.Dataset.

        Args:
            features (list or array-like): Input features to be used for training or evaluation.
            targets (list or array-like): Corresponding targets for the input features.
            batch_size (int): Size of batches for training.

        Returns:
            tf.data.Dataset: A dataset object that pairs features with targets.
        """
        # First convert to tensors if they aren't already
        features = tf.convert_to_tensor(features)
        targets = tf.convert_to_tensor(targets)

        # Create the datasets from tensor slices (unbatched)
        features_dataset = tf.data.Dataset.from_tensor_slices(features)
        targets_dataset = tf.data.Dataset.from_tensor_slices(targets)

        # Zip the datasets before batching
        dataset = tf.data.Dataset.zip((features_dataset, targets_dataset))

        # Then batch the combined dataset
        dataset = dataset.batch(batch_size)

        return dataset
    @staticmethod
    def create_multimodal_dataset(eeg_features, text_features, targets, batch_size):
        """
        Creates a dataset by padding text features to match EEG and combining them with targets.

        Args:
            eeg_features (list or array-like): EEG feature inputs.
            text_features (list or array-like): Text feature inputs.
            targets (list or array-like): Corresponding targets for the inputs.

        Returns:
            tf.data.Dataset: A zipped dataset of padded EEG, text features, and targets.
        """

        targets_dataset = tf.data.Dataset.from_tensor_slices(targets).batch(batch_size=batch_size)
        eeg_features_dataset = tf.data.Dataset.from_tensor_slices(eeg_features)
        text_features_dataset = tf.data.Dataset.from_tensor_slices(text_features)

        eeg_features_dataset = eeg_features_dataset.batch(batch_size=batch_size)
        text_features_dataset = text_features_dataset.batch(batch_size=batch_size)

        dataset = tf.data.Dataset.zip(({"eeg_input": eeg_features_dataset,
                                        "text_input": text_features_dataset}, targets_dataset))
        return dataset
    @staticmethod
    def calculate_qpp_metrics(y_true, y_pred):
        """
        Calculate comprehensive metrics for Query Performance Prediction evaluation.

        Args:
            y_true (array-like): True NDCG values
            y_pred (array-like): Predicted NDCG values

        Returns:
            dict: Dictionary containing all calculated metrics
        """
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Error metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        # Correlation metrics
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        kendall_corr, kendall_p = kendalltau(y_true, y_pred)

        # R-squared score
        r2 = r2_score(y_true, y_pred)

        # Binary classification metrics (good vs poor queries)
        threshold = 0.5  # Can be adjusted based on your needs
        y_true_binary = y_true >= threshold
        y_pred_binary = y_pred >= threshold

        accuracy = np.mean(y_true_binary == y_pred_binary)
        true_positives = np.sum((y_true_binary == True) & (y_pred_binary == True))
        false_positives = np.sum((y_true_binary == False) & (y_pred_binary == True))
        false_negatives = np.sum((y_true_binary == True) & (y_pred_binary == False))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'error_metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae)
            },
            'correlation_metrics': {
                'pearson': float(pearson_corr),
                'pearson_p': float(pearson_p),
                'spearman': float(spearman_corr),
                'spearman_p': float(spearman_p),
                'kendall': float(kendall_corr),
                'kendall_p': float(kendall_p)
            },
            'r2_score': float(r2),
            'classification_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
        }
    @staticmethod
    def dipslay_evaluation_results(overall_metrics):
        # Print comprehensive results
        print("\n=== QPP Evaluation Results ===")
        print("\nError Metrics:")
        print(f"MSE: {overall_metrics['error_metrics']['mse']:.4f}")
        print(f"RMSE: {overall_metrics['error_metrics']['rmse']:.4f}")
        print(f"MAE: {overall_metrics['error_metrics']['mae']:.4f}")

        print("\nCorrelation Metrics:")
        print(f"Pearson correlation: {overall_metrics['correlation_metrics']['pearson']:.4f}")
        print(f"Spearman correlation: {overall_metrics['correlation_metrics']['spearman']:.4f}")
        print(f"Kendall's tau: {overall_metrics['correlation_metrics']['kendall']:.4f}")

        print("\nR-squared Score:")
        print(f"R²: {overall_metrics['r2_score']:.4f}")

        print("\nBinary Classification Metrics (threshold=0.5):")
        print(f"Accuracy: {overall_metrics['classification_metrics']['accuracy']:.4f}")
        print(f"Precision: {overall_metrics['classification_metrics']['precision']:.4f}")
        print(f"Recall: {overall_metrics['classification_metrics']['recall']:.4f}")
        print(f"F1-score: {overall_metrics['classification_metrics']['f1']:.4f}")

    def optimize_parameters(self, features, targets, evaluator_to_use, trials=50, input_shape=None, groups=None):
        """
        Optimizes the hyperparameters for the deep learning model using Optuna.

        Parameters:
        - features (numpy.ndarray or tuple): The input features.
        - targets (numpy.ndarray): The target values.
        - evaluator_to_use (str): Type of evaluator to optimize. Options: "group_kfold_evaluator".
        - input_shape (tuple): Shape of the input features (optional).
        - groups (numpy.ndarray): Group labels for GroupKFold (required for GroupKFold).

        Returns:
        - dict: The best hyperparameters.
        """
        print('--Parameter optimization started--')
        def objective(trial):
            # hyperparameter search space
            conv_filters = trial.suggest_categorical("conv_filters", [16, 32, 64, 128])
            gru_units = trial.suggest_categorical("gru_units", [16, 32, 64, 128])
            dense_units = trial.suggest_categorical("dense_units", [16, 32, 64, 128])
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            projection_units = trial.suggest_categorical("projection_units",
                                                         [64, 128, 256]) if self.use_projections else None

            best_params = {
                "conv_filters": conv_filters,
                "gru_units": gru_units,
                "dense_units": dense_units,
                "batch_size": batch_size,
            }
            if projection_units is not None:
                best_params["projection_units"] = projection_units

            if evaluator_to_use == "group_kfold_evaluator":
                results = self.group_kfold_evaluator(features, targets, groups, input_shape, display_evaluation_results=False, **best_params)

            elif evaluator_to_use == "leave_one_out_evaluator":
                results = self.leave_one_out_evaluator(features, targets, input_shape, display_evaluation_results=False, **best_params)

            else:
                raise ValueError(f"Unsupported evaluator: {evaluator_to_use}")

            # Handle missing keys safely
            return results.get('overall_metrics', {}).get('error_metrics', {}).get('mse', float('inf'))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=trials)

        print("Best hyperparameters:", study.best_params)
        print("Best MSE:", study.best_value)

        return study.best_params

    def group_kfold_evaluator(self, features, targets, groups, input_shape, display_evaluation_results, **best_params):
        gkf = GroupKFold(n_splits=self.n_splits)

        fold_metrics = []
        all_predictions = []
        all_true_values = []
        if display_evaluation_results:
            print('--Model Training Started--')

        if self.select_single_modality:
            indexes_for_splitting = features
        else:
            indexes_for_splitting = features[0]

        conv_filters = best_params.get("conv_filters", self.conv_filters)
        gru_units = best_params.get("gru_units", self.gru_units)
        dense_units = best_params.get("dense_units", self.dense_units)
        batch_size = best_params.get("batch_size", self.batch_size)
        projection_units = best_params.get("projection_units", self.projection_units)

        for fold_idx, (train_index, test_index) in enumerate(gkf.split(indexes_for_splitting, targets, groups=groups)):
            if self.select_single_modality:
                X_train, X_test = features[train_index], features[test_index]
                y_train, y_test = targets[train_index], targets[test_index]

                train_dataset = self.combine_features_targets_dataset(X_train, y_train, batch_size)
                test_dataset = self.combine_features_targets_dataset(X_test, y_test, batch_size)

                model = SingleModalityRegressionModel(
                    input_shape, conv_filters, gru_units,
                    dense_units, self.select_eeg_features
                ).build_model()

            elif not self.select_single_modality:
                y_train, y_test = targets[train_index], targets[test_index]
                eeg_features, text_features = features[0], features[1]
                X_train_eeg, X_test_eeg = eeg_features[train_index], eeg_features[test_index]
                X_train_text, X_test_text = text_features[train_index], text_features[test_index]

                train_dataset = self.create_multimodal_dataset(X_train_eeg, X_train_text, y_train, batch_size)
                test_dataset = self.create_multimodal_dataset(X_test_eeg, X_test_text, y_test, batch_size)

                eeg_input_shape, text_input_shape = input_shape[0], input_shape[1]

                model = DualModalityRegressionModel(
                  eeg_input_shape, text_input_shape,
                  conv_filters, gru_units, dense_units,
                  projection_units
                ).build_model(use_projections=self.use_projections)

            model.compile(optimizer='adam', loss='mse')
            model.fit(train_dataset, epochs=self.epochs, batch_size=batch_size, verbose=0)
            y_pred = model.predict(test_dataset).reshape(-1)

            # Calculate metrics for this fold
            fold_metrics.append(self.calculate_qpp_metrics(y_test, y_pred))

            # Store predictions and true values
            all_predictions.extend(y_pred)
            all_true_values.extend(y_test)

            if display_evaluation_results:
                # Print sample predictions
                print(f"\nFold {fold_idx + 1} Sample Predictions:")
                for i in range(min(5, len(y_test))):
                    print(f"Query index: {test_index[i]}")
                    print(f"True NDCG: {y_test[i]:.3f}, Predicted NDCG: {y_pred[i]:.3f}")

            # Calculate average metrics across folds
        avg_metrics = {
            'error_metrics': {},
            'correlation_metrics': {},
            'classification_metrics': {}
        }

        metric_categories = ['error_metrics', 'correlation_metrics', 'classification_metrics']
        for category in metric_categories:
            for metric in fold_metrics[0][category]:
                values = [fold[category][metric] for fold in fold_metrics]
                avg_metrics[category][metric] = float(np.mean(values))
                avg_metrics[category][f'{metric}_std'] = float(np.std(values))

        # Calculate overall metrics using all predictions
        overall_metrics = self.calculate_qpp_metrics(all_true_values, all_predictions)

        results = {
            'average_fold_metrics': avg_metrics,
            'overall_metrics': overall_metrics,
            'fold_metrics': fold_metrics
        }
        if display_evaluation_results:
            # Print comprehensive results
            self.dipslay_evaluation_results(overall_metrics)

        return results

    import numpy as np

    @staticmethod
    def prepare_data_for_LOO_evaluation(data_and_targets, select_eeg_features, select_text_features):
        features_dict = {'Train': {}, 'Test': {}}
        targets_dict = {'Train': {}, 'Test': {}}
        subject_ids = list(data_and_targets.keys())
        input_shape = None

        def pad_array(arr, target_length, axis=0):
            """Helper function to pad arrays to the same length"""
            pad_size = target_length - arr.shape[axis]
            if pad_size <= 0:
                return arr
            npad = [(0, 0)] * arr.ndim
            npad[axis] = (0, pad_size)
            return np.pad(arr, pad_width=npad, mode='constant', constant_values=0)

        for subject in subject_ids:
            # Test set setup remains the same
            if select_eeg_features:
                test_subject_eeg_data = np.asarray(data_and_targets[subject]['EEG'])
                test_subject_eeg_data = np.transpose(test_subject_eeg_data, (0, 2, 3, 1))

            test_subject_queries = np.asarray(data_and_targets[subject]['Queries'])
            test_subject_targets = np.asarray(data_and_targets[subject]['Targets'])

            # Set up test dict
            if select_eeg_features and select_text_features:
                features_dict['Test'][subject] = {
                    'EEG': test_subject_eeg_data,
                    'Queries': test_subject_queries
                }
            elif select_text_features:
                features_dict['Test'][subject] = {'Queries': test_subject_queries}
            elif select_eeg_features:
                features_dict['Test'][subject] = {'EEG': test_subject_eeg_data}

            targets_dict['Test'][subject] = {'Targets': test_subject_targets}

            # Training set (all other subjects)
            training_subjects = [s for s in subject_ids if s != subject]

            if select_text_features:
                # Get all training queries and find max length
                training_subjects_queries = [np.asarray(data_and_targets[s]['Queries']) for s in training_subjects]
                max_query_length = max(arr.shape[1] for arr in training_subjects_queries)
                # Pad all arrays to max length
                training_subjects_queries = [pad_array(arr, max_query_length, axis=1) for arr in
                                             training_subjects_queries]

            if select_eeg_features:
                training_subjects_eeg_features = [data_and_targets[s]['EEG'] for s in training_subjects]

            training_subjects_targets = [data_and_targets[s]['Targets'] for s in training_subjects]

            # Set up training dict
            if select_eeg_features and select_text_features:
                concatenated_eeg_features = np.concatenate(training_subjects_eeg_features, axis=0)
                features_dict['Train'][subject] = {
                    'EEG': np.transpose(concatenated_eeg_features, (0, 2, 3, 1)),
                    'Queries': np.concatenate(training_subjects_queries, axis=0)
                }
            elif select_text_features:
                features_dict['Train'][subject] = {
                    'Queries': np.concatenate(training_subjects_queries, axis=0)
                }
            elif select_eeg_features:
                concatenated_eeg_features = np.concatenate(training_subjects_eeg_features, axis=0)
                features_dict['Train'][subject] = {
                    'EEG': np.transpose(concatenated_eeg_features, (0, 2, 3, 1))
                }

            targets_dict['Train'][subject] = {
                'Targets': np.concatenate(training_subjects_targets, axis=0)
            }

        # Rest of your input_shape logic remains the same
        if select_eeg_features and select_text_features:
            input_shape = ((features_dict['Train'][subject_ids[0]]['EEG'].shape[1],
                            features_dict['Train'][subject_ids[0]]['EEG'].shape[2], 1),
                           (None, 1))
        elif select_text_features:
            input_shape = (None, 1)
        elif select_eeg_features:
            input_shape = (features_dict['Train'][subject_ids[0]]['EEG'].shape[1],
                           features_dict['Train'][subject_ids[0]]['EEG'].shape[2], 1)

        return features_dict, targets_dict, input_shape

    def leave_one_out_evaluator(self, features, targets, input_shape, display_evaluation_results, **best_params):

        fold_metrics = []
        all_predictions = []
        all_true_values = []
        if display_evaluation_results:
            print('--Model Training Started--')

        conv_filters = best_params.get("conv_filters", self.conv_filters)
        gru_units = best_params.get("gru_units", self.gru_units)
        dense_units = best_params.get("dense_units", self.dense_units)
        batch_size = best_params.get("batch_size", self.batch_size)
        projection_units = best_params.get("projection_units", self.projection_units)

        feature_training_data, feature_test_data = features['Train'], features['Test']
        target_training_data, target_test_data = targets['Train'], targets['Test']

        for subject, data in feature_training_data.items():
            y_train, y_test = target_training_data[subject]['Targets'], target_test_data[subject]["Targets"]

            if self.select_single_modality:
                if self.select_text_features:
                    X_train, X_test = data['Queries'], feature_test_data[subject]['Queries']

                elif self.select_eeg_features:
                    X_train, X_test = data['EEG'], feature_test_data[subject]['EEG']

                train_dataset = self.combine_features_targets_dataset(X_train, y_train, batch_size)
                test_dataset = self.combine_features_targets_dataset(X_test, y_test, batch_size)

                model = SingleModalityRegressionModel(
                    input_shape, conv_filters, gru_units,
                    dense_units, self.select_eeg_features
                ).build_model()

            elif not self.select_single_modality:
                X_train_eeg, X_test_eeg = data['EEG'], feature_test_data[subject]['EEG']
                X_train_text, X_test_text = data['Queries'], feature_test_data[subject]['Queries']

                train_dataset = self.create_multimodal_dataset(X_train_eeg, X_train_text, y_train, batch_size)
                test_dataset = self.create_multimodal_dataset(X_test_eeg, X_test_text, y_test, batch_size)

                eeg_input_shape, text_input_shape = input_shape[0], input_shape[1]

                model = DualModalityRegressionModel(
                  eeg_input_shape, text_input_shape,
                  conv_filters, gru_units, dense_units,
                  projection_units
                ).build_model(use_projections=self.use_projections)

            model.compile(optimizer='adam', loss='mse')
            model.fit(train_dataset, epochs=self.epochs, batch_size=batch_size, verbose=0)
            y_pred = model.predict(test_dataset).reshape(-1)

            # Calculate metrics for this fold
            fold_metrics.append(self.calculate_qpp_metrics(y_test, y_pred))

            # Store predictions and true values
            all_predictions.extend(y_pred)
            all_true_values.extend(y_test)

            if display_evaluation_results:
                # Print sample predictions
                print(f"\nSubject {subject} Sample Predictions:")
                for i in range(min(5, len(y_test))):
                    print(f"Test data index: {i}")
                    print(f"True Score: {y_test[i]:.3f}, Predicted Score: {y_pred[i]:.3f}")

            # Calculate average metrics across folds
        avg_metrics = {
            'error_metrics': {},
            'correlation_metrics': {},
            'classification_metrics': {}
        }

        metric_categories = ['error_metrics', 'correlation_metrics', 'classification_metrics']
        for category in metric_categories:
            for metric in fold_metrics[0][category]:
                values = [fold[category][metric] for fold in fold_metrics]
                avg_metrics[category][metric] = float(np.mean(values))
                avg_metrics[category][f'{metric}_std'] = float(np.std(values))

        # Calculate overall metrics using all predictions
        overall_metrics = self.calculate_qpp_metrics(all_true_values, all_predictions)

        results = {
            'average_fold_metrics': avg_metrics,
            'overall_metrics': overall_metrics,
            'fold_metrics': fold_metrics
        }
        if display_evaluation_results:
            # Print comprehensive results
            self.dipslay_evaluation_results(overall_metrics)

        return results

    def choose_deep_learning_evaluator(self, name: str, *args, **kwargs):
        if hasattr(self, name) and callable(func := getattr(self, name)):
            return func(*args, **kwargs)


class MachineLearningEvaluator:
    def __init__(self, select_eeg_features, select_text_features, n_splits=5, random_state=42):
        self.random_state = random_state
        self.n_splits = n_splits
        self.select_eeg_features = select_eeg_features,
        self.select_text_features = select_text_features

    @staticmethod
    def calculate_qpp_metrics(y_true, y_pred):
        """
        Calculate comprehensive metrics for Query Performance Prediction evaluation.

        Args:
            y_true (array-like): True NDCG values
            y_pred (array-like): Predicted NDCG values

        Returns:
            dict: Dictionary containing all calculated metrics
        """
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Error metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        # Correlation metrics
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        kendall_corr, kendall_p = kendalltau(y_true, y_pred)

        # R-squared score
        r2 = r2_score(y_true, y_pred)

        # Binary classification metrics (good vs poor queries)
        threshold = 0.5  # Can be adjusted based on your needs
        y_true_binary = y_true >= threshold
        y_pred_binary = y_pred >= threshold

        accuracy = np.mean(y_true_binary == y_pred_binary)
        true_positives = np.sum((y_true_binary == True) & (y_pred_binary == True))
        false_positives = np.sum((y_true_binary == False) & (y_pred_binary == True))
        false_negatives = np.sum((y_true_binary == True) & (y_pred_binary == False))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'error_metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae)
            },
            'correlation_metrics': {
                'pearson': float(pearson_corr),
                'pearson_p': float(pearson_p),
                'spearman': float(spearman_corr),
                'spearman_p': float(spearman_p),
                'kendall': float(kendall_corr),
                'kendall_p': float(kendall_p)
            },
            'r2_score': float(r2),
            'classification_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
        }

    @staticmethod
    def dipslay_evaluation_results(overall_metrics):
        # Print comprehensive results
        print("\n=== QPP Evaluation Results ===")
        print("\nError Metrics:")
        print(f"MSE: {overall_metrics['error_metrics']['mse']:.4f}")
        print(f"RMSE: {overall_metrics['error_metrics']['rmse']:.4f}")
        print(f"MAE: {overall_metrics['error_metrics']['mae']:.4f}")

        print("\nCorrelation Metrics:")
        print(f"Pearson correlation: {overall_metrics['correlation_metrics']['pearson']:.4f}")
        print(f"Spearman correlation: {overall_metrics['correlation_metrics']['spearman']:.4f}")
        print(f"Kendall's tau: {overall_metrics['correlation_metrics']['kendall']:.4f}")

        print("\nR-squared Score:")
        print(f"R²: {overall_metrics['r2_score']:.4f}")

        print("\nBinary Classification Metrics (threshold=0.5):")
        print(f"Accuracy: {overall_metrics['classification_metrics']['accuracy']:.4f}")
        print(f"Precision: {overall_metrics['classification_metrics']['precision']:.4f}")
        print(f"Recall: {overall_metrics['classification_metrics']['recall']:.4f}")
        print(f"F1-score: {overall_metrics['classification_metrics']['f1']:.4f}")
    @staticmethod
    def get_param_grid(model_name):
        param_grid = None
        if model_name == "RandomForestRegressor":
            param_grid = {
                'n_estimators': [50, 200, 300],  # Add more options
                'max_depth': [3, 7, 10],  # Try smaller depths
                'min_samples_split': [5, 15, 20],  # Increase these values
                'min_samples_leaf': [4, 12, 16],  # Increase these values
                'max_features': ['sqrt', 'log2', None],  # Add feature selection
                'bootstrap': [True],  # Enable bootstrapping
                'max_samples': [0.7, 0.8, 0.9]  # Use subset of samples for trees
            }
        elif model_name == "SVR":
            param_grid = {
                'C': [0.1, 10, 20],
                'kernel': ['poly', 'rbf', 'sigmoid'],
                'epsilon': [0.01, 0.1, 0.5]
            }

        elif model_name == "LGBMRegressor":
            param_grid = {
                'num_leaves': [2, 5, 10],
                'max_depth': [2, 5, 10],
                'learning_rate': [0.001, 0.01, 0.1],
                'n_estimators': [50, 100, 200],
            }

        return param_grid

    def optimize_hyperparameters(self, features, targets, param_grid, evaluator_to_use, groups,
                                 model_name="RandomForestRegressor"):
        gkf = GroupKFold(n_splits=self.n_splits)

        # Create a custom scorer that accepts groups
        def custom_cv_scorer(estimator, X, y):
            cv_scores = []
            for train_idx, test_idx in gkf.split(X, y, groups=groups):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
                cv_scores.append(-mean_squared_error(y_test, y_pred))  # Negative because GridSearchCV maximizes
            return np.mean(cv_scores)

        def custom_loo_scorer(estimator, X, y):
            loo_scores = []
            feature_training_data, feature_test_data = X['Train'], X['Test']
            target_training_data, target_test_data = y['Train'], y['Test']
            for subject, data in feature_training_data.items():
                y_train, y_test = target_training_data[subject]['Targets'], target_test_data[subject]["Targets"]

                if self.select_text_features and self.select_eeg_features:
                    text_data_train, text_data_test = data['Queries'], feature_test_data[subject]['Queries']
                    eeg_data_train, eeg_data_test = data['EEG'], feature_test_data[subject]['EEG']
                    X_train = np.concatenate((text_data_train, eeg_data_train), axis=1)
                    X_test = np.concatenate((text_data_test, eeg_data_test), axis=1)

                if self.select_text_features:
                    X_train, X_test = data['Queries'], feature_test_data[subject]['Queries']

                elif self.select_eeg_features:
                    X_train, X_test = data['EEG'], feature_test_data[subject]['EEG']

                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
                loo_scores.append(-mean_squared_error(y_test, y_pred))  # Negative because GridSearchCV maximizes
            return np.mean(loo_scores)

        optimizer = ModelOptimizer(
            model_name=model_name,
            param_grid=param_grid,
            evaluator_to_use=evaluator_to_use,
            custom_cv_scorer=custom_cv_scorer,
            custom_loo_scorer=custom_loo_scorer,
            random_state=self.random_state
        )

        return optimizer.grid_search(features, targets)

    def group_kfold_evaluator(self, model, features, targets, groups):
        gkf = GroupKFold(n_splits=self.n_splits)
        fold_metrics, all_predictions, all_true_values = [], [], []

        print('--Model Training Started--')

        for fold_idx, (train_index, test_index) in enumerate(gkf.split(features, targets, groups=groups)):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = targets[train_index], targets[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics for this fold
            fold_metrics.append(self.calculate_qpp_metrics(y_test, y_pred))

            # Store predictions and true values
            all_predictions.extend(y_pred)
            all_true_values.extend(y_test)

            # Print sample predictions
            print(f"\nFold {fold_idx + 1} Sample Predictions:")
            for i in range(min(5, len(y_test))):
                print(f"Query Index: {test_index[i]}")
                print(f"True NDCG: {y_test[i]:.3f}, Predicted NDCG: {y_pred[i]:.3f}")

        # Calculate average metrics across folds
        avg_metrics = {
            'error_metrics': {},
            'correlation_metrics': {},
            'classification_metrics': {}
        }

        metric_categories = ['error_metrics', 'correlation_metrics', 'classification_metrics']
        for category in metric_categories:
            for metric in fold_metrics[0][category]:
                values = [fold[category][metric] for fold in fold_metrics]
                avg_metrics[category][metric] = float(np.mean(values))
                avg_metrics[category][f'{metric}_std'] = float(np.std(values))

        # Calculate overall metrics using all predictions
        overall_metrics = self.calculate_qpp_metrics(all_true_values, all_predictions)

        results = {
            'average_fold_metrics': avg_metrics,
            'overall_metrics': overall_metrics,
            'fold_metrics': fold_metrics
        }

        self.dipslay_evaluation_results(overall_metrics)

        return results

    def prepare_data_for_LOO_evaluation(self, data_and_targets):
        features_dict = {'Train': {}, 'Test': {}}
        targets_dict = {'Train': {}, 'Test': {}}
        subject_ids = list(data_and_targets.keys())
        input_shape = None
        for subject in subject_ids:
            # Test set
            if self.select_eeg_features:
                test_subject_eeg_data = np.asarray(data_and_targets[subject]['EEG'])

            test_subject_queries = np.asarray(data_and_targets[subject]['Queries'])
            test_subject_targets = np.asarray(data_and_targets[subject]['Targets'])

            if self.select_eeg_features and self.select_text_features:
                features_dict['Test'][subject] = {
                    'EEG': test_subject_eeg_data,
                    'Queries': test_subject_queries
                }

            elif self.select_text_features:
                features_dict['Test'][subject] = {'Queries': test_subject_queries}

            elif self.select_eeg_features:
                features_dict['Test'][subject] = {'EEG': test_subject_eeg_data}

            targets_dict['Test'][subject] = {'Targets': test_subject_targets}

            # Training set (all other subjects)
            training_subjects = [s for s in subject_ids if s != subject]

            if self.select_eeg_features:
                training_subjects_eeg_features = [data_and_targets[s]['EEG'] for s in training_subjects]

            training_subjects_targets = [data_and_targets[s]['Targets'] for s in training_subjects]
            training_subjects_queries = [data_and_targets[s]['Queries'] for s in training_subjects]

            if self.select_eeg_features and self.select_text_features:
                features_dict['Train'][subject] = {
                    'EEG': np.concatenate(training_subjects_eeg_features, axis=0),
                    'Queries': np.concatenate(training_subjects_queries, axis=0)
                }
            elif self.select_text_features:
                features_dict['Train'][subject] = {'Queries': np.concatenate(training_subjects_queries, axis=0)}

            elif self.select_eeg_features:
                features_dict['Train'][subject] = {'EEG': np.concatenate(training_subjects_eeg_features, axis=0)}

            targets_dict['Train'][subject] = {
                'Targets': np.concatenate(training_subjects_targets, axis=0)
            }

        return features_dict, targets_dict

    def leave_one_out_evaluator(self, model, features, targets, groups):
        fold_metrics, all_predictions, all_true_values = [], [], []

        print('--Model Training Started--')

        feature_training_data, feature_test_data = features['Train'], features['Test']
        target_training_data, target_test_data = targets['Train'], targets['Test']

        for subject, data in feature_training_data.items():
            y_train, y_test = target_training_data[subject]['Targets'], target_test_data[subject]["Targets"]

            if self.select_text_features and self.select_eeg_features:
                text_data_train, text_data_test = data['Queries'], feature_test_data[subject]['Queries']
                eeg_data_train, eeg_data_test = data['EEG'], feature_test_data[subject]['EEG']
                X_train = np.concatenate((text_data_train, eeg_data_train), axis=1)
                X_test = np.concatenate((text_data_test, eeg_data_test), axis=1)

            if self.select_text_features:
                X_train, X_test = data['Queries'], feature_test_data[subject]['Queries']

            elif self.select_eeg_features:
                X_train, X_test = data['EEG'], feature_test_data[subject]['EEG']

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics for this fold
            fold_metrics.append(self.calculate_qpp_metrics(y_test, y_pred))

            # Store predictions and true values
            all_predictions.extend(y_pred)
            all_true_values.extend(y_test)

            # Print sample predictions
            print(f"\nSubject {subject} Sample Predictions:")
            for i in range(min(5, len(y_test))):
                print(f"Test data index: {i}")
                print(f"True Score: {y_test[i]:.3f}, Predicted Score: {y_pred[i]:.3f}")

        # Calculate average metrics across folds
        avg_metrics = {
            'error_metrics': {},
            'correlation_metrics': {},
            'classification_metrics': {}
        }

        metric_categories = ['error_metrics', 'correlation_metrics', 'classification_metrics']
        for category in metric_categories:
            for metric in fold_metrics[0][category]:
                values = [fold[category][metric] for fold in fold_metrics]
                avg_metrics[category][metric] = float(np.mean(values))
                avg_metrics[category][f'{metric}_std'] = float(np.std(values))

        # Calculate overall metrics using all predictions
        overall_metrics = self.calculate_qpp_metrics(all_true_values, all_predictions)

        results = {
            'average_fold_metrics': avg_metrics,
            'overall_metrics': overall_metrics,
            'fold_metrics': fold_metrics
        }

        self.dipslay_evaluation_results(overall_metrics)

        return results

    def choose_machine_learning_evaluator(self, name: str, *args, **kwargs):
        if hasattr(self, name) and callable(func := getattr(self, name)):
            return func(*args, **kwargs)


class ModelOptimizer:
    def __init__(self, model_name, param_grid, evaluator_to_use, custom_cv_scorer, custom_loo_scorer, random_state=42):
        self.model_class = MODEL_MAPPING.get(model_name)
        self.model_name = model_name
        if self.model_class is None:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(MODEL_MAPPING.keys())}")

        self.param_grid = param_grid
        self.evaluator_to_use = evaluator_to_use
        self.custom_cv_scorer = custom_cv_scorer
        self.custom_loo_scorer = custom_loo_scorer
        self.random_state = random_state

    def grid_search(self, features, targets):
        best_params = None
        best_score = float('-inf')

        # Generate hyperparameter combinations
        param_combinations = list(product(*self.param_grid.values()))
        param_keys = list(self.param_grid.keys())

        for param_values in param_combinations:
            current_params = dict(zip(param_keys, param_values))
            if self.model_name != "SVR":
                current_params["random_state"] = self.random_state

            current_model = self.model_class(**current_params)

            # Select the evaluation method
            if self.evaluator_to_use == 'group_kfold_evaluator':
                current_score = self.custom_cv_scorer(current_model, features, targets)
            elif self.evaluator_to_use == 'leave_one_out_evaluator':
                current_score = self.custom_loo_scorer(current_model, features, targets)
            else:
                raise ValueError(f"Unsupported evaluator: {self.evaluator_to_use}")

            if current_score > best_score:
                best_score = current_score
                best_params = current_params
                print(f"New best parameters found: {best_params} with score: {-best_score}")

        return {'best_params': best_params, 'best_score': -best_score}  # Convert back to MSE
