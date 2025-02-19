import numpy as np
import tensorflow as tf

class EEGDataLoader:
    def __init__(self, score_data, select_eeg_features, select_deep_learning_model, selected_metric, tokenizer):
        """
        Initializes the EEGDataLoader with provided data and options.

        :param score_data: Dictionary containing EEG and metric data.
        :param select_eeg_features: Boolean flag to include EEG features.
        :param select_deep_learning_model: Boolean flag for transforming EEG according to CNN or ML model requirements.
        :param selected_metric: The metric to extract scores from.
        :param tokenizer: Tokenizer with which to tokenize queries
        """
        self.score_data = score_data
        self.select_eeg_features = select_eeg_features
        self.select_deep_learning_model = select_deep_learning_model
        self.selected_metric = selected_metric
        self.tokenizer = tokenizer

    @staticmethod
    def pad_eeg_features(eeg_list, is_deep_learning):
        """
        Pads all EEG tensors to have the same dimensions in the list.

        :param eeg_list: List of EEG tensors with different shapes.
        :param is_deep_learning: Boolean flag to determine 3D or 1D padding.
        :return: List of EEG tensors with uniform dimensions.
        """

        if is_deep_learning:
            max_dim_1 = max(tensor.shape[1] for tensor in eeg_list)
            max_dim_2 = max(tensor.shape[2] for tensor in eeg_list)
            target_shape = (1, max_dim_1, max_dim_2)
            padded_list = []

            for tensor in eeg_list:
                pad_0 = target_shape[1] - tensor.shape[1] if tensor.shape[1] < target_shape[1] else 0
                pad_1 = target_shape[2] - tensor.shape[2] if tensor.shape[2] < target_shape[2] else 0
                padded_tensor = tf.pad(tensor, [[0, 0], [0, pad_0], [0, pad_1]], mode='CONSTANT')
                padded_list.append(padded_tensor)

        else:
            max_length = max(tensor.shape[0] for tensor in eeg_list)
            target_shape = (max_length,)
            padded_list = []

            for tensor in eeg_list:
                pad_size = target_shape[0] - tensor.shape[0] if tensor.shape[0] < target_shape[0] else 0
                padded_tensor = tf.pad(tensor, [[0, pad_size]], mode='CONSTANT')
                padded_list.append(padded_tensor)

        return padded_list

    def load_eeg_averaged_across_subjects(self):
        """
        Loads EEG data and scores while averaging EEG features across subjects for each unique query.
        Pads the averaged EEG data.

        :return: If eeg feature are required it returns all the tokenized queries (tensor format), corresponding targets (list), and EEG features (tensor format).
                 Else it returns all the tokenized queries (tensor format), and corresponding targets (list).
        """
        # Create lists for unique queries, their features, and scores
        unique_queries = []
        unique_targets = []
        if self.select_eeg_features:
            unique_eeg_features = []  # For storing averaged EEG features per query

        # Process data maintaining unique queries
        for key, value in self.score_data['metrics'].items():
            if key not in unique_queries:
                unique_queries.append(key)
                unique_targets.append(value[self.selected_metric])

                if self.select_eeg_features:
                    # Average EEG features across subjects for this query
                    all_subject_eegs = []
                    with tf.device('/GPU:0'):
                        for subject, eeg_segment in value['eeg_data']['query_eeg'].items():
                            # Convert to tensor and ensure same shape by padding if necessary
                            eeg_tensor = tf.convert_to_tensor(eeg_segment, dtype=tf.float32)
                            # Reshape to 2D if needed
                            if len(eeg_tensor.shape) > 2:
                                eeg_tensor = tf.reshape(eeg_tensor, (eeg_tensor.shape[0], -1))
                            all_subject_eegs.append(eeg_tensor)

                        # Stack the tensors and compute mean across subjects
                        # First pad all tensors to the same length
                        max_length = max(tensor.shape[0] for tensor in all_subject_eegs)
                        padded_eegs = []
                        for tensor in all_subject_eegs:
                            padding_size = max_length - tensor.shape[0]
                            if padding_size > 0:
                                padded = tf.pad(tensor, [[0, padding_size], [0, 0]], mode='constant')
                                padded_eegs.append(padded)
                            else:
                                padded_eegs.append(tensor)

                        # Now we can safely stack and average
                        stacked_eegs = tf.stack(padded_eegs)
                        avg_eeg = tf.reduce_mean(stacked_eegs, axis=1)


                        #Change tensor size to 3D for CNN or 1D for ML model
                        if self.select_deep_learning_model:
                            avg_eeg_expanded = tf.expand_dims(avg_eeg, axis=0)
                            unique_eeg_features.append(avg_eeg_expanded)
                        else:
                            avg_eeg_reduced = tf.reshape(avg_eeg, [-1])
                            unique_eeg_features.append(avg_eeg_reduced)

        tokenized_arrayed_unique_queries = np.asarray(self.tokenizer(unique_queries, padding=True)['input_ids'])
        arrayed_unique_targets = np.asarray(unique_targets)
        if self.select_eeg_features:
            #Pad EEG features of all queries to be the same shape
            unique_eeg_features = self.__class__.pad_eeg_features(unique_eeg_features, self.select_deep_learning_model)
            arrayed_unique_eeg_features = np.asarray(unique_eeg_features)
            print('Loading Done')
            return {"Queries": tokenized_arrayed_unique_queries, "Targets": arrayed_unique_targets, "EEG": arrayed_unique_eeg_features, 'Groups': unique_queries}

        else:
            return {"Queries": tokenized_arrayed_unique_queries, "Targets": arrayed_unique_targets, 'Groups': unique_queries}

    def load_eeg_without_averaging_across_participants(self):
        """
        Loads EEG data without averaging EEG features across participants, preserving all queries, targets and EEG signals.
        Pads EEG according to the max EEG recording length.

        :return: If eeg feature are required it returns all the tokenized queries (tensor format), corresponding targets (list), and EEG features (tensor format).
                 Else it returns all the tokenized queries (tensor format), and corresponding targets (list).
        """
        queries = []
        targets = []
        if self.select_eeg_features:
            # Add EEG features of all subjects
            all_subject_eegs = []

        # Process data adding all queries
        for key, value in self.score_data['metrics'].items():
            if self.select_eeg_features:
                for subject, eeg_segment in value['eeg_data']['query_eeg'].items():
                    queries.append(key)
                    targets.append(value[self.selected_metric])
                    # Convert to tensor and ensure same shape by padding if necessary
                    eeg_tensor = tf.convert_to_tensor(eeg_segment, dtype=tf.float32)
                    # Reshape to 2D if needed
                    if len(eeg_tensor.shape) > 2:
                        eeg_tensor = tf.reshape(eeg_tensor, (eeg_tensor.shape[0], -1))
                    all_subject_eegs.append(eeg_tensor)
            else:
                queries.append(key)
                targets.append(value[self.selected_metric])

                # Pad all tensors to the same length
        if self.select_eeg_features:
            with tf.device('/GPU:0'):
                max_length = max(tensor.shape[0] for tensor in all_subject_eegs)
                padded_eegs = []
                for tensor in all_subject_eegs:
                    padding_size = max_length - tensor.shape[0]
                    if padding_size > 0:
                        padded = tf.pad(tensor, [[0, padding_size], [0, 0]], mode='constant')

                        if self.select_deep_learning_model:
                            padded_expanded_dims = tf.expand_dims(padded, axis=0)
                            padded_eegs.append(padded_expanded_dims)

                        else:
                            padded_reduced_dims = tf.reshape(padded, [-1])
                            padded_eegs.append(padded_reduced_dims)

                    else:

                        if self.select_deep_learning_model:
                            tensor_expanded_dims = tf.expand_dims(tensor, axis=0)
                            padded_eegs.append(tensor_expanded_dims)

                        else:
                            tensor_reduced_dims = tf.reshape(tensor, [-1])
                            padded_eegs.append(tensor_reduced_dims)

        tokenized_tensored_queries = tf.convert_to_tensor(self.tokenizer(queries, padding=True)['input_ids'])
        if self.select_eeg_features:
            tensored_eeg_features = tf.convert_to_tensor(padded_eegs)
            return {
                "Queries": tokenized_tensored_queries.numpy(),  # Convert to numpy
                "Targets": np.array(targets),
                "EEG": tensored_eeg_features.numpy(),  # Convert to numpy
                "Groups": queries
            }
        else:
            return {
                "Queries": tokenized_tensored_queries.numpy(),  # Convert to numpy
                "Targets": np.array(targets),
                "Groups": queries
            }

    def load_eeg_per_subject(self):
        """
        Organizes EEG data and scores per subject, maintaining query-specific EEG features separately for each participant.
        Pads EEG signals so that all subjects EEG recordings are the same length.

        :return: Dictionary where each key is a subject ID:
         If eeg feature are required dictionary includes all the tokenized queries (tensor format), corresponding targets (list), and EEG features (tensor format).
         Else dictionary includes  all the tokenized queries (tensor format), and corresponding targets (list).
        """
        all_subjects_dict = {}

        if self.select_eeg_features:
            # Add EEG features of all subjects
            all_subject_eegs = []

        # Process data adding all queries
        for key, value in self.score_data['metrics'].items():
            for subject, eeg_segment in value['eeg_data']['query_eeg'].items():
                if subject not in list(all_subjects_dict.keys()):
                    if self.select_eeg_features:
                        all_subjects_dict[subject] = {'EEG': [], 'Queries':[], 'Targets':[]}
                    else:
                        all_subjects_dict[subject] = {'Queries': [], 'Targets': []}

                all_subjects_dict[subject]['Queries'].append(key)
                all_subjects_dict[subject]['Targets'].append(value[self.selected_metric])

                if self.select_eeg_features:
                    # Convert to tensor and ensure same shape by padding if necessary
                    eeg_tensor = tf.convert_to_tensor(eeg_segment, dtype=tf.float32)
                    # Reshape to 2D if needed
                    if len(eeg_tensor.shape) > 2:
                        eeg_tensor = tf.reshape(eeg_tensor, (eeg_tensor.shape[0], -1))

                    all_subject_eegs.append(eeg_tensor)
                    all_subjects_dict[subject]['EEG'].append(eeg_tensor)

            # Pad all tensors to the same length
        if self.select_eeg_features:
            with tf.device('/GPU:0'):
                max_length = max(tensor.shape[0] for tensor in all_subject_eegs)
                for subject, data_dict in all_subjects_dict.items():
                    padded_eegs = []
                    for tensor in data_dict['EEG']:
                        padding_size = max_length - tensor.shape[0]
                        if padding_size > 0:
                            padded = tf.pad(tensor, [[0, padding_size], [0, 0]], mode='constant')

                            if self.select_deep_learning_model:
                                padded_expanded_dims = tf.expand_dims(padded, axis=0)
                                padded_eegs.append(padded_expanded_dims)

                            else:
                                padded_reduced_dims = tf.reshape(padded, [-1])
                                padded_eegs.append(padded_reduced_dims)
                        else:

                            if self.select_deep_learning_model:
                                tensor_expanded_dims = tf.expand_dims(tensor, axis=0)
                                padded_eegs.append(tensor_expanded_dims)

                            else:
                                tensor_reduced_dims = tf.reshape(tensor, [-1])
                                padded_eegs.append(tensor_reduced_dims)

                    all_subjects_dict[subject]['EEG'] = tf.convert_to_tensor(padded_eegs)
                    all_subjects_dict[subject]['Queries'] = tf.convert_to_tensor(self.tokenizer(all_subjects_dict[subject]['Queries'], padding=True)['input_ids'])
        else:
            for subject, data_dict in all_subjects_dict.items():
                all_subjects_dict[subject]['Queries'] = tf.convert_to_tensor(self.tokenizer(all_subjects_dict[subject]['Queries'], padding=True)['input_ids'])

        return all_subjects_dict
    def choose_loader(self, name: str):
        if hasattr(self, name) and callable(func := getattr(self, name)):
            return func()

    def check_loader_output_format(self, function_name: str):
        """
        Prints detailed information about the output structure of a given data loading function.

        :param function_name: The name of the data loading function to evaluate.
        """
        if not hasattr(self, function_name) or not callable(getattr(self, function_name)):
            raise ValueError(f"Invalid function name: {function_name}")

        print(f'Viewing output of function {function_name}')

        output = getattr(self, function_name)()

        if function_name == 'load_eeg_averaged_across_subjects':
            if self.select_eeg_features:
                print(f'With EEG features')
                unique_queries, unique_targets, unique_eeg_features = output
                print(f'Number of unique queries: {len(unique_queries)}')
                print(f'Number of unique targets: {len(unique_targets)}')
                print(f'Number of unique EEG segments: {len(unique_eeg_features)}')
                print(f'A query example is: {unique_queries[0]}')
                print(f'Dimensions of queries are: {unique_queries.shape}')
                print(f'A target example is: {unique_targets[0]}')
                print(f'The shape of an EEG tensor is: {unique_eeg_features[0].shape}')
                print(f'The shape of another EEG tensor is: {unique_eeg_features[100].shape}')

            else:
                print(f'Without EEG features')
                unique_queries, unique_targets = output
                print(f'Number of unique queries: {len(unique_queries)}')
                print(f'Dimensions of queries are: {unique_queries.shape}')
                print(f'Number of unique targets: {len(unique_targets)}')
                print(f'A query example is: {unique_queries[0]}')
                print(f'A target example is: {unique_targets[0]}')

        if function_name == 'load_eeg_without_averaging_across_participants':
            if self.select_eeg_features:
                print(f'With EEG features')
                queries, targets, eeg_features = output
                print(f'Number of queries: {len(queries)}')
                print(f'Number of targets: {len(targets)}')
                print(f'Number of EEG segments: {len(eeg_features)}')
                print(f'A query example is: {queries[0]}')
                print(f'Dimensions of queries are: {queries.shape}')
                print(f'A target example is: {targets[0]}')
                print(f'The shape of an EEG tensor is: {eeg_features[0].shape}')
                print(f'The shape of another EEG tensor is: {eeg_features[100].shape}')

            else:
                print(f'Without EEG features')
                queries, targets = output
                print(f'Number of unique queries: {len(queries)}')
                print(f'Number of unique targets: {len(targets)}')
                print(f'A query example is: {queries[0]}')
                print(f'Dimensions of queries are: {queries.shape}')
                print(f'A target example is: {targets[0]}')

        if function_name == 'load_eeg_per_subject':
            if self.select_eeg_features:
                print(f'With EEG features')
                subjects_dict = output
                subject_ids = list(subjects_dict.keys())
                example_subject_id = subject_ids[0]
                another_example_id = subject_ids[5]

                example_subject_queries = subjects_dict[example_subject_id]["Queries"]
                example_subject_targets= subjects_dict[example_subject_id]["Targets"]
                example_subject_eeg = subjects_dict[example_subject_id]["EEG"]

                another_example_subject_queries = subjects_dict[another_example_id]["Queries"]
                another_example_subject_targets= subjects_dict[another_example_id]["Targets"]
                another_example_subject_eeg = subjects_dict[another_example_id]["EEG"]

                print(f'Number of subjects: {len(subject_ids)}')
                print(f'The keys of the data dictionary (subject ids) are: {subject_ids}')
                print(f'The keys of the data dictionary of each subject is: {list(subjects_dict[example_subject_id].keys())}')

                print(f'The number of queries for example subject {example_subject_id} is: {len(example_subject_queries)}')
                print(f'The number of targets for example subject {example_subject_id} is: {len(example_subject_targets)}')
                print(f'The number of EEG segments for example subject {example_subject_id} is: {len(example_subject_eeg)}')

                print(f'The number of queries for another example subject {another_example_id} is: {len(another_example_subject_queries)}')
                print(f'The number of targets for another example subject {another_example_id} is: {len(another_example_subject_targets)}')
                print(f'The number of EEG segments for another example subject {another_example_id} is: {len(another_example_subject_eeg)}')

                print(f'A query example is: {example_subject_queries[0]}')
                print(f'Dimensions of queries are: {example_subject_queries.shape}')
                print(f'A target example is: {another_example_subject_targets[0]}')
                print(f'The shape of an EEG tensor for example subject {example_subject_id} is: {example_subject_eeg[0].shape}')
                print(f'The shape of another EEG tensor for another example subject {another_example_id} is: {another_example_subject_eeg[0].shape}')

            else:
                print(f'Without EEG features')
                subjects_dict = output
                subject_ids = list(subjects_dict.keys())
                example_subject_id = subject_ids[0]
                another_example_id = subject_ids[5]

                example_subject_queries = subjects_dict[example_subject_id]["Queries"]
                example_subject_targets = subjects_dict[example_subject_id]["Targets"]

                another_example_subject_queries = subjects_dict[example_subject_id]["Queries"]
                another_example_subject_targets = subjects_dict[example_subject_id]["Targets"]

                print(f'Number of subjects: {len(subject_ids)}')
                print(f'The keys of the data dictionary (subject ids) are: {subject_ids}')
                print(f'The keys of the data dictionary of each subject is: {list(subjects_dict[example_subject_id].keys())}')

                print(f'The number of queries for example subject {example_subject_id} is: {len(example_subject_queries)}')
                print(f'The number of targets for example subject {example_subject_id} is: {len(example_subject_targets)}')

                print(f'The number of queries for another example subject {another_example_id} is: {len(another_example_subject_queries)}')
                print(f'The number of targets for another example subject {another_example_id} is: {len(another_example_subject_targets)}')

                print(f'A query example is: {example_subject_queries[0]}')
                print(f'Dimensions of queries are: {example_subject_queries.shape}')
                print(f'A target example is: {another_example_subject_targets[0]}')
