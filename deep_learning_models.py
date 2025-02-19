from tensorflow.keras.layers import (Input, Conv1D, Conv2D, BatchNormalization, ReLU,
                                     MaxPooling1D, MaxPooling2D, Reshape, Bidirectional, GRU, Dense, Concatenate)
from tensorflow.keras.models import Model


class SingleModalityRegressionModel:
    def __init__(self, input_shape, conv_filters, gru_units, dense_units, select_eeg_features):
        """
        Initializes the regression model.

        Args:
            input_shape (tuple): Shape of the input data.
            conv_filters (int): Number of filters for the Conv layer.
            gru_units (int): Number of units for the GRU layers.
            dense_units (int): Number of units for the Dense layer.
            modality (str): 'eeg' for EEG data, 'text' for text data.
        """
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.select_eeg_features = select_eeg_features

    def build_model(self):
        """
        Builds and returns the appropriate regression model based on modality.

        Returns:
            tf.keras.Model: A compiled regression model.
        """
        inputs = Input(shape=self.input_shape, name='input')

        if self.select_eeg_features:
            x = Conv2D(filters=self.conv_filters, kernel_size=3, dilation_rate=2, padding='same', activation='relu',
                       name='conv')(inputs)
            x = MaxPooling2D(pool_size=2, padding='same')(x)
            x = Reshape((x.shape[1]*x.shape[2], self.conv_filters))(x)
        else:
            x = Conv1D(filters=self.conv_filters, kernel_size=3, dilation_rate=2, padding='same', activation='relu',
                       name='conv')(inputs)
            x = MaxPooling1D(pool_size=2, padding='same')(x)

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Bidirectional(GRU(units=self.gru_units, return_sequences=True, name='gru1'))(x)
        x = Bidirectional(GRU(units=self.gru_units, name='gru2'))(x)
        x = Dense(units=self.dense_units, activation='relu')(x)
        x = Dense(units=1, name='output')(x)

        model = Model(inputs=inputs, outputs=x, name='nn_regressor')
        return model
class DualModalityRegressionModel:
    def __init__(self, eeg_input_shape, text_input_shape, conv_filters, gru_units, dense_units,
                 projection_units=None):
        self.eeg_input_shape = eeg_input_shape
        self.text_input_shape = text_input_shape
        self.conv_filters = conv_filters
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.projection_units = projection_units

    def build_model(self, use_projections=False):
        eeg_inputs = Input(shape=self.eeg_input_shape, name='eeg_input')
        text_inputs = Input(shape=self.text_input_shape, name='text_input')

        # EEG Branch
        x = Conv2D(filters=self.conv_filters, kernel_size=3, dilation_rate=2, padding='same', activation='relu',
                   name='conv2d_eeg')(eeg_inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=2, padding='same')(x)
        x = Reshape((x.shape[1]*x.shape[2], self.conv_filters))(x)

        if use_projections:
            x = Dense(units=self.projection_units, name='eeg_projection')(x)
        else:
            x = Bidirectional(GRU(units=self.gru_units, return_sequences=True, name='gru1_eeg'))(x)
            x = Bidirectional(GRU(units=self.gru_units, name='gru2_eeg'))(x)

        x_model = Model(inputs=eeg_inputs, outputs=x)

        # Text Branch
        y = Conv1D(filters=self.conv_filters, kernel_size=3, dilation_rate=2, padding='same', activation='relu',
                   name='conv1d_text')(text_inputs)
        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = MaxPooling1D(pool_size=2)(y)

        if use_projections:
            y = Dense(units=self.projection_units, name='text_projection')(y)
        else:
            y = Bidirectional(GRU(units=self.gru_units, return_sequences=True, name='gru1_text'))(y)
            y = Bidirectional(GRU(units=self.gru_units, name='gru2_text'))(y)

        y_model = Model(inputs=text_inputs, outputs=y)

        # Fusion Layer
        combined = Concatenate()([x_model.output, y_model.output])

        if use_projections:
            z = Bidirectional(GRU(units=self.gru_units, return_sequences=True, name='gru1_combined'))(combined)
            z = Bidirectional(GRU(units=self.gru_units, name='gru2_combined'))(z)
        else:
            z = Dense(units=self.dense_units, activation='relu', name='dense_combined')(combined)

        z = Dense(units=1, name='output')(z)
        model = Model(inputs=[x_model.input, y_model.input], outputs=z, name='bimodal_nn_regressor')

        return model
