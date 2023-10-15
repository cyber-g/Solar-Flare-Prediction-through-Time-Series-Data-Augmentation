import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

from utils.utils import calculate_metrics
from utils.utils import save_test_duration, save_logs


class Classifier_LSTM_FCN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def squeeze_excite_block(self, input):
        ''' Create a squeeze-excite block
        Args:
            input: input tensor
            filters: number of output filters
            k: width factor
        Returns: a keras tensor
        '''
        filters = input.shape[-1]  # channel_axis = -1 for TF

        se = keras.layers.GlobalAveragePooling1D()(input)
        se = keras.layers.Reshape((1, filters))(se)
        se = keras.layers.Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = keras.layers.multiply([input, se])
        return se

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(shape=input_shape)

        lstm = keras.layers.Masking()(input_layer)
        lstm = keras.layers.LSTM(8)(lstm)
        lstm = keras.layers.Dropout(0.8)(lstm)

        conv1 = keras.layers.Permute((2, 1))(input_layer)
        conv1 = keras.layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(conv1)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation('relu')(conv1)
        conv1 = self.squeeze_excite_block(conv1)

        conv2 = keras.layers.Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)
        conv2 = self.squeeze_excite_block(conv2)

        conv3 = keras.layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)
        # conv3 = keras.layers.squeeze_excite_block(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
        in_tol = keras.layers.concatenate([lstm, gap_layer])

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(in_tol)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate= 0.1),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10,
                                                      min_lr=0.00001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='min')

        self.callbacks = [reduce_lr, model_checkpoint, early_stopping]

        return model

    def fit(self, x_train, y_train, x_test, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 64
        nb_epochs = 500

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_split=0.25, callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        y_pred = self.predict(x_test, y_true,
                              return_df_metrics=False)

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)
        np.save(self.output_directory + 'y_true.npy', y_true)
        np.save(self.output_directory + 'x_test.npy', x_test)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
        return y_pred
