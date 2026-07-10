from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LSTM, MaxPooling1D, TimeDistributed
from tensorflow.keras.models import Sequential

from .utils import evaluate, plot_predictions, to_series


### CNN-LSTM - build a class for cnn_lstm model

class CNN_LSTM:

    def __init__(self, X_train, X_valid, X_test, Y_train, Y_valid, Y_test):
        # define the parameters
        epochs = 100
        batch = 256
        lr = 0.0003
        adam = optimizers.Adam(lr)

        X_train_series, X_valid_series = to_series(X_train), to_series(X_valid)  # reshape the data

        X_train_series_sub, X_valid_series_sub = self.set_data(X_train_series, X_valid_series)

        model_cnn_lstm = self.model_arch(X_train_series_sub, adam)

        self.model_train(X_train_series_sub, X_valid_series_sub, Y_train, Y_valid, epochs, batch, model_cnn_lstm)

        self.metrics = self.model_predict(X_test, Y_test, model_cnn_lstm)

        self.model = model_cnn_lstm


    def model_predict(self, X_test, Y_test, model_cnn_lstm):
        """
        Model for prediction
        :param X_test:
        :param Y_test:
        :param model_cnn_lstm:
        :return:
        """
        X_test_series = to_series(X_test)
        n_past = X_test_series.shape[1]
        X_test_series_sub = X_test_series.reshape((X_test_series.shape[0], 1, n_past, 1))
        cnn_lstm_pred = model_cnn_lstm.predict(X_test_series_sub)
        plot_predictions("CNN-LSTM", Y_test, cnn_lstm_pred, "cnn_lstm.png")
        return evaluate("CNN-LSTM", Y_test, cnn_lstm_pred)


    def model_train(self, X_train_series_sub, X_valid_series_sub, Y_train, Y_valid, epochs, batch, model_cnn_lstm):
        """
        Model for train
        :param X_train_series_sub:
        :param X_valid_series_sub:
        :param Y_train:
        :param Y_valid:
        :param epochs:
        :param batch:
        :param model_cnn_lstm:
        :return:
        """
        model_cnn_lstm.fit(X_train_series_sub, Y_train,
                           validation_data=(X_valid_series_sub, Y_valid),
                           epochs=epochs, batch_size=batch, verbose=2)


    def model_arch(self, X_train_series_sub, adam):
        """
        Setting the model Arch
        :param X_train_series_sub:
        :param adam:
        :return:
        """
        model_cnn_lstm = Sequential()
        model_cnn_lstm.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(
        None, X_train_series_sub.shape[2], X_train_series_sub.shape[3])))
        model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model_cnn_lstm.add(TimeDistributed(Flatten()))
        model_cnn_lstm.add(LSTM(50, activation='relu'))
        model_cnn_lstm.add(Dense(1))
        model_cnn_lstm.compile(loss='mse', optimizer=adam)
        return model_cnn_lstm


    def set_data(self, X_train_series, X_valid_series):
        """
        reshaping the data into (samples, 1, n_past, 1) sub-sequences for TimeDistributed layers
        :param X_train_series:
        :param X_valid_series:
        :return:
        """
        n_past = X_train_series.shape[1]
        X_train_series_sub = X_train_series.reshape((X_train_series.shape[0], 1, n_past, 1))
        X_valid_series_sub = X_valid_series.reshape((X_valid_series.shape[0], 1, n_past, 1))
        print('Train set shape', X_train_series_sub.shape)
        print('Validation set shape', X_valid_series_sub.shape)
        return X_train_series_sub, X_valid_series_sub
