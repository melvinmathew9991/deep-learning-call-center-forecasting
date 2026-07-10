from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

from .utils import evaluate, plot_predictions, to_series


### LSTM - build a class for lstm model
class LSTM_model:

    def __init__(self, X_train, X_valid, X_test, Y_train, Y_valid, Y_test):
       # define the parameters
        epochs = 100
        batch = 256
        lr = 0.0003
        adam = optimizers.Adam(lr)

        X_train_series, X_valid_series = to_series(X_train), to_series(X_valid)

        model_lstm = self.model_arch(X_train_series, adam)

        model_lstm = self.model_train(X_train_series, X_valid_series, epochs, batch, model_lstm, Y_train, Y_valid)

        self.metrics = self.model_predict(model_lstm, X_test, Y_test)

        self.model = model_lstm


    def model_predict(self, model_lstm, X_test, Y_test):
        """
        model prediction
        :param model_lstm:
        :param X_test:
        :param Y_test:
        :return:
        """
        X_test_series = to_series(X_test)
        lstm_pred = model_lstm.predict(X_test_series)
        plot_predictions("LSTM", Y_test, lstm_pred, "lstm.png")
        return evaluate("LSTM", Y_test, lstm_pred)


    def model_train(self, X_train_series, X_valid_series, epochs, batch, model_lstm, Y_train, Y_valid):
        """
        model training
        :param X_train_series:
        :param X_valid_series:
        :param epochs:
        :param batch:
        :param model_lstm:
        :param Y_train:
        :param Y_valid:
        :return:
        """
        model_lstm.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid),
                       epochs=epochs, batch_size=batch, verbose=2)
        return model_lstm


    def model_arch(self, X_train_series, adam):
        """
        model architecture
        :param X_train_series:
        :param adam:
        :return:
        """
        model_lstm = Sequential()
        model_lstm.add(LSTM(50, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
        model_lstm.add(Dense(1))
        model_lstm.compile(loss='mse', optimizer=adam)
        model_lstm.summary()
        return model_lstm
