from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential

from .utils import evaluate, plot_predictions, to_series


### CNN - build a class for cnn model

class CNN_model:

    def __init__(self, X_train, X_valid, X_test, Y_train, Y_valid, Y_test):
        # define the parameters
        epochs = 100
        batch = 256
        lr = 0.0003
        adam = optimizers.Adam(lr)

        X_train_series, X_valid_series = to_series(X_train), to_series(X_valid)

        model_cnn = self.model_arch(X_train_series, adam)

        self.model_train(X_train_series, X_valid_series, epochs, batch, model_cnn, Y_train, Y_valid)

        self.metrics = self.model_predict(model_cnn, X_test, Y_test)

        self.model = model_cnn


    def model_predict(self, model_cnn, X_test, Y_test):
        """
        model for prediction
        :param model_cnn:
        :param X_test:
        :param Y_test:
        :return:
        """
        X_test_series = to_series(X_test)
        cnn_pred = model_cnn.predict(X_test_series)
        plot_predictions("CNN", Y_test, cnn_pred, "cnn.png")
        return evaluate("CNN", Y_test, cnn_pred)


    def model_train(self, X_train_series, X_valid_series, epochs, batch, model_cnn, Y_train, Y_valid):
        """
        model for training
        :param X_train_series:
        :param X_valid_series:
        :param epochs:
        :param batch:
        :param model_cnn:
        :param Y_train:
        :param Y_valid:
        :return:
        """
        model_cnn.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid),
                      epochs=epochs, batch_size=batch, verbose=2)


    def model_arch(self, X_train_series, adam):
        """
        model architecture
        :param X_train_series:
        :param adam:
        :return:
        """
        model_cnn = Sequential()
        model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                             input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
        model_cnn.add(MaxPooling1D(pool_size=2))
        model_cnn.add(Flatten())
        model_cnn.add(Dense(50, activation='relu'))
        model_cnn.add(Dense(1))
        model_cnn.compile(loss='mse', optimizer=adam)
        model_cnn.summary()
        return model_cnn
