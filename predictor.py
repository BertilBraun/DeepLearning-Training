from pandas_datareader import data

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

start_date = '2010-01-01'
end_date = '2019-08-12'

def show(title, close):
    close = close.reindex(pd.date_range(start=start_date, end=end_date, freq='B'))
    close = close.fillna(method='ffill')

    short_rolling_msft = close.rolling(window=20).mean()
    long_rolling_msft = close.rolling(window=100).mean()

    plt.title(title)
    plt.plot(close)
    plt.plot(short_rolling_msft, label='20 days rolling')
    plt.plot(long_rolling_msft, label='100 days rolling')
    plt.legend(['No rolling', '20 days rolling', '100 days rolling'])
    plt.xlabel('Date')
    plt.ylabel('Adjusted closing price ($)')
    plt.show()


def loop(model, close):
    workingRange = close[:500]

    for offset in range(500, len(close)):

        workingRange = workingRange[-500:]
        x = [np.array((workingRange))]

        predict = model.predict(np.array(x))[0][0]
        actual = close[offset]
        workingRange.add(actual)

        off = abs(actual - predict)

        if (off > 1):
            print("UFF : " + str(off))
        else:
            print("YAAAS : " + str(off))


def makeModel():
    model = Sequential()
    model.add(Dense(500, input_shape=(500,), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(500, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(500, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='softmax'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    return model

def train(model, close):
    X_train = []
    y_train = []

    for offset in range(500, len(close) - 1):
        x = close[offset-500:offset]
        y = [close[offset]]

        X_train.append(np.array(x))
        y_train.append(np.array(y))

        # print("Added x val: " + str(close[offset-500:offset][499]))
        # print("Close x val: " + str(close[offset - 1]))
        # print("Y val: " + str(y_train[len(y_train) - 1][0]))
        # print("Close y val: " + str(close[offset]))
        # print("")

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print("Now Training")
    model.fit(X_train, y_train, shuffle=True, epochs=5000, batch_size=200, verbose=2)
    print("Finished Training")

def main():
    close = data.DataReader('AAPL', 'yahoo', start_date, end_date)['Close']

    # show('APPL', close)

    model = makeModel()
    # model.summary()
    path = "test.model"

    #TODO
    if (os.path.exists(path)):
        os.remove(path)

    if (not os.path.exists(path)):
        train(model, close)
        model.save_weights(path)
    else:
        model.load_weights(path)

    #TODO
    # loop(model, close)

if (__name__ == "__main__"):
    main()