import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM

lookBack = 28
epochs = 25
metricsDivider = 100000

def main():
    lastCoursesDataFrame = pd.read_csv('files/BTC-USD.csv')
    lastCourses = lastCoursesDataFrame.filter(['Close'])
    lastCourses = np.array(lastCourses) / metricsDivider
    trainX = []
    trainY = []
    for i in range(lookBack, len(lastCourses)):
        trainX.append(lastCourses[i-lookBack:i, 0])
        trainY.append(lastCourses[i, 0])
    trainX = np.array(trainX)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    trainY = np.array(trainY)

    lstmnnModel = Sequential()
    lstmnnModel.add(LSTM(50, return_sequences=True, input_shape=(trainX.shape[1], 1)))
    lstmnnModel.add(LSTM(50, return_sequences=False))
    lstmnnModel.add(Dense(25))
    lstmnnModel.add(Dense(1))
    lstmnnModel.compile(optimizer='adam', loss='mean_squared_error')

    lstmnnModel.fit(trainX, trainY, batch_size=1, epochs=epochs)

    lstmnnModelFile = open('files/lstmnnModel.json', 'w')
    lstmnnModelFile.write(lstmnnModel.to_json())
    lstmnnModelFile.close()
    lstmnnModel.save_weights('files/lstmnnModel.h5')

main()