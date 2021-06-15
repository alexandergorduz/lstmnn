import numpy as np
import pandas as pd
import pandas_datareader as pdr
import requests as rq
import config
from keras.models import model_from_json

lookBack = 28
epochs = 25
metricsDivider = 100000
botToken = config.botToken

def main():
    lastCoursesDataFrame = pd.read_csv('files/BTC-USD.csv')
    lastCoursesSamples = lastCoursesDataFrame.tail(lookBack)
    trainX = lastCoursesSamples.filter(['Close'])
    trainX = np.array(trainX) / metricsDivider
    trainX = np.reshape(trainX, (1, trainX.shape[0], trainX.shape[1]))

    actualCoursesSamples = pdr.DataReader('BTC-USD', 'yahoo')
    actualCourseSample = actualCoursesSamples.tail(2).iloc[[0]]
    trainY = actualCourseSample.filter(['Close'])
    trainY = np.array(trainY) / metricsDivider
    trainY = trainY[:, 0]

    lastDayCourse = actualCourseSample.filter(['Close'])
    lastDayCourse = round(np.array(lastDayCourse)[0][0], 2)

    lstmnnModelFile = open('files/lstmnnModel.json', 'r')
    lstmnnModel = model_from_json(lstmnnModelFile.read())
    lstmnnModelFile.close()
    lstmnnModel.load_weights('files/lstmnnModel.h5')
    lstmnnModel.compile(optimizer='adam', loss='mean_squared_error')

    lstmnnModel.fit(trainX, trainY, batch_size=1, epochs=epochs)

    lstmnnModelFile = open('files/lstmnnModel.json', 'w')
    lstmnnModelFile.write(lstmnnModel.to_json())
    lstmnnModelFile.close()
    lstmnnModel.save_weights('files/lstmnnModel.h5')

    lastCoursesDataFrame = lastCoursesDataFrame.append(actualCourseSample)
    lastCoursesDataFrame.to_csv('files/BTC-USD.csv', index=False)

    lastCoursesDataFrame = pd.read_csv('files/BTC-USD.csv')
    lastCoursesSamples = lastCoursesDataFrame.tail(lookBack)
    predictionX = lastCoursesSamples.filter(['Close'])
    predictionX = np.array(predictionX) / metricsDivider
    predictionX = np.reshape(predictionX, (1, predictionX.shape[0], predictionX.shape[1]))

    predictionY = lstmnnModel.predict(predictionX)
    predictionY = round(predictionY[0, 0] * metricsDivider, 2)

    if predictionY - lastDayCourse > 0:
        botTextMessage = 'Привет!\nПрогноз курса на сегодня:\n1BTC = {0}USD\nРост ⬆️ на {1}%'.format(predictionY, round(100*(predictionY-lastDayCourse)/lastDayCourse, 2))
    else:
        botTextMessage = 'Привет!\nПрогноз курса на сегодня:\n1BTC = {0}USD\nПадение ⬇️ на {1}%'.format(predictionY, round(100*(lastDayCourse-predictionY)/lastDayCourse, 2))

    rq.get('https://api.telegram.org/bot{0}/sendMessage'.format(botToken), params=dict(chat_id='284449448', text=botTextMessage))

main()