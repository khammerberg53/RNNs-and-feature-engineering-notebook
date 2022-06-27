from keras.layers.recurrent import RNN
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error
import numpy as np
import yfinance as yf
import pandas as pd
import seaborn as sns
import datetime
import math
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

# defining functions

def convert2matrix(data_arr, look_back):
   X, Y =[], []
   for i in range(len(data_arr)-look_back):
    d=i+look_back  
    X.append(data_arr[i:d,])
    Y.append(data_arr[d,])
   return np.array(X), np.array(Y)

import warnings

warnings.simplefilter("ignore")

ticker_list = ['SPY','QQQ', '^BCOM','XLF','XLE','XLV','XLU','XLRE']

df = pd.DataFrame(columns = ['Ticker','RMSE'])


def model_rnn(look_back):
  model=Sequential()
  model.add(Dense(units=64, input_dim=(look_back), activation="relu"))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(256, activation='relu'))
  model.add(Dense(128))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error',  optimizer='adam',metrics = ['mse', 'mae'])
  return model

for t in ticker_list:
    data = yf.Ticker(t)
    prices = data.history(start='2017-01-01', end='2022-01-03').Close
    returns = prices.pct_change().dropna()
    
    #Split data set into train test and val sets
    #Split data set into train test and val sets
    train_size = math.floor(len(returns)*0.6)
    train = returns.values[0:train_size]
    test = returns.values[train_size:len(returns.values)]
    # setup look_back window 
    look_back = 15
    #convert dataset into right shape in order to input into the DNN
    trainX, trainY = convert2matrix(train, look_back)
    testX, testY = convert2matrix(test, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1]))

    
    model=model_rnn(look_back)
    history = model.fit(trainX, trainY, epochs=20, batch_size=30, verbose=False, validation_data=(testX,testY), callbacks=[EarlyStopping(monitor="val_loss", patience=100)], shuffle=False)
    train_score = model.evaluate(trainX, trainY, verbose=0)
    #print('Train Root Mean Squared Error(RMSE): %.4f; Train Mean Absolute Error(MAE) : %.4f ' 
    #% (np.sqrt(train_score[1]), train_score[2]))
    test_score = model.evaluate(testX, testY, verbose=0)
    #print('Test Root Mean Squared Error(RMSE): %.4f; Test Mean Absolute Error(MAE) : %.4f ' 
    #% (np.sqrt(test_score[1]), test_score[2]))
    #model_loss(history)
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)
    #prediction_plot(testY, test_predict)
    rmse = np.sqrt(mean_squared_error(testY, test_predict))
    print(f'{t} rmse is {rmse}')

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)