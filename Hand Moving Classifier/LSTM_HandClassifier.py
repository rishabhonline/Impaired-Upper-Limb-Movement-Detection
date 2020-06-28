import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, LSTM
from keras.layers.core import Activation, Dropout, Dense
import numpy as np
import train_data_load as load

class MovingHandModel:
  
  def __init__ (self, X, Y, epochs=10, validation_split=0.2, train = False):
    
    self.batch_size = X.shape[0]
    self.timestep = X.shape[1]
    self.num_features = X.shape[2]

    self.epochs = epochs
    self.model = Sequential()
    self.validation_split = validation_split

    self.train = train

    self.X = X
    self.Y = Y
  
  def build_model(self):

    self.model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(self.timestep, self.num_features)))
    self.model.add(LSTM(100, activation='relu', return_sequences=True))
    self.model.add(LSTM(50, activation='relu', return_sequences=True))
    self.model.add(LSTM(25, activation='relu'))
    self.model.add(Dense(20, activation='relu'))
    self.model.add(Dense(10, activation='relu'))
    self.model.add(Dense(1))
    self.model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.Accuracy()])

  def train_model(self):
    history = self.model.fit(self.X, self.Y, epochs=self.epochs, validation_split=self.validation_split, verbose=1)

  def print_parameters(self):
    print('BATCH_SIZE : {} ; TIME_STEPS : {} ; FEATURES_SIZE : {}'.format(self.batch_size, self.timestep, self.num_features))
    
    self.build_model()
    print(self.model.summary())

    if(self.train):
      self.train_model()
    
def main():
	[X, Y] = load.preprocess()

	obj = MovingHandModel(X,Y, validation_split=0.3, epochs=10)
	obj.print_parameters()	

if __name__ == '__main__':
	main()