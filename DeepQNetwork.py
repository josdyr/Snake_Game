from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add



class DQNAgent():

    def __init__(self):
        self.current_reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame() # ???
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.model = self.network()
        # self.model = self.network("weights.hdf5")
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def neural_network(self, weights=None):
        self.model = Sequential()
        self.model.add(Dense(output_dim=120, activation='relu', input_dim=9))
        self.model.add(Dropout(0.15))
        self.model.add(Dense(output_dim=120, activation='relu'))
        self.model.add(Dropout(0.15))
        self.model.add(Dense(output_dim=120, activation='relu'))
        self.model.add(Dropout(0.15))
        self.model.add(Dense(output_dim=3, activation='softmax'))
        self.opt = Adam(self.learning_rate)
        self.model.compile(loss='mse', optimizer=opt)
