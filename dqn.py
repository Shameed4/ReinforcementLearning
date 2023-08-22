from tictactoe import TicTacToe2D
from opponents import *
import numpy as np
import tensorflow as tf
from replaybuffer import ReplayBuffer

class DQN:
    def __init__(self, states=3**9, actions=9, game=None, alpha=0.01, gamma=0.01, epsilon=0.99, epsilonMultiplier=0.9995, randomEpisodes=5000, capacity=5000):
        self.states = states
        self.actions = actions
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMultiplier = epsilonMultiplier
        self.randomEpisodes = randomEpisodes
        self.replayBuffer = ReplayBuffer(capacity)

    def build_model(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Flatten(input_shape=(3,3)))
        model.add(tf.keras.layers.Dense(units=24,activation='relu'))
        model.add(tf.keras.layers.Dense(units=24, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.actions))

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.alpha),
                      loss='mean_square_error',
                      metrics=['mae'])

        return model

    def pickMove(self, episode=0):
        actions = self.game.getPossibleActions()

        if episode < self.randomEpisodes:
            move = np.random.choice(actions)
            game.place(move)
            return move

        self.epsilon *= self.epsilonMultiplier

        if np.random.rand() < self.epsilon:
            move = np.random.choice(actions)
            game.place(move)
            return move

        rewards = self.table[self.game.getState()][
            tuple(zip(*actions))]  # numpy cannot index by a list of tuples so need to convert to tuple of lists
        move = np.argmax(rewards)
        self.game.place(actions[move])
        return actions[move]

    def train(self):
        pass

    def updateBuffer(self, state, action, reward, next_state, done):
        self.replayBuffer.add(state, action, reward, next_state, done)
