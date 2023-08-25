from tictactoe import TicTacToe2D
from opponents import *
import numpy as np
import tensorflow as tf
from replaybuffer import ReplayBuffer

class DQN:
    def __init__(self, states=3**9, actions=9, game=None, alpha=0.01, gamma=0.01, epsilon=0.99, epsilonMultiplier=0.9995, randomEpisodes=5000, capacity=5000):
        self.states = states # is this needed?
        self.actions = actions
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMultiplier = epsilonMultiplier
        self.randomEpisodes = randomEpisodes
        self.replayBuffer = ReplayBuffer(capacity) # can this be replaced with a list or Queue?

        self.model = self.build_model()

    # maybe we can make this accept a list as a parameter to avoid hard-coding the network?
    def build_model(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Flatten(input_shape=(self.game.board.shape)))
        model.add(tf.keras.layers.Dense(units=24, activation='relu'))
        model.add(tf.keras.layers.Dense(units=24, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.actions))

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.alpha),
                      loss='mean_square_error',
                      metrics=['mae'])

        return model

    def pickMove(self, state, episode=0):
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

        pred = self.model.predict(state)
        move = np.argmax(pred[actions])
        self.game.place(actions[move])
        return actions[move]

    def train(self, episodes, opponent=None):
        if opponent is None:
            opponent = RandomPlayer(self.game)
        
        wins = 0
        draws = 0
        losses = 0
        
        for episode in range(episodes):
            self.game.reset() # reset the game after each episode
            episodeDone = False
            
            if episode % 2 == 0:
                opponent.pickMove()
            
            while not episodeDone:
                # alternate between X and O
                s = self.game.getState()
                a = self.pickMove(episode, s)

                curr_q = self.table[s][a] # index the q table with the current state and action

                if self.game.gameOver == True: # max reward when game is won
                    reward = 1
                    episodeDone = True
                    wins += 1
                elif self.game.remainingTurns == 0: # no reward when there is a draw
                    reward = 0
                    episodeDone = True
                    draws += 1
                else:          
                    opponent.pickMove()
                    new_s = self.game.getState()

                    if self.game.gameOver == True: # max punishment when game is lost
                        reward = -1
                        episodeDone = True
                        losses += 1
                    elif self.game.remainingTurns == 0: # no reward when there is a draw
                        reward = 0
                        episodeDone = True
                        draws += 1
                    else: # otherwise no reward
                        reward = 0

                new_q = curr_q + self.alpha * (reward + self.gamma * np.max(self.table[new_s]) - curr_q)
                self.table[s][a] = new_q

                if episodeDone:
                    break
            
        print(f'Wins={wins/episodes}, Draws={draws/episodes}, Losses={losses/episodes}, Epsilon={self.epsilon}')

    def updateBuffer(self, state, action, reward, next_state, done):
        self.replayBuffer.add(state, action, reward, next_state, done)
