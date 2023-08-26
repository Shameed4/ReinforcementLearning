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

        model.add(tf.keras.layers.Flatten(input_shape=self.game.board.shape))
        model.add(tf.keras.layers.Dense(units=24, activation='relu'))
        model.add(tf.keras.layers.Dense(units=24, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.actions))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha),
                      loss='mean_square_error',
                      metrics=['mae'])
        # print(model.summary())
        return model

    def chooseMove(self, state, episode=0):
        if episode < self.randomEpisodes:
            return self.game.chooseRandomAction()

        self.epsilon *= self.epsilonMultiplier

        if np.random.rand() < self.epsilon:
            return self.game.chooseRandomMove()

        actions = self.game.getPossibleActions()
        pred = self.model.predict(state)
        move = np.argmax(pred[np.unravel_index(actions, shape=self.game.board.shape)])
        self.game.place(actions[move])
        return actions[move]

    # Trains the model for the specified number of episodes
    # If an opponent is not specified, the opponent will make a random legal move each time
    def train(self, episodes, opponent=None):
        if opponent is None:
            opponent = RandomPlayer(self.game)
        
        wins = 0
        draws = 0
        losses = 0
        
        for episode in range(episodes):
            self.game.reset() # reset the game after each episode
            episodeDone = False
            
            # alternate between x and o
            if episode % 2 == 0:
                self.game.place(opponent.chooseMove())
            
            state = self.game.getState()
            while not episodeDone:
                legalMoves = self.game.getPossibleActions(flatten=True)
                print("Legal moves", legalMoves)

                predQ = self.model.predict(np.expand_dims(self.game.board, axis=0))[0]  # index the q table with the current state and action
                print("Predictions:", predQ)
                action = np.unravel_index(legalMoves[np.argmax(predQ[legalMoves])], shape=self.game.board.shape)

                # take game step
                newState, reward, episodeDone = self.game.step(action, opponent)

                # # update Q values
                # new_q = predQ + self.alpha * (reward + self.gamma * np.max(self.table[newState]) - predQ)
                # self.table[state][action] = new_q
                #
                # # update state
                state = newState

                # update stats
                if episodeDone:
                    if reward == 1:
                        wins += 1
                    elif reward == 0:
                        draws += 1
                    else:
                        losses += 1
            
        print(f'Wins={wins/episodes}, Draws={draws/episodes}, Losses={losses/episodes}, Epsilon={self.epsilon}')

    def updateBuffer(self, state, action, reward, next_state, done):
        self.replayBuffer.add(state, action, reward, next_state, done)


if __name__ == "__main__":
    game = TicTacToe2D()
    myDQN = DQN(game=game)
    myDQN.train(1)