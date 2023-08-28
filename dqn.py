from tictactoe import TicTacToe2D
from opponents import *
import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQN:
    def __init__(self, actions=9, game=None, alpha=0.01, gamma=0.95, epsilon=0.99, epsilonMultiplier=0.9995,
                 randomEpisodes=5000, bufferCapacity=100, sampleSize=5, mainUpdateFreq=10, targetUpdateFreq=200):
        self.actions = actions
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMultiplier = epsilonMultiplier
        self.randomEpisodes = randomEpisodes
        self.replayBuffer = deque([], maxlen=bufferCapacity)
        self.mainModel, self.targetModel = self.build_model()
        self.sampleSize = sampleSize
        self.mainUpdateFreq = mainUpdateFreq
        self.targetUpdateFreq = targetUpdateFreq

    # maybe we can make this accept a list as a parameter to avoid hard-coding the network?
    def build_model(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Dense(units=24, activation='relu'))
        model.add(tf.keras.layers.Dense(units=24, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.actions))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha),
                      loss='mean_square_error',
                      metrics=['mae'])
        # print(model.summary())
        return model, tf.keras.models.clone_model(model)

    def bestLegalMoveReward(self, state):
        legalMoves = self.game.getPossibleActions(state=state, flatten=True)

        # find Q-values for all possible moves (including illegal ones)
        predQ = self.mainModel.predict(np.expand_dims(state, axis=0))[0]

        # finds the Q-value for the best legal move and turns it into a board coordinate
        action = np.unravel_index(legalMoves[np.argmax(predQ[legalMoves])], shape=self.game.board.shape)
        return action, predQ[action]

    def chooseMove(self, state, episode=0):
        if episode < self.randomEpisodes:
            return self.game.chooseRandomAction()

        self.epsilon *= self.epsilonMultiplier

        if np.random.rand() < self.epsilon:
            return self.game.chooseRandomMove()

        return self.bestLegalMoveReward(state)[0]

    # Trains the model for the specified number of episodes
    # If an opponent is not specified, the opponent will make a random legal move each time
    def train(self, episodes, opponent=None):
        if opponent is None:
            opponent = RandomPlayer(self.game)
        
        wins = 0
        draws = 0
        losses = 0
        step = 0
        
        for episode in range(episodes):
            self.game.reset() # reset the game after each episode
            episodeDone = False
            
            # alternate between x and o
            if episode % 2 == 0:
                self.game.place(opponent.chooseMove())
            
            state = self.game.getState()
            while not episodeDone:
                step += 1
                action = self.chooseMove(state, episode)

                # take game step
                newState, reward, episodeDone = self.game.step(action, opponent)

                # update replay buffer
                self.replayBuffer.append((state, action, reward, newState, episodeDone))

                state = newState

                # update stats
                if episodeDone:
                    if reward == 1:
                        wins += 1
                    elif reward == 0:
                        draws += 1
                    else:
                        losses += 1

                if len(self.replayBuffer) == self.replayBuffer.maxlen:
                    if step % self.mainUpdateFreq == 0:
                        batch = random.sample(self.replayBuffer, self.sampleSize)
                        states, actions, rewards, nextStates, episodeDones = zip(*batch)
                        target_queue_values = []
                        for i in range(self.sampleSize):
                            if episodeDones[i]:
                                target_queue_values.append(rewards[i])
                            else:
                                nextReward = self.bestLegalMoveReward(states[i])
                                target_queue_values.append(rewards[i] + self.gamma * nextReward)

                        states = tf.convert_to_tensor(states, dtype=np.float32)
                        target_queue_values = tf.convert_to_tensor(target_queue_values, dtype=np.float32)

                        with tf.GradientTape() as tape:
                            q_values = self.mainModel(states)
                            actions = tf.convert_to_tensor(actions.unravel_index)
                            loss = tf.reduce_mean(tf.square())


                    if step % self.targetUpdateFreq == 0:
                        self.targetModel = tf.keras.models.clone_model(self.mainModel)

            
        print(f'Wins={wins/episodes}, Draws={draws/episodes}, Losses={losses/episodes}, Epsilon={self.epsilon}')


if __name__ == "__main__":
    game = TicTacToe2D()
    myDQN = DQN(game=game, epsilon=0)
    myDQN.train(20)