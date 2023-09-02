from tictactoe import TicTacToe2D
from opponents import *
import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQN:
    def __init__(self, actions=9, game=None, alpha=0.01, gamma=0.95, epsilon=0.99, epsilonMultiplier=0.9995,
                 randomEpisodes=5000, bufferCapacity=10000, batchSize=32, mainUpdateFreq=200, targetUpdateFreq=1000):
        self.actions = actions
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMultiplier = epsilonMultiplier
        self.randomEpisodes = randomEpisodes
        self.replayBuffer = deque([], maxlen=bufferCapacity)
        self.mainModel, self.targetModel = self.build_model()
        self.batchSize = batchSize
        self.mainUpdateFreq = mainUpdateFreq
        self.targetUpdateFreq = targetUpdateFreq

    # maybe we can make this accept a list as a parameter to avoid hard-coding the network?
    def build_model(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Dense(units=24, activation='relu'))
        model.add(tf.keras.layers.Dense(units=24, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.actions))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha),
                      loss='mean_squared_error',
                      metrics=['mae'])

        # print(model.summary())
        return model, tf.keras.models.clone_model(model)

    def bestLegalMoveReward(self, state):
        legalMoves = self.game.getPossibleActions(state=state)

        # find Q-values for all possible moves (including illegal ones)
        predQ = self.mainModel.predict(np.expand_dims(state, axis=0))[0]

        # finds the Q-value for the best legal move and turns it into a board coordinate
        action = legalMoves[np.argmax(predQ[legalMoves])]
        return action, predQ[action]

    def chooseMove(self, state, episode=0):
        if episode < self.randomEpisodes:
            return self.game.chooseRandomAction()

        self.epsilon *= self.epsilonMultiplier

        if np.random.rand() < self.epsilon:
            return self.game.chooseRandomAction()

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
            if episode % 100 == 0:
                print(episode)
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
                    if reward == self.game.winReward:
                        wins += 1
                    elif reward == self.game.drawReward:
                        draws += 1
                    else:
                        losses += 1

                if step % self.mainUpdateFreq == 0:
                    self.replayUpdate(self.replayBuffer, self.batchSize)

                if step % self.targetUpdateFreq == 0:
                    self.updateTarget()
        
        print(f'Wins={wins/episodes}, Draws={draws/episodes}, Losses={losses/episodes}, Epsilon={self.epsilon}')

    def updateTarget(self):
        main_weights = self.mainModel.get_weights()
        self.targetModel.set_weights(main_weights)

    def replayUpdate(self, replayBuffer, batchSize):
        batch = random.sample(replayBuffer, batchSize)
        states, actions, rewards, nextStates, episodeDones = zip(*batch)
        targets = np.array(rewards)
        current_state_q_values = [ ]
        for i in range(len(batch)):
            if not episodeDones[i]:
                # use the target network to predict the next state q-values
                next_state_q_value = self.targetModel.predict(np.expand_dims(nextStates[i], axis=0))[0]
                targets[i] += self.gamma * np.max(next_state_q_value)

            # use the main network to predict the current state q-values
            current_state_q_value = self.mainModel.predict(np.expand_dims(states[i], axis=0))[0]

            # update the q-value for the chosen action with target
            current_state_q_value[actions[i]] = targets[i]
            current_state_q_values.append(current_state_q_value)


        self.mainModel.fit(np.array(states), np.array(current_state_q_values))
            
if __name__ == "__main__":
    game = TicTacToe2D()
    myDQN = DQN(game=game)
    myDQN.train(20000)
    myDQN.train(20000)
    myDQN.train(20000)