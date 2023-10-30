import keras

from tictactoe import TicTacToe2D
from opponents import *
import numpy as np
import tensorflow as tf
from collections import deque
import random

print("Tensorflow", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

class DQN:
    def __init__(self, actions=9, game=None, alpha=0.02, gamma=0.95, epsilon=0.99, epsilon_multiplier=0.9995,
                 buffer_capacity=10000, batch_size=32, main_update_freq=200, target_update_freq=1000):
        self.actions = actions
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_multiplier = epsilon_multiplier
        self.replay_buffer = deque([], maxlen=buffer_capacity)
        self.batch_size = batch_size
        self.mainModel, self.targetModel = self.build_model()
        self.main_update_freq = main_update_freq
        self.target_update_freq = target_update_freq

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

    def best_move(self, state):
        legal_moves = self.game.get_possible_actions(state=state)

        # find Q-values for all possible moves (including illegal ones)
        pred_Q = self.mainModel.predict(np.expand_dims(state, axis=0))[0]

        # finds the Q-value for the best legal move and turns it into a board coordinate
        return legal_moves[np.argmax(pred_Q[legal_moves])]

    def choose_move(self, episode=0):
        self.epsilon *= self.epsilon_multiplier
        if np.random.rand() < self.epsilon:
            return self.game.choose_random_action()

        return self.best_move(self.game.get_state())

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
            episode_done = False
            
            # alternate between x and o
            if episode % 2 == 0:
                self.game.place(opponent.choose_move())
            
            state = self.game.get_state()
            while not episode_done:
                step += 1
                action = self.choose_move(episode)

                # take game step
                new_state, reward, episode_done = self.game.step(action, opponent)

                # update replay buffer
                self.replay_buffer.append((state, action, reward, new_state, episode_done))

                state = new_state

                # update stats
                if episode_done:
                    if reward == self.game.win_reward:
                        wins += 1
                    elif reward == self.game.draw_reward:
                        draws += 1
                    else:
                        losses += 1

                if step % self.main_update_freq == 0:
                    self.replayUpdate(self.replay_buffer, self.batch_size)

                if step % self.target_update_freq == 0:
                    self.updateTarget()
        
        print(f'Wins={wins/episodes}, Draws={draws/episodes}, Losses={losses/episodes}, Epsilon={self.epsilon}')

    def updateTarget(self):
        main_weights = self.mainModel.get_weights()
        self.targetModel.set_weights(main_weights)

    def replayUpdate(self, replay_buffer, batch_size):
        batch = random.sample(replay_buffer, batch_size)
        states, actions, rewards, next_states, episode_dones = zip(*batch)
        targets = np.array(rewards)
        current_state_q_values = [ ]
        for i in range(len(batch)):
            if not episode_dones[i]:
                # use the target network to predict the next state q-values
                next_state_q_value = self.targetModel.predict(np.expand_dims(next_states[i], axis=0))[0]
                targets[i] += self.gamma * np.max(next_state_q_value)

            # use the main network to predict the current state q-values
            current_state_q_value = self.mainModel.predict(np.expand_dims(states[i], axis=0))[0]

            # update the q-value for the chosen action with target
            current_state_q_value[actions[i]] = targets[i]
            current_state_q_values.append(current_state_q_value)

        self.mainModel.fit(np.array(states), np.array(current_state_q_values))

    def save(self):
        self.mainModel.save_weights("dqnModel.weights.h5")

    def load(self):
        self.mainModel.load_weights("dqnModel.weights.h5")
        self.targetModel.load_weights("dqnModel.weights.h5")


if __name__ == "__main__":
    game = TicTacToe2D()
    myDQN = DQN(game=game)
    while input("Keep training? (Y|N)") != "N":
        myDQN.train(1000)
    else:
        pass

