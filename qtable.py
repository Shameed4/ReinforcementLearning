from tictactoe import TicTacToe2D
from opponents import *
import numpy as np

class QLearning:
    # alpha - learning rate
    # gamma - discount rate
    # epsilon - greedy-epsilon parameter
    # numEpisodes - number of simulated episodes
    def __init__(self, game=None, alpha=0.01, gamma=0.01, epsilon=0.99, epsilonMultiplier=0.9995, randomEpisodes=5000, table=None):       
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMultiplier = epsilonMultiplier
        self.randomEpisodes = randomEpisodes

        if table is None:
            self.table = np.zeros((game.dimSize,) * 11) # 9 for state, 2 for action

    # Places and returns a move based on the epsilon-greedy algorithm
    # In the beginning, it will initially pick moves that are ENTIRELY random to set up the Q-table
    # Later, it will choose between random and its best move
    # Episode - The number of episodes that have been simulated
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
        
        rewards = self.table[self.game.getState()][tuple(zip(*actions))] # numpy cannot index by a list of tuples so need to convert to tuple of lists
        move = np.argmax(rewards)
        self.game.place(actions[move])
        return actions[move]
    
    def train(self, episodes, opponent=None):
        if opponent is None:
            opponent = RandomPlayer(self.game)
        for episode in range(episodes):
            self.game.reset() # reset the game after each episode
            episodeDone = False

            while not episodeDone:
                # alternate between X and O
                if episode % 2 == 0:
                    opponent.pickMove()
                
                s = self.game.getState()
                a = self.pickMove(episode)

                curr_q = self.table[s][a] # index the q table with the current state and action

                if self.game.gameOver == True: # max reward when game is won
                    reward = 1
                    episodeDone = True
                elif self.game.remainingTurns == 0: # no reward when there is a draw
                    reward = 0
                    episodeDone = True
                else:          
                    opponent.pickMove()
                    new_s = self.game.getState()

                    if self.game.gameOver == True: # max punishment when game is lost
                        reward = 1
                        episodeDone = True
                    elif self.game.remainingTurns == 0: # no reward when there is a draw
                        reward = 0
                        episodeDone = True
                    else: # otherwise no reward
                        reward = 0

                new_q = curr_q + self.alpha * (reward + self.gamma * np.max(self.table[new_s]) - curr_q)
                self.table[s][a] = new_q

                if episodeDone:
                    break
    
    def cloneTable(self, epsilon=0.75, epsilonMultiplier=1):
        return QLearning(game=self.game, epsilon=epsilon, epsilonMultiplier=epsilonMultiplier, table=np.copy(self.table))


if __name__ == "__main__":
    game = TicTacToe2D()
    model = QLearning(game, epsilonMultiplier=0.99999)
    model.train(10000)
    print("After first round epsilon: ", model.epsilon)
    model.train(10000, opponent=model.cloneTable())
    print("After second round epsilon: ", model.epsilon)
    model.train(10000, opponent=model.cloneTable(epsilon=0.5))
    print("After third round epsilon: ", model.epsilon)
    model.train(10000, opponent=model.cloneTable(epsilon=0.2))
    model.epsilon = 0
    model.train(5, HumanPlayer(game))