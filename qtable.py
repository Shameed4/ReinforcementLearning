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
            # dimSize^dims for the state, and dims for the actions
            self.table = np.zeros((game.dimSize,) * (int(game.dimSize ** game.dims) + game.dims))
        else:
            self.table = table

    # Places and returns a move based on the epsilon-greedy algorithm
    # In the beginning, it will initially pick moves that are ENTIRELY random to set up the Q-table
    # Later, it will choose between random and its best move
    # Episode - The number of episodes that have been simulated
    def pickMove(self, episode=0):
        if episode < self.randomEpisodes:
            return self.game.pickRandomAction()
        
        self.epsilon *= self.epsilonMultiplier
        
        if np.random.rand() < self.epsilon:
            return self.game.pickRandomAction()
        
        actions = self.game.getPossibleActions()
        rewards = self.table[self.game.getState()][tuple(zip(*actions))] # numpy cannot index by a list of tuples so need to convert to tuple of lists
        
        move = np.argmax(rewards)
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
            
            if episode % 2 == 0:
                opponent.pickMove()
            
            while not episodeDone:
                # alternate between X and O
                s = self.game.getState()
                a = self.pickMove(episode)

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
    
    def cloneTable(self, epsilon=0.75, epsilonMultiplier=1):
        return QLearning(game=self.game, randomEpisodes=0, epsilon=epsilon, epsilonMultiplier=epsilonMultiplier, table=np.copy(self.table))


if __name__ == "__main__":
    game = TicTacToe2D()
    model = QLearning(game, epsilonMultiplier=0.999995, randomEpisodes=10000, alpha=0.02, gamma=0.95)
    
    print("Random")
    model.train(20000)
    model.randomEpisodes = 0
    
    for epsilon in range(100):
        print("Remaining rounds:", 50 - epsilon, model.alpha)
        print("Clone")
        model.train(5000, opponent=model)
        print("Random")
        model.train(5000)
        model.alpha = model.alpha * 0.95 + 0.0001
    
    # playing against human
    model.epsilon = 0

    model.train(5, HumanPlayer(game))