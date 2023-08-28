from tictactoe import TicTacToe2D
from opponents import *
import numpy as np

class QLearning:
    # alpha - learning rate
    # gamma - discount rate
    # epsilon - greedy-epsilon parameter
    # numEpisodes - number of simulated episodes
    def __init__(self, game, alpha=0.01, gamma=0.95, epsilon=0.99, epsilonMultiplier=0.9995, randomEpisodes=5000, table=None):
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMultiplier = epsilonMultiplier
        self.randomEpisodes = randomEpisodes

        if table is None:
            # (numPlayers+1)^boardSize for the state, and boardSize for action
            self.table = np.zeros((game.numPlayers + 1,) * game.board.size + (game.board.size,))
        else:
            self.table = table

    # Chooses and returns a move based on the epsilon-greedy algorithm (without placing it)
    # In the beginning, it will initially choose moves that are ENTIRELY random to set up the Q-table
    # Later, it will choose between random and its best move
    # Episode - The number of episodes that have been simulated
    def chooseMove(self, episode=0):
        if episode < self.randomEpisodes:
            return self.game.chooseRandomAction()
        
        self.epsilon *= self.epsilonMultiplier
        
        if np.random.rand() < self.epsilon:
            return self.game.chooseRandomAction()
        
        actions = self.game.getPossibleActions()
        rewards = self.table[self.game.getState()][actions]
        
        move = np.argmax(rewards)
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
                action = self.chooseMove(episode)
                predQ = self.table[state][action] # index the q table with the current state and action
                
                # take game step
                newState, reward, episodeDone = self.game.step(action, opponent)

                # update Q values
                new_q = predQ + self.alpha * (reward + self.gamma * np.max(self.table[newState]) - predQ)
                self.table[state][action] = new_q

                # update state
                state = newState

                if episodeDone:
                    if reward == 1:
                        wins += 1
                    elif reward == 0:
                        draws += 1
                    else:
                        losses += 1
            
        print(f'Wins={wins/episodes}, Draws={draws/episodes}, Losses={losses/episodes}, Epsilon={self.epsilon}')
    
    def cloneTable(self, epsilon=0.75, epsilonMultiplier=1):
        return QLearning(game=self.game, randomEpisodes=0, epsilon=epsilon, epsilonMultiplier=epsilonMultiplier, table=np.copy(self.table))


if __name__ == "__main__":
    game = TicTacToe2D()
    model = QLearning(game, epsilonMultiplier=0.999995, randomEpisodes=10000, alpha=0.02, gamma=0.95)
    
    print("Random")
    model.train(20000, opponent=RandomPlayer(game))
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