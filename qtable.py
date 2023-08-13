from tictactoe import TicTacToe
import numpy as np

class QLearning:
    # alpha - learning rate
    # gamma - discount rate
    # epsilon - greedy-epsilon parameter
    # numEpisodes - number of simulated episodes
    def __init__(self, game=None, alpha=0.01, gamma=0.01, epsilon=0.99, epsilonMultiplier=0.999, randomEpisodes=500, stimulatedEpisodes=2000):       
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMultiplier = epsilonMultiplier
        self.randomEpisodes = randomEpisodes
        self.stimulatedEpisodes = stimulatedEpisodes

        self.table = np.zeros((3,) * 9)


    # Returns the index to use when looking up the reward in the Q-table
    # move - Returns the index if the current player placed their move there. 
    # If move is not specified, it returns the index at the current position.
    def getState(self, move=None):
        ret = self.game.board.flatten()
        if move is None:
            return tuple(ret)
        ret[move] = game.turn
        return tuple(ret)
        
    # Returns the list of possible positions to move at a given position
    def getPossibleActions(self):
        ret = np.zeros(game.remainingTurns, dtype=object)
        counter = 0
        for i in range(3):
            for j in range(3):
                if game.board[i, j] == -1:
                    ret[counter] = (i, j)
                    counter += 1
        
        return ret
    

    # Selects a move based on the epsilon-greedy algorithm
    # In the beginning, it will initially pick moves that are ENTIRELY random to set up the Q-table
    # Later, it will choose between random and its best move
    # Episode - The number of episodes that have been simulated
    def selectMove(self, episode=0):
        actions = self.getPossibleActions()
        print("Actions", actions)

        if episode < self.randomEpisodes:
            return np.random.choice(actions)
        
        self.epsilon *= self.epsilonMultiplier
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)
        

        rewards = np.array(self.table(i) for i in actions)
        return actions[np.argmax(rewards)]
    
    def train(self, episodes):
        for episode in range(episodes):
            self.game.reset() # reset the game after each episode
            episodeDone = False
            while not episodeDone:
                s = self.getState()
                a = self.selectMove(episode)

                curr_q = self.table[s] # index the q table with the current state

                self.game.place(a)
                new_s = self.getState()

                if self.game.gameOver == True: # max reward when game is won
                    reward = 1
                    episodeDone = True
                elif self.game.remainingTurns == 0: # half reward when there is a draw
                    reward = 0.5
                    episodeDone = True
                else: # otherwise no reward
                    reward = 0

                new_q = curr_q + self.alpha * (reward + self.gamma * np.max(self.table[new_s]) - curr_q)
                self.table[s] = new_q

                if episodeDone:
                    break

if __name__ == "__main__":
    game = TicTacToe()
    model = QLearning(game)
    model.train(5000)
