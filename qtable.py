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

    # Selects a move based on the epsilon-greedy algorithm
    # In the beginning, it will initially pick moves that are ENTIRELY random to set up the Q-table
    # Later, it will choose between random and its best move
    # Episode - The number of episodes that have been simulated
    def selectMove(self, episode=0):
        actions = self.game.getPossibleActions()

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
                s = self.game.getState()
                a = self.selectMove(episode)

                curr_q = self.table[s] # index the q table with the current state

                self.game.place(a)
                new_s = self.game.getState()

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
    print(np.max(model.table))
