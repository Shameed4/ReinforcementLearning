from tictactoe import TicTacToe
import numpy as np

class QLearning:
    # alpha - learning rate
    # gamma - discount rate
    # epsilon - greedy-epsilon parameter
    # numEpisodes - number of simulated episodes
    def __init__(self, game=None, alpha=0.01, gamma=0.01, epsilon=0.99):       
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.table = np.zeros((3,) * 9)

    
    def getState(self):
        return tuple(self.game.board.flatten())
    

    def getPossibleActions(self):
        ret = []
        for i in range(3):
            for j in range(3):
                if self.game.board[i, j] == -1:
                    ret.append((i, j))        
        return ret
    

if __name__ == "__main__":
    game = TicTacToe()
    model = QLearning(game)
    game.place((0, 0))
    print(model.getPossibleActions())
    