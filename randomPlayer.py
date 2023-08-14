import numpy as np
from tictactoe2d import TicTacToe2D

# player that makes completely random moves
class RandomAgent():
    def __init__(self, game) -> None:
        self.game = game
    
    def selectMove(self):
        actions = self.game.getPossibleActions()
        return np.random.choice(actions)


if __name__ == "__main__":
    agent = RandomAgent(TicTacToe2D())
    print(agent.selectMove())