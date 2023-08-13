import numpy as np

# player that makes completely random moves
class RandomAgent():
    def __init__(self, game) -> None:
        self.game = game
    
    def selectMove(self):
        return np.random.choice(actions)