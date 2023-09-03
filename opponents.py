import numpy as np
from tictactoe import TicTacToe2D

# player that makes moves by human input
class HumanPlayer():
    def __init__(self, game) -> None:
        self.game = game
    
    def choose_move(self):
        print("Playing as player", self.game.turn+1)
        print(self.game)
        move = input("Select a move: ")
        while not self.game.try_place(move):
            move = input("Select a move: ")

# player that chooses a random legal move
class RandomPlayer():
    def __init__(self, game) -> None:
        self.game = game
    
    def choose_move(self):
        return self.game.choose_random_action()


if __name__ == "__main__":
    game = TicTacToe2D()
    agent = HumanPlayer(game)
    agent.choose_move()
    print(game)