import numpy as np
from tictactoe import TicTacToe2D

# player that makes moves by human input
class HumanPlayer():
    def __init__(self, game) -> None:
        self.game = game
    
    def chooseMove(self):
        print("Playing as player", self.game.turn+1)
        print(self.game)
        while True:
            try:
                move = np.array(list(input("Select a move: ")), dtype=int)
            except:
                print("Invalid move - Input contains non-integers")
                continue

            if len(move) != self.game.dims:
                print("Invalid move - Input was wrong length")
                continue

            success = True
            for m in move:
                if m < 0 or m >= self.game.dimSize:
                    print("Invalid move - Input outside board")
                    success = False
                    break
            if not success:
                continue
            
            move = tuple(move)
            if self.game.board[move] == -1:
                return move
            
            print("Invalid move - Piece already placed there")

# player that chooses a random legal move
class RandomPlayer():
    def __init__(self, game) -> None:
        self.game = game
    
    def chooseMove(self):
        return self.game.chooseRandomAction()


if __name__ == "__main__":
    game = TicTacToe2D()
    agent = HumanPlayer(game)
    game.place((0, 0))
    print(agent.chooseMove())