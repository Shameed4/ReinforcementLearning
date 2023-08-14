import numpy as np
from tictactoe import TicTacToe2D

# player that makes moves by human input
class HumanPlayer():
    def __init__(self, game) -> None:
        self.game = game
    
    def selectMove(self):
        input("Human Playing (Press Enter)")
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
            
            if self.game.board[tuple(move)] == -1:
                return tuple(move)
            
            print("Invalid move - Piece already placed there")

# player that chooses a random legal move
class RandomPlayer():
    def __init__(self, game) -> None:
        self.game = game
    
    def selectMove(self):
        actions = self.game.getPossibleActions()
        return np.random.choice(actions)


if __name__ == "__main__":
    game = TicTacToe2D()
    agent = HumanPlayer(game)
    game.place((0, 0))
    print(agent.selectMove())