import numpy as np
from tictactoeND import TicTacToe

class TicTacToe2D(TicTacToe):
    # sets up the game
    # 
    # arguments:
    # - players: The appearance of the placed items
    #
    # created variables:
    # - board: The board
    # - numPlayers: Number of players playing
    # - turn: Whose turn it is (indexed at zero)
    def __init__(self) -> None:
        super().__init__(dims=2, dimSize=3, players=['', 'x', 'o'])


    # checks if there are any 3-in-a-row's containing the given input
    def check_win(self, input) -> bool:
        x, y = input

        # check row
        if np.all(self.board[x] == self.board[input]):
            return True
        
        # check col
        if np.all(self.board[:, y] == self.board[input]):
            return True
        
        # check / diagonal
        if np.all(self.board.diagonal() == self.board[input]):
            return True
        
        # check \ diagonal
        if np.all(np.fliplr(self.board).diagonal() == self.board[input]):
            return True

        return False


if __name__ == "__main__":
    input("When prompted to enter a move, enter an integer. For example, to place in (1, 2), enter '12'.")
    game = TicTacToe2D()

    while True:
        myStr = input("Enter a move: ")

        if myStr.lower() == "reset":
            game.reset()
            print("Game has been reset.")
            continue
        
        try:
            myInt = int(myStr)
        except ValueError:
            print("Invalid move, enter a valid integer")
            continue
        
        x = myInt // 10
        y = myInt % 10
        
        if (x not in range(0, game.dimSize)) or (y not in range(0, game.dimSize)):
            print("Move is out of bounds, try again")
            continue

        game.place((x, y))

        if game.spotTaken == False:
            print(game)

        if game.gameOver == True:
            print(f"Player '{game.players[game.turn+1]}' wins!")
            break

        if game.remainingTurns == 0:
            print("Draw")
            break
