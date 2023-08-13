import numpy as np
import math
from tictactoeND import TicTacToe

class TicTacToe3D(TicTacToe):
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
        super().__init__(dims=3, dimSize=4, players=['', 'x', 'o'])

    # checks if there are any 3-in-a-row's containing the given input
    def check_win(self, input) -> bool:
        x,y,z = input

        # check row with the same x and z but y is changing
        if np.all(self.board[x,:,z] == self.board[input]):
            return True

        # check row with the same y and z but x is changing
        if np.all(self.board[:,y,z] == self.board[input]):
            return True

        # check row with the same x and y but z is changing
        if np.all(self.board[x,y,:] == self.board[input]):
            return True

        # check diagonal with same z but x and y are changing
        zSlice = self.board[:,:,z] 
        if x == y and np.all(zSlice.diagonal() == self.board[input]):
            return True
        elif x == self.dimSize - y - 1 and np.all(np.fliplr(zSlice).diagonal() == self.board[input]):
            return True
        del zSlice
        
        # check diagonal with same y but x and z are changing
        ySlice = self.board[:,y,:]
        if x == z and np.all(ySlice.diagonal() == self.board[input]):
            return True
        elif x == self.dimSize - z - 1 and np.all(np.fliplr(ySlice).diagonal() == self.board[input]):
            return True
        del ySlice

        # check diagonal with same x but y and z are changing
        xSlice = self.board[x,:,:]
        if y == z and np.all(xSlice.diagonal() == self.board[input]):
            return True
        elif y == self.dimSize - z - 1 and np.all(np.fliplr(xSlice).diagonal() == self.board[input]):
            return True
        del xSlice

        return False


if __name__ == "__main__":
    input("When prompted to enter a move, enter an integer. For example, to place in (1, 2, 1), enter '121'.")
    game = TicTacToe3D()

    while True:
        myStr = input("Enter a move: ")
        try:
            myInt = int(myStr)
        except ValueError:
            print("Invalid move, enter a valid integer")
            continue

        x = math.floor(myInt / 100)
        y = math.floor(myInt / 10) % 10
        z = math.floor(myInt % 10)

        if (x not in range(0, game.dimSize)) or (y not in range(0, game.dimSize)) or (z not in range(0, game.dimSize)):
            print("Move is out of bounds, try again")
            continue
    
        game.place((x, y, z))

        if game.spotTaken == False:
            print(game)

        if game.gameOver == True:
            print(f"Player '{game.players[game.turn+1]}' wins!")
            break

        if game.remainingTurns == 0:
            print("Draw")
            break

