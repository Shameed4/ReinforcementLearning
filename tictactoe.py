import numpy as np
import math

class TicTacToe:
    # sets up the game
    # 
    # arguments:
    # - players: The appearance of the placed items
    #
    # created variables:
    # - board: The board
    # - numPlayers: Number of players playing
    # - turn: Whose turn it is (indexed at zero)
    def __init__(self, dims, dimSize, players) -> None:
        self.board = np.full([dimSize for _ in range(dims)], -1, int)
        self.players = players

        self.numPlayers = len(players) - 1

        self.turn = 0
        self.gameOver = False
        self.remainingTurns = int(dimSize ** dims)

        self.spotTaken = False
        self.dims = dims
        self.dimSize = dimSize

    def reset(self):
        self.board = np.full([self.dimSize for _ in range(self.board.ndim)], -1, int)
        self.turn = 0
        self.gameOver = False
        self.remainingTurns = int(self.dimSize ** self.board.ndim)
        self.spotTaken = False
    
    def is_legal(self, input) -> bool:
        # input is the wrong size (probably will never happen but good to be safe)
        if len(input) != self.board.ndim:
            return False
        
        # input is out of bounds (probably will happen many times)
        for i in input:
            if i < 0 or i >= len(self.board):
                return False

        # check if move was already placed there
        if self.board[input] != -1:
            print("Move was already placed here!")
        
        return self.board[input] == -1
        
    # places an item on the board, assuming the move is legal
    def place(self, input):
        self.spotTaken = False

        if not self.is_legal(input):
            return
        
        if self.board[input] != -1:
            print("Spot already taken, try again.")
            self.spotTaken = True
            return

        self.remainingTurns -= 1
        self.board[input] = self.turn
        
        if self.check_win(input):
            self.gameOver = True
            return

        self.turn += 1
        self.turn %= self.numPlayers

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
    
    # returns string representation of the board
    def __str__(self):
        boardCopy = self.board.astype(str)
        for i, s in enumerate(self.players):
            boardCopy[boardCopy == str(i - 1)] = s
        return str(boardCopy)
    
    # Returns the index to use when looking up the reward in the Q-table
    # move - Returns the index if the current player placed their move there. 
    # If move is not specified, it returns the index at the current position.
    def getState(self, move=None):
        ret = self.board.flatten()
        if move is None:
            return tuple(ret)
        ret[move] = self.turn
        return tuple(ret)


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
    
    # Returns the list of possible positions to move at a given position
    def getPossibleActions(self):
        ret = np.zeros(self.remainingTurns, dtype=object)
        counter = 0
        for i in range(self.dimSize):
            for j in range(self.dimSize):
                if self.board[i, j] == -1:
                    ret[counter] = (i, j)
                    counter += 1
        return ret


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
    # TODO: implement getPossibleActions, add 3D diagonals 
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

def test2D():
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

def test3D():
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

if __name__ == "__main__":
    test3D()