import numpy as np
import math


class TicTacToe:
    # sets up the game
    #
    # arguments:
    # - players: The appearence of the placed items
    #
    # created variables:
    # - board: The board
    # - numPlayers: Number of players playing
    # - turn: Whose turn it is (indexed at zero)
    def __init__(self, dims=3, dimSize=4, players=['', 'x', 'o']) -> None:
        self.board = np.full([dimSize for _ in range(dims)], -1, int)
        self.players = players

        self.numPlayers = len(players) - 1

        self.turn = 0
        self.gameOver = False
        self.remainingTurns = int(dimSize ** dims)

    def is_legal(self, input) -> bool:
        # input is the wrong size (probably will never happen but good to be safe)
        if len(input) != game.board.ndim:
            return False

        # input is out of bounds (probably will happen many times)
        for i in input:
            if i < 0 or i >= len(self.board):
                return False

        return True

    # places an item on the board, assuming the move is legal
    def place(self, input):
        if not self.is_legal(input):
            return

        self.board[input] = self.turn

        if self.check_win(input):
            game.gameOver = True
            return

        self.turn += 1
        self.turn %= self.numPlayers

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


if __name__ == "__main__":
    input("When prompted to enter a move, enter an integer. For example, to place in (1, 2, 1), enter '121'.")
    game = TicTacToe()

    while True:
        myStr = int(input("Enter a move: "))
        game.place((math.floor(myStr / 100), math.floor(myStr / 10) % 10, math.floor(myStr % 10)))
        print(game)

        if game.gameOver == True:
            break