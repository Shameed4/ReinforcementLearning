import numpy as np
import math
from tictactoe import TicTacToe3D

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