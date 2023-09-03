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
    # - num_players: Number of players playing
    # - turn: Whose turn it is (indexed at zero)
    def __init__(self, dims, dim_size, players) -> None:
        self.board = np.full([dim_size for _ in range(dims)], -1, int)
        self.players = players

        self.num_players = len(players) - 1

        self.turn = 0
        self.game_over = False
        self.remaining_turns = self.board.size

        self.spot_taken = False
        self.dims = dims
        self.dim_size = dim_size
        self.state_size = self.board.size
        self.n_actions = self.board.size

        self.win_reward = 1
        self.draw_reward = 0.2
        self.lose_punishment = 1

    def reset(self):
        self.board = np.full([self.dim_size for _ in range(self.board.ndim)], -1, int)
        self.turn = 0
        self.game_over = False
        self.remaining_turns = int(self.dim_size ** self.board.ndim)
        self.spot_taken = False
    
    # Tries placing the given move. If legal, returns True and places. Otherwise, returns False
    def try_place(self, move : str) -> bool:
        move = move.strip()
        
        # check if moves is the correct type and size
        if not move.isdigit() or len(move) != self.dims:
            return False
        
        move = np.array(list(move), dtype=int)
    
        # check if move is in bounds
        if np.any((move < 0) | (move >= self.dim_size)):
            return False
        
        # check if move was already placed there
        if self.board[tuple(move)] != -1:
            print("Move was already placed here!")
        
        self.place(move)
        return True
        
    # places an item on the board, assuming the move is legal
    def place(self, move):
        # interprets the move if it is not an integer but a tuple instead
        if np.issubdtype(type(move), np.integer):
            move = np.unravel_index(move, shape=self.board.shape)
        move = tuple(move)

        self.spot_taken = False
        
        if self.board[move] != -1:
            print("Spot already taken, try again.")
            self.spot_taken = True
            return

        self.remaining_turns -= 1
        self.board[move] = self.turn
        
        if self.check_win(move):
            self.game_over = True
            return

        self.turn += 1
        self.turn %= self.num_players

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
        board_copy = self.board.astype(str)
        for i, s in enumerate(self.players):
            board_copy[board_copy == str(i - 1)] = s
        return str(board_copy)
    
    # Returns the index to use when looking up the reward in the Q-table
    # move - Returns the index if the current player placed their move there. 
    # If move is not specified, it returns the index at the current position.
    def get_state(self, move=None):
        return tuple(self.board.ravel())
    
    # Returns the list of possible positions to move at a given position
    def get_possible_actions(self, state=None):
        if state is None:
            state = self.board.ravel()
        return np.where(np.array(state) == -1)[0]
    
    # Returns a random legal move
    def choose_random_action(self):
        return np.random.choice(self.get_possible_actions())
    
    # Performs specified action and returns the new state, reward, and game_over
    def step(self, action, opponent):
        self.place(action)
        
        # win
        if self.game_over:
            return self.get_state(), self.win_reward, True
        # draw
        if self.remaining_turns == 0:
            return self.get_state(), self.draw_reward, True
        
        self.place(opponent.choose_move())

        # loss
        if self.game_over:
            return self.get_state(), -self.lose_punishment, True
        # draw
        if self.remaining_turns == 0:
            return self.get_state(), self.draw_reward, True
        
        # game is not over
        return self.get_state(), 0, False



class TicTacToe2D(TicTacToe):
    # sets up the game
    # 
    # arguments:
    # - players: The appearance of the placed items
    #
    # created variables:
    # - board: The board
    # - num_players: Number of players playing
    # - turn: Whose turn it is (indexed at zero)
    def __init__(self) -> None:
        super().__init__(dims=2, dim_size=3, players=['', 'x', 'o'])


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


class TicTacToe3D(TicTacToe):
    # sets up the game
    # 
    # arguments:
    # - players: The appearance of the placed items
    #
    # created variables:
    # - board: The board
    # - num_players: Number of players playing
    # - turn: Whose turn it is (indexed at zero)
    # TODO: implement get_possible_actions, add 3D diagonals 
    def __init__(self) -> None:
        super().__init__(dims=3, dim_size=4, players=['', 'x', 'o'])

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
        z_slice = self.board[:,:,z] 
        if x == y and np.all(z_slice.diagonal() == self.board[input]):
            return True
        elif x == self.dim_size - y - 1 and np.all(np.fliplr(z_slice).diagonal() == self.board[input]):
            return True
        del z_slice
        
        # check diagonal with same y but x and z are changing
        y_slice = self.board[:,y,:]
        if x == z and np.all(y_slice.diagonal() == self.board[input]):
            return True
        elif x == self.dim_size - z - 1 and np.all(np.fliplr(y_slice).diagonal() == self.board[input]):
            return True
        del y_slice

        # check diagonal with same x but y and z are changing
        x_slice = self.board[x,:,:]
        if y == z and np.all(x_slice.diagonal() == self.board[input]):
            return True
        elif y == self.dim_size - z - 1 and np.all(np.fliplr(x_slice).diagonal() == self.board[input]):
            return True
        del x_slice

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
        
        if (x not in range(0, game.dim_size)) or (y not in range(0, game.dim_size)):
            print("Move is out of bounds, try again")
            continue

        game.place((x, y))

        if game.spot_taken == False:
            print(game)

        if game.game_over == True:
            print(f"Player '{game.players[game.turn+1]}' wins!")
            break

        if game.remaining_turns == 0:
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

        if (x not in range(0, game.dim_size)) or (y not in range(0, game.dim_size)) or (z not in range(0, game.dim_size)):
            print("Move is out of bounds, try again")
            continue
    
        game.place((x, y, z))

        if game.spot_taken == False:
            print(game)

        if game.game_over == True:
            print(f"Player '{game.players[game.turn+1]}' wins!")
            break

        if game.remaining_turns == 0:
            print("Draw")
            break

if __name__ == "__main__":
    test3D()