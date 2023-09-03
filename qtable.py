from tictactoe import TicTacToe2D
from opponents import *
import numpy as np

class QLearning:
    # alpha - learning rate
    # gamma - discount rate
    # epsilon - greedy-epsilon parameter
    # num_episodes - number of simulated episodes
    def __init__(self, game, alpha=0.01, gamma=0.95, epsilon=0.99, epsilon_multiplier=0.9995, table=None):
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_multiplier = epsilon_multiplier

        if table is None:
            # (num_players+1)^boardSize for the state, and boardSize for action
            self.table = np.zeros((game.num_players + 1,) * game.board.size + (game.board.size,))
        else:
            self.table = table

    # Chooses and returns a move based on the epsilon-greedy algorithm (without placing it)
    # In the beginning, it will initially choose moves that are ENTIRELY random to set up the Q-table
    # Later, it will choose between random and its best move
    # Episode - The number of episodes that have been simulated
    def choose_move(self):
        self.epsilon *= self.epsilon_multiplier
        
        if np.random.rand() < self.epsilon:
            return self.game.choose_random_action()
        
        actions = self.game.get_possible_actions()
        rewards = self.table[self.game.get_state()][actions]
        
        move = np.argmax(rewards)
        return actions[move]
    
    # Trains the model for the specified number of episodes
    # If an opponent is not specified, the opponent will make a random legal move each time
    def train(self, episodes, opponent=None):
        if opponent is None:
            opponent = RandomPlayer(self.game)
        
        wins = 0
        draws = 0
        losses = 0
        
        for episode in range(episodes):
            self.game.reset() # reset the game after each episode
            episode_done = False
            
            # alternate between x and o
            if episode % 2 == 0:
                self.game.place(opponent.choose_move())
            
            state = self.game.get_state()
            while not episode_done:
                action = self.choose_move()
                pred_q = self.table[state][action] # index the q table with the current state and action
                
                # take game step
                new_state, reward, episode_done = self.game.step(action, opponent)

                # update Q values
                new_q = pred_q + self.alpha * (reward + self.gamma * np.max(self.table[new_state]) - pred_q)
                self.table[state][action] = new_q

                # update state
                state = new_state

                # update stats
                if episode_done:
                    if reward == self.game.win_reward:
                        wins += 1
                    elif reward == self.game.draw_reward:
                        draws += 1
                    else:
                        losses += 1
            
        print(f'Wins={wins/episodes}, Draws={draws/episodes}, Losses={losses/episodes}, _epsilon={self.epsilon}')
    
    def cloneTable(self, epsilon=0.75, epsilon_multiplier=1):
        return QLearning(game=self.game, random_episodes=0, epsilon=epsilon, epsilon_multiplier=epsilon_multiplier, table=np.copy(self.table))


if __name__ == "__main__":
    game = TicTacToe2D()
    model = QLearning(game, epsilon_multiplier=0.999995, alpha=0.02, gamma=0.95)

    if input("Load model? (Y|N): ".strip().upper()) == "Y":
        model.table = np.load("./qlearning.npy")
    
    def train_epoch(random_episodes=1000, clone_episodes=1000, steps=40, start_epsilon=1.5):
        if start_epsilon is not None:
            model.epsilon = start_epsilon

        print("-----------\nNew Epoch\n-----------")
        for train_step in range(steps):
            print("Remaining rounds:", steps - train_step, "alpha=", model.alpha)
            if clone_episodes != 0:
                print("Clone")
                model.train(clone_episodes, opponent=model)
            if random_episodes != 0:
                print("Random")
                model.train(random_episodes)

    def gui():
        while True:        
            inp = input('''
              T - Continue training
              P - Play against model
              S - Save model
              Q - Quit
              ''').strip().upper()
            if inp == "T":
                cont = input("Continue training? (y/n)") != "n"
                while cont:
                    if input("Change parameters (y/n)") != "n":
                        rand_episodes = int(input("Random episodes:"))
                        clone_episodes = int(input("Clone episodes:"))
                        steps = int(input("Steps:"))
                        epsilon = float(input("Starting epsilon:"))
                    train_epoch(rand_episodes, clone_episodes, steps, epsilon)
                    cont = input("Continue training? (y/n)") != "n"
            
            elif inp == "P":
                inp2 = input("How many games?")
                if inp2.isdigit():
                    inp2 = int(inp2)
                    model.train(inp2, HumanPlayer(game))
                else:
                    print("Invalid input")
            
            elif inp == "S":
                print("Saving")
                np.save('./qlearning.npy', model.table)

            elif inp == "Q":
                print("Quitting")
                return

    
    # large number of random moves          
    train_epoch(10000, 0, 15, 30)
    
    # # some exploration
    # train_epoch(2000, 0, 30, 1)

    # # getting better, start training against itself
    # train_epoch(2000, 500, 25, 1.2)
    # train_epoch(2000, 1000, 25, 1.2)
    # train_epoch(2000, 1500, 25, 1.2)

    # train_epoch(2000, 2000, 50, 1.2)
    # train_epoch(2000, 2000, 50, 1.2)
    
    # # train against itself
    # train_epoch(0, 2000, 20, 1)
    
    # train_epoch(2000, 2000, 50, 1.2)
    # train_epoch(2000, 2000, 50, 1.2)
    # train_epoch(2000, 2000, 50, 1.2)
    # train_epoch(2000, 2000, 50, 1.2)
    # train_epoch(2000, 2000, 50, 1.2)
    # train_epoch(2000, 2000, 50, 1.2)

    # # final test to see if it has reached perfection
    # train_epoch(10000, 10000, 5, 0)

    rand_episodes = 2000
    clone_episodes = 2000
    steps = 20
    epsilon = 1

    while input("Continue training? (y/n)") != "n":
        print(f"Parameters: rand_episodes={rand_episodes} clone_episodes={clone_episodes} steps={steps} epsilon={1}")
        if input("Change parameters (y/n)") != "n":
            rand_episodes = int(input("Random episodes:"))
            clone_episodes = int(input("Clone episodes:"))
            steps = int(input("Steps:"))
            epsilon = float(input("Epsilon:"))
        train_epoch(rand_episodes, clone_episodes, steps, epsilon)

    gui()