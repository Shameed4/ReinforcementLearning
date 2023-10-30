import numpy as np
from qtable import QLearning
from dqn import DQN
from tictactoe import TicTacToe2D
from opponents import HumanPlayer, RandomPlayer
from userinput import promptTwoLetters, promptYesNo, promptDigit, promptFloat

if __name__ == "__main__":
    game = TicTacToe2D()

    if promptTwoLetters("Neural network or Q-learning", "n", "q"):
        model = DQN(game=game)
    else:
        model = QLearning(game, epsilon_multiplier=0.999995, alpha=0.02, gamma=0.95)

    if promptYesNo("Load model?"):
        model.load()

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
            model.save() # remove when we're happy with model's accuracy

    def gui():
        rand_episodes = 2000
        clone_episodes = 2000
        steps = 20
        epsilon = 1
        while True:
            inp = input('''
              T - Continue training
              P - Play against model
              S - Save model
              Q - Quit
              ''').strip().upper()
            if inp == "T":
                cont = True
                while cont:
                    print(f"Parameters: rand_episodes={rand_episodes} clone_episodes={clone_episodes} steps={steps} epsilon={epsilon}")
                    if promptYesNo("Change parameters (y/n)"):
                        rand_episodes = promptDigit("Random Episodes")
                        clone_episodes = promptDigit("Clone episodes")
                        steps = promptDigit("Steps")
                        epsilon = promptFloat("Starting epsilon:")
                    train_epoch(rand_episodes, clone_episodes, steps, epsilon)
                    cont = promptYesNo("Continue training? (y/n)")

            elif inp == "P":
                inp2 = promptDigit("Number of games")
                model.train(inp2, HumanPlayer(game))

            elif inp == "S":
                print("Saving")
                model.save()

            elif inp == "Q":
                print("Quitting")
                return



    # while not promptYesNo("Train?"):
    #     print(f"Parameters: rand_episodes={rand_episodes} clone_episodes={clone_episodes} steps={steps} epsilon={epsilon}")
    #     if not promptYesNo("Change parameters?"):
    #         rand_episodes = promptDigit("Random episodes")
    #         clone_episodes = promptDigit("Clone episodes")
    #         steps = promptDigit("Steps")
    #         epsilon = float(input("Epsilon:"))
    #     train_epoch(rand_episodes, clone_episodes, steps, epsilon)

    gui()