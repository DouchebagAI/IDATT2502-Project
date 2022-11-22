from MCTSDNN.MCTSDNN import MCTSDNN

import copy
import numpy as np
import torch
import matplotlib.pyplot as plt

class GameManager:
    def __init__(self, env):
        self.env = env
        #print valid moves

            
    def play_as_white(self, mcts):
        # Same logic as in training, but instead user takes action when whites turn
        self.env.reset()
        mcts.reset()
        valid = False
        while not self.env.game_ended():
         
            state, reward, done, info = self.env.step(mcts.take_turn_play())
            while not valid:
                try:
                    action = self.env.render(mode="human")
                    state, reward, done, info = self.env.step(action)
                    valid = True
                except:
                    print("Invalid move")
            valid = False
            mcts.opponent_turn_update(action)
        self.print_winner()
        
    def test(self, mcts1:MCTSDNN, mcts2:MCTSDNN):
        # Same logic as in training, but instead user takes action when whites turn
        self.env.reset()
        mcts1.reset()
        mcts2.reset()
        done = False
        while not done:
            action = mcts1.take_turn_play()
            # print(action)
            _, _, done, _ = self.env.step(action)
            
            mcts2.opponent_turn_update(action)
                        
            if done:
                break
            
            action = mcts2.take_turn_play()
            _, _, done, _ = self.env.step(action)
            mcts1.opponent_turn_update(action)
        #mcts1.backpropagate(mcts1.current_node, self.env.winner())
        #mcts2.backpropagate(mcts2.current_node, self.env.winner())
        return self.env.winner()


    def play_tournament(self):
        # Extracting all models
        players = []
        print("Henter ut alle spillerne")

        for i in range(20):
            model = torch.load(f"models/SavedModels/{i}.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            model.eval()
            # Convert to tree
            tree = MCTSDNN(self.env, 5, "Go2", kernel_size=3)
            tree.model = model
            players.append(tree)
        win_dict = dict()
        # Playing the tournament
        print("Playing turnering")
        for i in range(1,20):
            print(f"Round {i}")
            numberOfWins = 0
            for j in range(1000):
                if(j%2 == 0):
                    winner = self.test(players[0], players[i])
                    if winner == 1:
                        numberOfWins += 1
                else:
                    winner = self.test(players[i], players[0])
                    if winner == -1:
                        numberOfWins += 1
            # Update win dict
            win_dict.update({i: numberOfWins})
            print(f"Player 0 won {numberOfWins} times against player {i}")
        # Plot the results
        print(len(list(win_dict.keys())))
        print(len(list(win_dict.values())))
        plt.figure()
        plt.plot(np.array(list(win_dict.keys())),np.array(list(win_dict.values())))
        plt.show()

        
    def print_winner(self):
            if self.env.winner() == 1:
                print("Black Won")
            elif self.env.winner() == -1:
                print("White Won")
            else:
                print("Tie")
