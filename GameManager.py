from MCTSDNN.MCTSDNN import MCTSDNN

import copy

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
        mcts1.backpropagate(mcts1.current_node, self.env.winner())
        mcts2.backpropagate(mcts2.current_node, self.env.winner())
        return self.env.winner()

    def print_winner(self):
            if self.env.winner() == 1:
                print("Black Won")
            elif self.env.winner() == -1:
                print("White Won")
            else:
                print("Tie")
