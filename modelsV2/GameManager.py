from modelsV2.MCTS import MCTS

class GameManager:
    def __init__(self, env):
        self.env = env

    #Trener treet n ganger ved å smiulere runder
    def train(self, mcts, n=100):
        self.env.reset()
        black_wins = 0
        white_wins = 0
        tie = 0
        
        for i in range(n):
            if i % 100 == 0: print(f"Training round {i}")
            # Nullstiller brettet
            self.env.reset()
            done = False
            while not done:
                
                # Gjør et trekk
                action = mcts.take_turn()
                state, reward, done, info = self.env.step(action)

            match self.env.winner():
                case 1:
                    black_wins += 1
                case -1:
                    white_wins += 1
                case _:
                    tie += 1

            mcts.backpropagate(mcts.currentNode, self.env.winner())
            
        print(f"Black wins: {black_wins}")
        print(f"White wins: {white_wins}")
        print(f"Tie: {tie}")
            
    def play_as_white(self, mcts):
        # Same logic as in training, but instead user takes action when whites turn
        self.env.reset()
        valid = False
        while not self.env.game_ended():
            mcts.traverse_policy(mcts.currentNode)
            state, reward, done, info = self.env.step(mcts.currentNode.action)
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
        
    def test(self, mcts1, mcts2):
        # Same logic as in training, but instead user takes action when whites turn
        self.env.reset()
        while not self.env.game_ended():
            action = mcts1.take_turn(mcts1.currentNode)
            state, reward, done, info = self.env.step(action)
            mcts2.opponent_turn_update(action)
                        
            if self.env.game_ended():
                break
            
            action = mcts2.take_turn(mcts2.currentNode)
            state, reward, done, info = self.env.step(action)
            mcts1.opponent_turn_update(action)
        
        return self.env.winner()

    def print_winner(self):
            if self.env.winner() == 1:
                print("Black Won")
            elif self.env.winner() == -1:
                print("White Won")
            else:
                print("Tie")
