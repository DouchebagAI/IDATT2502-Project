from models.MCTS import MCTS
class GameManager:

    def __init__(self, env, black: MCTS, white: MCTS):
        self.env = env
        self.black = black
        self.white = white

    def train(self, x=100):
        self.env.reset()
        #self.black.monte_carlo_tree_search(render=False)
        black_w = 0
        white_w = 0
        tie = 0
        for i in range(x):
            print(i)
            self.env.reset()
            while not self.env.game_ended():
                self.black.take_turn()
                self.white.opponent_turn_update(self.black.currentNode.action)
                if(self.env.game_ended()):
                    break
                self.white.take_turn()
                self.black.opponent_turn_update(self.white.currentNode.action)

            if self.env.winner() == 1: black_w += 1
            elif self.env.winner() == -1: white_w += 1
            else:
                tie += 1
            #self.env.render('terminal')
            self.black.backpropagate(self.black.currentNode, self.env.winner())
            self.white.backpropagate(self.white.currentNode,self.env.winner() * (-1))

        print(f"Black wins: {black_w}")
        print(f"White wins: {white_w}")
        print(f"Tie: {tie}")

    def play_as_white(self):
        self.env.reset()
        while not self.env.game_ended():
            self.black.take_turn()
            action = self.env.render(mode="human")
            state, reward, done, info = self.env.step(action)
            self.black.opponent_turn_update(action)
        self.print_winner()

    def play_as_black(self):
        self.env.reset()
        while not self.env.game_ended():
            action = self.env.render(mode="human")
            state, reward, done, info = self.env.step(action)
            self.white.opponent_turn_update(action)
            self.white.take_turn()
        self.print_winner()

    def print_winner(self):
        if self.env.winner() == 1:
            print("You Won!")
        elif self.env.winner() == -1:
            print("The Machine won!")
        else:
            print("Tie")