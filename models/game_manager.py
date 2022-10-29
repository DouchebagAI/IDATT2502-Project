from models.MCTS import MCTS


# Class for training wo clients
class GameManager:

    def __init__(self, env, black: MCTS, white: MCTS):
        self.env = env
        self.black = black
        self.white = white

    def train(self, x=100):

        self.env.reset()
        black_w = 0
        white_w = 0
        tie = 0

        for i in range(x):
            self.env.reset()
            if i % 1000 == 0:
                print(i)
            done = False
            # Plays until game ends
            while not done:
                # Black starts and takes a turn
                action = self.black.take_turn()
                # White chooses node based on black action / step
                self.white.opponent_turn_update(action)
                state, reward, done, info = self.env.step(action)
                if done:
                    break
                # White takes turn
                action = self.white.take_turn()
                # Black updates node based on whites turn
                self.black.opponent_turn_update(action)
                state, reward, done, info = self.env.step(action)

            if self.env.winner() == 1:
                black_w += 1

            elif self.env.winner() == -1:
                white_w += 1

            else:
                tie += 1

            # self.env.render('terminal')
            self.black.backpropagate(self.black.currentNode, self.env.winner())
            self.white.backpropagate(self.white.currentNode, self.env.winner() * -1)

        # print(self.black.R.print_self())
        # print(self.white.R.print_self())
        print(f"Black wins: {black_w}")
        print(f"White wins: {white_w}")
        print(f"Tie: {tie}")

    def play_as_white(self):
        # Same logic as in training, but instead user takes action when whites turn
        self.env.reset()
        valid = False
        while not self.env.game_ended():
            action = self.black.take_turn()
            state, reward, done, info = self.env.step(action)
            while not valid:
                try:
                    action = self.env.render(mode="human")
                    state, reward, done, info = self.env.step(action)
                    valid = True
                except:
                    print("Invalid move")
            valid = False
            self.black.opponent_turn_update(action)
        self.print_winner(-1)

    def play_as_black(self):
        self.env.reset()
        done = False
        while not done:
            action = self.env.render(mode="human")
            state, reward, done, info = self.env.step(action)
            self.white.opponent_turn_update(action)
            if done:
                return
            action = self.white.take_turn()
            state, reward, done, info = self.env.step(action)
        self.print_winner(1)

    def print_winner(self, x):
        if self.env.winner() == x:
            print("You Won!")
        elif self.env.winner() == x * (-1):
            print("The Machine won!")
        else:
            print("Tie")
