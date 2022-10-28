from models.MCTS import MCTS
class GameManager:

    def __init__(self, env, black: MCTS, white: MCTS):
        self.env = env
        self.black = black
        self.white = white

    def train(self):
        self.env.reset()
        self.black.monte_carlo_tree_search(render=True)
