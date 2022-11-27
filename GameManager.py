from MCTSDNN.MCTSDNN import MCTSDNN

import torch


class GameManager:
    def __init__(self, env):
        self.env = env

    def play_as_white(self, mcts):
        """
        This is a function for playing against a trained network with the white pieces.

        :param mcts: Player 1: MCTS with a trained network
        :param mcts: Player 2: MCTS with a trained network
        """
        # Same logic as in training, but instead user takes action when whites turn
        self.env.reset()
        mcts.reset()
        valid = False
        while not self.env.game_ended():

            state, reward, done, info = self.env.step(mcts.take_turn_2)
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

    def test(self, mcts1: MCTSDNN, mcts2: MCTSDNN):
        """
        This is a function for playing two MCST w/ network against each other

        :param mcts1: One MCTS with the trained network
        :param mcts2: MCTS with the trained network
        """

        self.env.reset()
        mcts1.reset()
        mcts2.reset()
        done = False
        while not done:
            action = mcts1.take_turn_2()
            _, _, done, _ = self.env.step(action)
            self.env.render("terminal")

            mcts2.opponent_turn_update(action)

            if done:
                break

            action = mcts2.take_turn_2()
            _, _, done, _ = self.env.step(action)
            mcts1.opponent_turn_update(action)
            self.env.render("terminal")

        return self.env.winner()

    def play_tournament(self, player=6, num_opponents=10):
        """
        This is a function for playing different trained MCTS /w models against each other.
        Starting with extracting all models and creating players.
        Then play the main player against the others players

        :param player: The main player in the tournament
        :param num_opponents: Number of opponents for the player in the tournament
        :return: (dict) With numbers of wins against each player
        """

        players = []
        # Extracting all opponent models
        for i in range(num_opponents):
            if i != player and i % 2 == 0:
                model = torch.load(f"models/SavedModels/{i}_Gen2_Go2.pt",
                                   map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                model.eval()
                value_model = torch.load(f"models/SavedModels/{i}_Gen2_Go2_value.pt",
                                         map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                value_model.eval()
                # Convert to tree
                tree = MCTSDNN(self.env, 5, "Go2", kernel_size=3, prob_policy=False)
                tree.model = model
                tree.value_model = value_model
                tree.trainingRoundsCompleted = i + 1
                players.append(tree)

        # Extracting player model
        model = torch.load(f"models/SavedModels/{player}_Gen2_Go2.pt",
                           map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        value_model = torch.load(f"models/SavedModels/{player}_Gen2_Go2_value.pt",
                                 map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        value_model.eval()

        # Convert to tree
        tree = MCTSDNN(self.env, 5, "Go2", kernel_size=3, prob_policy=False)
        tree.model = model
        tree.value_model = value_model
        tree.trainingRoundsCompleted = player
        players.append(tree)

        win_dict = dict()

        # Playing the tournament
        print("Playing turnering")
        for i in range(1, num_opponents):
            print(f"Round {i}")
            numberOfWins = 0
            for j in range(30):
                print(f"Game {i}.{j}")
                if (j % 2 == 0):
                    print("Player is BLACK")
                    winner = self.test(players[len(players) - 1], players[i])
                    if winner == 1:
                        numberOfWins += 1
                else:
                    print("Player is WHITE")
                    winner = self.test(players[i], players[len(players) - 1])
                    if winner == -1:
                        numberOfWins += 1
            # Update win dict
            win_dict.update({i: numberOfWins})
            print(f"Player {player} won {numberOfWins} times against player {i}")
        return win_dict

    def print_winner(self):
        """
        This is a function for printing the winner
        """
        if self.env.winner() == 1:
            print("Black Won")
        elif self.env.winner() == -1:
            print("White Won")
        else:
            print("Tie")
