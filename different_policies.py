import gym
import numpy as np
from MCTS.MCTS import MCTS as MCTSTREE
import matplotlib.pyplot as plt
from MCTSDNN.MCTSDNN import MCTSDNN
from GameManager import GameManager
size = 5
go_env = gym.make('gym_go:go-v0', size=size, komi=0, reward_method='real')

gm = GameManager(go_env)


def plot_training(mcts: MCTSDNN, title, loss, acc):
    losses = np.array(loss)
    accuracy = np.array(acc)
    plt.figure()
    plt.ylim(0,10)
    plt.plot(losses, label="Loss", color="blue")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title + " Loss -" f" Min: {np.min(losses)}")
    plt.show()
    # plt.savefig(f"graphs/{uuid.uuid4()}.png")

    plt.figure()
    plt.ylim(100)
    plt.plot(accuracy, label="Accuracy", color="Green")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(title + " Accuracy -" + f" Max: {np.max(accuracy)}")
    plt.show()
    # plt.savefig(f"graphs/{uuid.uuid4()}.png")

model_prob_pol = MCTSDNN(go_env, size, "Go2", kernel_size=3, prob_policy=True)
model_greedy_pol = MCTSDNN(go_env, size, "Go2", kernel_size=3, prob_policy=False)

model_prob_pol.train(3)
model_greedy_pol.train(3)
prob = 0
greedy = 0
tie = 0
for i in range(100):
    # Same logic as in training, but instead user takes action when whites turn
    #print(i)
    go_env.reset()
    model_prob_pol.reset()
    model_greedy_pol.reset()
    done = False
    while not done:
        action = model_prob_pol.take_turn_play(go_env)
        # print(action)
        _, _, done, _ = go_env.step(action)

        model_greedy_pol.opponent_turn_update(action)

        if done:
            break

        action = model_greedy_pol.take_turn_play(go_env)
        _, _, done, _ = go_env.step(action)
        model_prob_pol.opponent_turn_update(action)


    model_prob_pol.backpropagate(model_prob_pol.current_node, go_env.winner())
    model_greedy_pol.backpropagate(model_greedy_pol.current_node, go_env.winner())

    if(go_env.winner() == 1):
        prob += 1
    elif(go_env.winner() == -1):
        greedy += 1
    else:
        tie += 1
    go_env.reset()
    model_prob_pol.reset()
    model_greedy_pol.reset()

print(f"Score is Prob: {prob}(black), Greedy: {greedy}(white), Tie: {tie}")
plot_training(model_prob_pol, "Model W/ Prob Policy", model_prob_pol.model_losses, model_prob_pol.model_accuracy)

plot_training(model_greedy_pol, "Model W/ Greedy Policy", model_greedy_pol.model_losses, model_greedy_pol.model_accuracy)