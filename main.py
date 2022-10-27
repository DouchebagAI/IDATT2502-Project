import gym

go_env = gym.make('gym_go:go-v0', size=7, komi=0, reward_method='heuristic')

first_action = (2,5)
second_action = (5,2)
go_env.reset()
state, reward, done, info = go_env.step(first_action)
print("Reward: ", reward)
a = go_env.valid_moves()
print(a)
print("",len(a))
state, reward, done, info = go_env.step(second_action)

go_env.render('terminal')


