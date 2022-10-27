import gym

go_env = gym.make('gym_go:go-v0', size=7, komi=0, reward_method='real')

first_action = (2,5)
second_action = (5,2)
a = go_env.action_space
print(a.shape)
print(a.n)
go_env.reset()
state, reward, done, info = go_env.step(first_action)
#go_env.render('terminal')


