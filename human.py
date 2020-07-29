import gym
import getch

env = gym.make('CartPole-v0')
env.reset()
env.render()
frames = 200
score = 0

for episode in range(5):
	env.reset()
	print('Press 0 or 1 to start. Controls are 0 or 1.\n1 applies leftwards force on the cart, while 0 applies rightwards force on the cart.')
	for f in range(frames):
		action = int(getch.getche())
		observation, reward, done, info = env.step(action)
		score += reward
		print(' ', observation, reward, done, info)
		env.render()
		print('Frame: ', f)
		#if done:
			#break
	print('\n***Game', (episode+1), 'done***\nScore: ', score, '\n')