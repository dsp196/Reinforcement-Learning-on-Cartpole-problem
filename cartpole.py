import gym
import numpy as np
import random
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

env = gym.make("CartPole-v1")
env.reset()

frames = 200
score_requirement = 75
initial_games = 10000

def not_trained_games():
	for episode in range(5):
		env.reset()

		for t in range(200):
			env.render()
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
                
not_trained_games()



def get_data():

	# [OBS, MOVES]
	training_data = []
	scores = []
	accepted_scores = []
	# iterate through however many games we want:
	for _ in range(initial_games):
		score = 0
		# moves specifically from this environment:
		game_memory = []
		# previous observation that we saw
		prev_observation = []
		
		# for each frame in 200
		for _ in range(frames):
			# choose random action (0 or 1)
			action = random.randrange(0,2)
			observation, reward, done, info = env.step(action)
			if len(prev_observation) > 0 :
				game_memory.append([prev_observation, action])
			
			prev_observation = observation
			score+=reward
			if done:
				break

		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				# convert to one-hot (this is the output layer for our neural network)
				if data[1] == 1:
					output = [0,1]
				elif data[1] == 0:
					output = [1,0]
				# saving our training data
				training_data.append([data[0], output])

		# reset env to play again
		env.reset()
		# save overall scores
		scores.append(score)

		#save_training_data = np.array(training_data)
		#np.save('saved.npy', save_training_data)

	print('Average accepted score:',mean(accepted_scores))
	print('Median score for accepted scores:',median(accepted_scores))
	print(Counter(accepted_scores))
	return training_data


#Neural Network: Simple Multilayer Perceptron Model

def neural_network_model(input_size):

	network = input_data(shape=[None, input_size, 1], name='input_layer')

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 512, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 2, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy', name='target_layer')
	model = tflearn.DNN(network, tensorboard_dir='log')

	return model


def train_model(training_data, model=False):

	X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
	y = [i[1] for i in training_data]
	#print(y)

	if not model:

		model = neural_network_model(input_size = len(X[0]))

		model.fit({'input_layer': X}, {'target_layer': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')

	return model

training_data = get_data()

		#X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
		#y = [i[1] for i in training_data]
		#model = neural_network_model(input_size = len(X[0]))
		#model = model.load("Test1.tfl")

model = train_model(training_data)

trained_scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(frames):
        env.render()

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

        choices.append(action)
                
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done: break

    trained_scores.append(score)

print('Average Score:',sum(trained_scores)/len(trained_scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_requirement)

#model.save("Test1.tfl")
