from ai4u.core import RemoteEnv
import numpy as np
from ai4u.utils import image_decode
import getch

def transform(frame):
	f = np.array(frame)
	x = f[20] #agent position x
	z = f[21] #agent position z
	features = []
	j = 0
	for _ in range(10):
		x2 = f[j] # a target position x
		z2 = f[j+1] # a target position z
		d = np.linalg.norm(np.array([x, z]) - np.array([x2, z2])) #distance between agent and current target
		features.append(d/500.) #nomalize this feature
		j += 2
	features.append(f[22]/360.0) #agent orientation (normalized)
	features.append(f[23]) #first touch on red target?
	features.append(f[24]) #first touch on green target?
	return features

def agent():
	env = RemoteEnv(IN_PORT=8080, OUT_PORT=8081, host="127.0.0.1")
	env.open(0)
	actions = {'w':0, 's': 3, 'a': 4, 'd': 1}
	for i in range(10000000):
		#state = env.step("restart")
		state = env.step("restart")
		prev_energy = state['energy']
		done = state['done']
		print("OK")
		while not done:
			print('Action: ', end='')
			action = getch.getche()
			#action = np.random.choice([0, 1])
			reward_sum = 0
			state = env.step("act", actions[action])
			reward_sum += state['reward']
			for i in range(1):
				done = state['done']
				if (not done):
					state = env.step('act', -1)
					reward_sum += state['reward']
				else:
					break

			energy = state['energy']
			prev_energy = energy
			frame = state['frame']
			print(transform(frame))
			print('Reward: ', reward_sum)
			print('Done: ', state['done'])
			prev_energy = energy
		if i >= 2:
			break
	env.close()

agent()