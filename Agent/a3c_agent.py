from ai4u.ml.a3c.train import run as run_train
from ai4u.ml.a3c.run_checkpoint import run as run_test
from ai4u.utils import environment_definitions
import AI4UGym
from AI4UGym import BasicAgent
import numpy as np
import argparse
from ai4u.utils import image_decode
from collections import deque
import sys
import time

H_SIZE = 4
FEATURE_SHAPE = (25 * H_SIZE, )
ACTION_SHAPE = (5, )
WORKERS = 4

def transform(frame):
	f = np.array(frame)
	return f * 1/1000.

def get_entropy(actions):
	s = np.zeros(ACTION_SHAPE[0])
	for action in actions:
		s[action] += 1
	p = s/len(actions)
	result = np.where(p > 0.00, p, -1)
	return -np.sum(p * np.log2(result, out=result, where=result>0))

def make_inference_network(obs_shape, n_actions, debug=False, extra_inputs_shape=None):
	import tensorflow as tf
	from ai4u.ml.a3c.multi_scope_train_op import make_train_op 
	from ai4u.ml.a3c.utils_tensorflow import make_grad_histograms, make_histograms, make_rmsprop_histograms, \
		logit_entropy, make_copy_ops

	observations = tf.placeholder(tf.float32, [None] + list(obs_shape))

	hidden1 = tf.keras.layers.Dense(256, activation='relu', name='hidden1')(observations)
	hidden2 = tf.keras.layers.Dense(256, activation='relu', name='hidden2')(hidden1)
	
	action_logits = tf.keras.layers.Dense(n_actions, activation=None, name='action_logits')(hidden2)
	action_probs = tf.nn.softmax(action_logits)

	values = tf.layers.Dense(1, activation=None, name='value')(hidden2)

	# Shape is currently (?, 1)
	# Convert to just (?)
	values = values[:, 0]

	layers = [hidden1, hidden2]

	return observations, action_logits, action_probs, values, layers


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--run",
						choices=['train', 'test'],
						default='train')
	parser.add_argument("--id", default='0')
	parser.add_argument('--path', default='.')
	parser.add_argument('--preprocessing', choices=['generic', 'user_defined'])
	return parser.parse_args()

class Agent(BasicAgent):
	def __init__(self):
		BasicAgent.__init__(self)
		self.action_hist = deque(maxlen=30)
		self.entropy_list = deque(maxlen=30)

	def reset(self, env):
		self.action_hist = deque(maxlen=30)
		self.entropy_list = deque(maxlen=30)
		self.h = deque(maxlen=4)
		env_info = env.remoteenv.step("restart")
		for i in range(np.random.choice(30)):
			env_info = env.one_step(np.random.choice([0, 1, 4]))
		f = transform(env_info['frame'])
		self.h.append(f)
		self.h.append(f)
		self.h.append(f)
		self.h.append(f)
		state = np.array(self.h)
		state = state.reshape( (FEATURE_SHAPE[0], ) )
		return state

	def act(self, env, action, info=None):
		self.action_hist.append(action)
		ae = get_entropy(self.action_hist)
		er = 0
		if (len(self.entropy_list) >= 30):
			m = np.mean(self.entropy_list)
			s = np.std(self.entropy_list)
			if ae > m + s:
				er += 0.0001
		self.entropy_list.append(ae)
		envinfo = env.one_step(action)
		sum_reward = envinfo['reward']
		f = transform(envinfo['frame'])
		self.h.append(f)
		state = np.array(self.h)
		state = state.reshape( (FEATURE_SHAPE[0], ))		
		return state, sum_reward, envinfo['done'], envinfo

def make_env_def():
	environment_definitions['state_shape'] = FEATURE_SHAPE
	environment_definitions['action_shape'] = ACTION_SHAPE
	environment_definitions['actions'] = [('act', 0), ('act', 1), ('act', 3), ('act', 4), ('act', -1)]
	environment_definitions['input_port'] = 8080
	environment_definitions['output_port'] = 7070
	environment_definitions['agent'] = Agent
	environment_definitions['host'] = '127.0.0.1'
	BasicAgent.environment_definitions = environment_definitions

def train():
		args = ['--n_workers=4', '--steps_per_update=30', 'AI4U-v0']
		make_env_def()
		run_train(environment_definitions, args)

def test(path, id=0):
		args = ['AI4U-v0', path]
		make_env_def(id)
		run_test(environment_definitions, args)

if __name__ == '__main__':
   args = parse_args()
   if args.run == "train":
	   train()
   elif args.run == "test":
	   test(args.path, int(args.id))
