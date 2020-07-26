import gym
import argparse
from ai4u.utils import environment_definitions
from ai4u.ml.a3c.run_checkpoint import run as run_test
from ai4u.ml.a3c.train import run as run_train
import AI4UGym
from AI4UGym import BasicAgent
import numpy as np
from ai4u.utils import image_decode
from stable_baselines.common.callbacks import CheckpointCallback
from math import log, e
from collections import deque
import time

H_SIZE = 4
FEATURE_SHAPE = (20, 20, H_SIZE)
ACTION_SHAPE = (4, )
ARRAY_SIZE = 10

def make_inference_network(obs_shape, n_actions, debug=False, extra_inputs_shape=None):
	import tensorflow as tf
	from ai4u.ml.a3c.multi_scope_train_op import make_train_op 
	from ai4u.ml.a3c.utils_tensorflow import make_grad_histograms, make_histograms, make_rmsprop_histograms, \
		logit_entropy, make_copy_ops

	observations = tf.placeholder(tf.float32, [None] + list(obs_shape))
	proprioceptions = tf.placeholder(tf.float32, (None, ARRAY_SIZE) )

	normalized_obs = tf.keras.layers.Lambda(lambda x : x/255.0)(observations)

	# Numerical arguments are filters, kernel_size, strides
	conv1 = tf.keras.layers.Conv2D(128, (2,2), (1,1), activation='relu', name='conv1')(normalized_obs)
	if debug:
		# Dump observations as fed into the network to stderr for viewing with show_observations.py.
		conv1 = tf.Print(conv1, [observations], message='\ndebug observations:',
						 summarize=2147483647)  # max no. of values to display; max int32	
	conv2 = tf.keras.layers.Conv2D(128, (2,2), (2,2), activation='relu', name='conv2')(conv1)

	hp1 = tf.keras.layers.Dense(30, activation='relu', name='phidden')(proprioceptions[:, 0:ARRAY_SIZE])
	flattened = tf.keras.layers.Flatten()(conv2)
	expanded_features = tf.keras.layers.Concatenate()([flattened, hp1])

	hidden1 = tf.keras.layers.Dense(512, activation='relu', name='hidden1')(expanded_features)
	hidden2 = tf.keras.layers.Dense(64, activation='relu', name='hidden2')(hidden1)

	action_logits = tf.keras.layers.Dense(n_actions, activation=None, name='action_logits')(hidden2)
	action_probs = tf.nn.softmax(action_logits)

	values = tf.layers.Dense(1, activation=None, name='value')(hidden2)

	# Shape is currently (?, 1)
	# Convert to just (?)
	values = values[:, 0]

	layers = [conv1, conv2, hp1, expanded_features, hidden1, hidden2]

	return (observations, proprioceptions), action_logits, action_probs, values, layers

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--run",
						choices=['train', 'test'],
						default='train')
	parser.add_argument("--id", default='0')
	parser.add_argument('--path', default='.')
	parser.add_argument('--load_ckpt')
	parser.add_argument('--preprocessing', choices=['generic', 'user_defined'])
	return parser.parse_args()


def get_entropy(actions):
	s = np.zeros(ACTION_SHAPE[0])
	for action in actions:
		s[action] += 1
	p = s/len(actions)
	result = np.where(p > 0.00, p, -1)
	return -np.sum(p * np.log2(result, out=result, where=result>0))

def make_env_def():
	environment_definitions['state_shape'] = FEATURE_SHAPE #shape of the observation space
	environment_definitions['action_shape'] = ACTION_SHAPE #shape of the action space
	environment_definitions['actions'] = [('act', 0), ('act', 1), ('act', 4), ('act', -1)] #list of actions
	environment_definitions['input_port'] = 8080 #port of this machine
	environment_definitions['output_port'] = 7070 #port of the machine running the environment
	environment_definitions['min_value'] = 0 #observation space min value
	environment_definitions['max_value'] = 800 #observation space max value
	environment_definitions['make_inference_network'] = make_inference_network
	environment_definitions['agent'] = Agent #Agent class defines callbacks to enviroment and learn methods.
	environment_definitions['host'] = '127.0.0.1' #ip address of the machine running the environment.
	environment_definitions['extra_inputs_shape'] = (ARRAY_SIZE,)
	BasicAgent.environment_definitions = environment_definitions #set the environment properties

'''def transform(frame):
	x = frame[28]
	z = frame[29]
	j = 0
	f = []
	for _ in range(14):
		f.append( (frame[j] - x)/500. )
		f.append(( frame[j+1] - z)/500.0 )
		j += 2
	f.append(frame[30]/360.)
	f.append(frame[31])
	f.append(frame[32])
	return np.array(f)
'''

class Agent(BasicAgent):
	def __init__(self):
		BasicAgent.__init__(self)

	def __make_state__(imageseq, extrainfo):
		frameseq = np.array(imageseq, dtype=np.float32)
		frameseq = np.moveaxis(frameseq, 0, -1)
		return (frameseq, extrainfo)

	def reset(self, env):
		self.h = deque(maxlen=H_SIZE)
		env_info = env.remoteenv.step("restart")
		time.sleep(1)
		for i in range(np.random.choice(15)):
			env_info = env.one_step(np.random.choice([0, 1, 2]))

		f = image_decode(env_info['frame'], 20, 20, dtype=np.float32)
		for _ in range(H_SIZE):
			self.h.append(f)
		return Agent.__make_state__(self.h, np.array(env_info['reward_hist']))

	def act(self, env, action, info=None):
		envinfo = env.one_step(action)
		sum_reward = envinfo['reward']
		for k in range(1):
			if envinfo['done']:
				break
			else:
				envinfo = env.one_step(3)
				sum_reward += envinfo['reward']
		f = image_decode(envinfo['frame'], 20, 20, dtype=np.float32)
		self.h.append(f)
		return Agent.__make_state__(self.h, np.array(envinfo['reward_hist'])), sum_reward, envinfo['done'], envinfo


def train(load_ckpt=None):
		if load_ckpt is None:
			args = ['--n_workers=8', '--steps_per_update=30', 'AI4U-v0', '--n_steps=10e8']
		else:
			args = ['--n_workers=8', '--steps_per_update=30', 'AI4U-v0', '--load_ckpt=%s'%(load_ckpt)]
		make_env_def()
		run_train(environment_definitions, args)

def test(path, id=0):
		args = ['AI4U-v0', path]
		make_env_def()
		run_test(environment_definitions, args)

if __name__ == '__main__':
	make_env_def()
	args = parse_args()
	if args.run == "train":
		train(args.load_ckpt)
	elif args.run == "test":
		test(args.path, int(args.id))
