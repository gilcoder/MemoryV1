import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2, ACKTR
from ai4u.utils import environment_definitions
import AI4UGym
from AI4UGym import BasicAgent
import numpy as np
from math import log, e
from collections import deque

H_SIZE = 4
FEATURE_SHAPE = (25 * H_SIZE, )
ACTION_SHAPE = (5, )
WORKERS = 4

model = None

def get_entropy(actions):
	s = np.zeros(ACTION_SHAPE[0])
	for action in actions:
		s[action] += 1
	p = s/len(actions)
	result = np.where(p > 0.00, p, -1)
	return -np.sum(p * np.log2(result, out=result, where=result>0))

def make_env_def():
	environment_definitions['state_shape'] = FEATURE_SHAPE
	environment_definitions['action_shape'] = ACTION_SHAPE
	environment_definitions['actions'] = [('act', 0), ('act', 1), ('act', 3), ('act', 4), ('act', -1)]
	environment_definitions['input_port'] = 8080
	environment_definitions['output_port'] = 7070
	environment_definitions['agent'] = Agent
	environment_definitions['host'] = '127.0.0.1'
	BasicAgent.environment_definitions = environment_definitions


def transform(frame):
	f = np.array(frame)
	return f * 1/1000.


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

def make_model_presets():
	import tensorflow as tf
	make_env_def()
	env = make_vec_env('AI4U-v0', n_envs=WORKERS)
	policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[256, dict(vf=[128], pi=[128])])
	return env, policy_kwargs
	
def train(pretrained_model=None):
	# multiprocess environment
	env, policy_kwargs = make_model_presets()
	if pretrained_model is None:
		model = PPO2(MlpPolicy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs/", learning_rate=0.001)
		model.learn(total_timesteps=5000000)
	else:
		model = PPO2.load(pretrained_model)
		model.set_env(env)
		model.learn(total_timesteps=5000000, reset_num_timesteps=True)
	model.save("ppo2_model")
	del model # remove to demonstrate saving and loading

def test():
	env, policy_kwargs = make_model_presets()
	model = PPO2.load("ppo2_model", policy=MlpPolicy, policy_kwargs=policy_kwargs, tensorboard_log="./logs/")
	model.set_env(env)
	# Enjoy trained agent
	obs = env.reset()
	while True:
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		env.render()

if __name__ == '__main__':
	train()

