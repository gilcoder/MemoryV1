import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2, A2C
from ai4u.utils import environment_definitions
import AI4UGym
from AI4UGym import BasicAgent
import numpy as np
from math import log, e
from collections import deque

H_SIZE = 3
FEATURE_SHAPE = (31 * H_SIZE, )
ACTION_SHAPE = (5, )
WORKERS = 8

model = None

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
	environment_definitions['actions'] = [('act', 0), ('act', 1), ('act', 3), ('act', 4), ('act', -1)] #list of actions
	environment_definitions['input_port'] = 8080 #port of this machine
	environment_definitions['output_port'] = 7070 #port of the machine running the environment
	environment_definitions['min_value'] = 0 #observation space min value
	environment_definitions['max_value'] = 800 #observation space max value
	environment_definitions['agent'] = Agent #Agent class defines callbacks to enviroment and learn methods.
	environment_definitions['host'] = '127.0.0.1' #ip address of the machine running the environment.
	BasicAgent.environment_definitions = environment_definitions #set the environment properties


def transform(frame):
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


class Agent(BasicAgent):
	def __init__(self):
		BasicAgent.__init__(self)

	def reset(self, env):
		self.h = deque(maxlen=H_SIZE)
		env_info = env.remoteenv.step("restart")
		for i in range(np.random.choice(15)):
			env_info = env.one_step(np.random.choice([0, 1, 4]))
		f = transform(env_info['frame'])
		for _ in range(H_SIZE):
			self.h.append(f)
		state = np.array(self.h)
		state = state.reshape( (FEATURE_SHAPE[0], ) )
		return state

	def act(self, env, action, info=None):
		envinfo = env.one_step(action)
		sum_reward = envinfo['reward']
		for k in range(1):
			if envinfo['done']:
				break
			else:
				envinfo = env.one_step(4)
				sum_reward += envinfo['reward']
		f = transform(envinfo['frame'])
		self.h.append(f)
		state = np.array(self.h)
		state = state.reshape( (FEATURE_SHAPE[0], ))		
		return state, sum_reward, envinfo['done'], envinfo

def make_model_presets():
	import tensorflow as tf
	make_env_def()
	env = make_vec_env('AI4U-v0', n_envs=WORKERS)
	policy_kwargs = dict( net_arch=[256, dict(vf=[128], pi=[128])] )
	return env, policy_kwargs
	
def train(pretrained_model=None):
	# multiprocess environment
	env, policy_kwargs = make_model_presets()
	if pretrained_model is None:
		model = PPO2(MlpPolicy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs/")
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

