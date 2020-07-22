# A Memory Challenge to Reinforcement Learning Algorithms.

This is a simple but challenging game for reinforcement learning algorithms. An agent needs to touch objects of the same color. This means that the agent needs to remember the first object that was touched. The objective of this environment is to verify if reinforcement learning agents are able to learn tasks that require the memorization of past actions.

# Ações

* Discret (gym)
* 0: forward
* 1: turn right
* 2: backward
* 3: turn left
* 4: Idle

# State

* User can define a observation_space. Environment simulation returns a twenty five elements array. Elements in this array are target positions, agent' position, agent' orientation, if first touch was in red object, and if first touch was in green object.

# Agent

Folder **Agent** contains the rlagent.py example.