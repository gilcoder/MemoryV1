from ai4u.core import RemoteEnv
import numpy as np
from ai4u.utils import image_decode
import getch

TOUCH_SIZE = 4

def agent():
    env = RemoteEnv(IN_PORT=8080, OUT_PORT=7070, host="127.0.0.1")
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
            touched = np.zeros(TOUCH_SIZE)
            for i in range(TOUCH_SIZE):
                state = env.step("act", actions[action])
                energy = state['energy']
                touched[i] = state['touched']
                reward_sum += state['reward']
                if state['done']:
                    break
            done = state['done']
            prev_energy = energy
            frame = state['frame']
            frame = image_decode(state['frame'], 20, 20)
            print(frame)
            print('Reward: ', reward_sum)
            print('Touched: ', touched)
            print('Signal: ', state['signal'])
            print('Done: ', state['done'])
            print('redreward: ', state['redreward'])
            print('greenreward: ', state['greenreward'])
            print('boxreward: ', state['boxreward'])
            print('x: ', state['x'])
            print('z: ', state['z'])
            print('orientation: ', state['orientation'])
            prev_energy = energy
        if i >= 2:
            break
    env.close()

agent()