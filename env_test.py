import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces
import time
import itertools
import argparse
import datetime
import random
from stable_baselines.common.env_checker import check_env

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="ContinuousDubinGym",
                    help='Dubin Gym environment (default: ContinuousDubinGym)')
args = parser.parse_args()

def main():
	
	if args.env-name == ContinuousDubinGym:
		env =  ContinuousDubinGymDubinGym()
	else:
		env = DiscreteDubinGym()

	print(check_env(env))

	max_steps = int(1e6)
	state = env.reset()
	env.render()
	for ep in range(3):
		state = env.reset()
		env.render()
		for i in range(max_steps):
			action = [1.0, 0.]
	 		n_state,reward,done,info = env.step(action)
	 		env.render()
	 		if done:
	 			state = env.reset()
	 			done = False                   
	 			break

if __name__ == '__main__':
	main()
