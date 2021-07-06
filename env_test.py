
import itertools
import argparse
from agent import ContinuousDubinGym, DiscreteDubinGym
from stable_baselines.common.env_checker import check_env

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="ContinuousDubinGym",
					help='Dubin Gym environment (default: ContinuousDubinGym)')
args = parser.parse_args()

def main():
	
	if args.env_name == "ContinuousDubinGym":
		env =  ContinuousDubinGym()
	else:
		env = DiscreteDubinGym()

	print("Issues with Custom Environment : ", check_env(env))

	print("Testing sample action...")

	max_steps = int(1e6)
	state = env.reset()
	for ep in range(3):
		state = env.reset()
		for i in range(max_steps):
			if args.env_name == "ContinuousDubinGym":
				action = [1., 0.]
			else:
				action = 7			
			n_state,reward,done,info = env.step(action)
			env.render()
			if done:
				done = False                   
				break

if __name__ == '__main__':
	main()
