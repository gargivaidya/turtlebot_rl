#!/usr/bin/env python3
"""

This script uses Gazebo simulation environment via ROS interface for an 'RL Evaluation' Task in PyTorch.
Task - Evaluate RL model from Dubins Gym to navigate from any random point to global origin

"""
import rospy
import time
from std_msgs.msg import Bool, Float32, Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from agent import ContinuousDubinGym, DiscreteDubinGym
from gym import spaces
from std_srvs.srv import Empty
import argparse
import datetime
import itertools
import torch, gc

import sys
sys.path.append('./algorithm/SAC')
from sac import SAC
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter

MAX_STEER = 2.84
MAX_SPEED = 0.22
MIN_SPEED = 0.
THRESHOLD_DISTANCE_2_GOAL = 0.05
GRID = 3.
THETA0 = np.pi/4

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
					help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
					help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
					help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
					help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
					help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
					help='Temperature parameter α determines the relative importance of the entropy\
							term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
					help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
					help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
					help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=500000, metavar='N',
					help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
					help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
					help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1, metavar='N',
					help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
					help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
					help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda',type=int, default=0, metavar='N',
					help='run on CUDA (default: False)')
parser.add_argument('--max_episode_length', type=int, default=3000, metavar='N',
					help='max episode length (default: 3000)')
args = parser.parse_args()

class ContinuousDubinGym(gym.Env):

	def __init__(self):
		super(ContinuousDubinGym,self).__init__()		
		metadata = {'render.modes': ['console']}
		print("Initialising Continuous Dubin Gym Enviuronment...")
		self.action_space = spaces.Box(np.array([-0.22, -2.84]), np.array([0.22, 2.84]), dtype = np.float16) # max rotational velocity of burger is 2.84 rad/s
		low = np.array([-1.,-1.,-4.])
		high = np.array([1.,1.,4.])
		self.observation_space = spaces.Box(low, high, dtype=np.float16)
		self.target = [0., 0., 1.57]
		self.pose = [0., 0., 1.57]
		self.action = [0., 0.]
		self.traj_x = [self.pose[0]]
		self.traj_y = [self.pose[1]]
		self.traj_yaw = [self.pose[2]]

	def reset(self): 
		self.pose = np.array([0., 0., 0.])
		x = random.uniform(-1., 1.)
		y = random.choice([-1., 1.])

		self.target[0], self.target[1] = random.choice([[x, y], [y, x]])

		head_to_target = self.get_heading(self.pose, self.target)
		yaw = random.uniform(head_to_target - THETA0, head_to_target + THETA0)

		self.pose[2] = yaw
		self.target[2] = yaw
		self.traj_x = [0.]
		self.traj_y = [0.]
		self.traj_yaw = [self.pose[2]]

		print("Reset target to : [{:.2f}, {:.2f}]".format(self.target[0], self.target[1]))

		obs = [(self.target[0] - self.pose[0])/GRID, (self.target[1] - self.pose[1])/GRID, head_to_target - self.pose[2]]
		obs = [round(x, 2) for x in obs]

		return np.array(obs)

	def get_distance(self,x1,x2):
		return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

	def get_heading(self, x1,x2):
		return math.atan2((x2[1] - x1[1]), (x2[0] - x1[0]))		

	def get_reward(self):
		x_target = self.target[0]
		y_target = self.target[1]
		yaw_target = self.target[2]

		x = self.pose[0]
		y = self.pose[1]

		yaw_car = self.pose[2]
		head_to_target = self.get_heading(self.pose, self.target)

		alpha = head_to_target - yaw_car
		ld = self.get_distance(self.pose, self.target)
		crossTrackError = math.sin(alpha) * ld

		headingError = abs(alpha)
		alongTrackError = abs(x - x_target) + abs(y - y_target)		

		return -1*(abs(crossTrackError)**2 + alongTrackError + 3*headingError/1.57)/6

	def check_goal(self):
		done = False
		if abs(self.pose[0]) < GRID and abs(self.pose[1]) < GRID:
			if(abs(self.pose[0]-self.target[0]) < THRESHOLD_DISTANCE_2_GOAL and  abs(self.pose[1]-self.target[1]) < THRESHOLD_DISTANCE_2_GOAL):
				done = True
				reward = 10
				print("Goal Reached!")
				self.stop_car()
			else:
				reward = self.get_reward()
		else:
			done = True
			reward = -1
			print("Outside Range")
			self.stop_car()
			
		return done, reward

	def step(self,action):
		
		reward = 0
		done = False
		info = {}
		self.action = [round(x, 2) for x in action]
		msg = Twist()
		msg.linear.x = self.action[0]
		msg.angular.z = self.action[1]

		print("Lin Vel : , Rot Vel : ", msg.linear.x, msg.angular.z)
		x_pub.publish(msg)
		time.sleep(0.02)
		head_to_target = self.get_heading(self.pose, self.target)

		done, reward = self.check_goal()

		obs = [(self.target[0] - self.pose[0])/GRID, (self.target[1] - self.pose[1])/GRID, head_to_target - self.pose[2]]
		obs = [round(x, 2) for x in obs]

		return np.array(obs), reward, done, info  

	def stop_car(self):
		'''
		Stop the vehicle
		'''
		global x_pub		
		msg = Twist()
		msg.linear.x = 0.
		msg.angular.z = 0.
		x_pub.publish(msg)
		time.sleep(1)		

# RL Model paths
actor_path = "models/sac_actor_burger_2021-07-06_16-14-21_"
critic_path = "models/sac_critic_burger_2021-07-06_16-14-21_"

# Instantiate RL Environment and load saved model
env =  ContinuousDubinGym()
env.target = [1., 1.]
agent = SAC(env.observation_space.shape[0], env.action_space, args)
memory = ReplayMemory(args.replay_size, args.seed)
agent.load_model(actor_path, critic_path)
state = np.zeros(env.observation_space.shape[0])

def euler_from_quaternion(x, y, z, w):
	'''
	Convert a quaternion into euler angles (roll, pitch, yaw)
	roll is rotation around x in radians (counterclockwise)
	pitch is rotation around y in radians (counterclockwise)
	yaw is rotation around z in radians (counterclockwise)
	'''
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + y * y)
	roll_x = math.atan2(t0, t1)

	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	pitch_y = math.asin(t2)

	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (y * y + z * z)
	yaw_z = math.atan2(t3, t4)

	return roll_x, pitch_y, yaw_z # in radians

def pose_callback(pose_data):
	'''
	Callback function to run the trained RL policy
	'''
	global episode_steps, action, done, sub
	pos[0] = pose_data.pose.pose.position.x
	pos[1] = pose_data.pose.pose.position.y
	orientation = pose_data.pose.pose.orientation
	q = (orientation.x,orientation.y,orientation.z,orientation.w)
	
	euler =  euler_from_quaternion(q[0], q[1], q[2], q[3])
	head = math.atan2(pos[1], pos[0]) # Heading to the origin
	yaw = euler[2] 
	state = np.array([(pos[0]/MAX_X), (pos[1]/MAX_Y), yaw]) # golden1 model
	done = False # Ends episode

	print("State : ", state)
	
	# Sample action from policy
	action = agent.select_action(state, True)	

	print("Network Output : ", action)

	if done: 
		# Stop the car and reset episode		
		env.stop_car()
		env.reset()
		sub.unregister()
		print('Counter:',episode_steps)
	else:
		# Execute action
		next_state, reward, done, _ = env.step(action)
		episode_steps += 1
			 
def start():
	'''
	Subscribe to robot pose topic and initiate callback thread
	'''
	global ts, episode_steps, action1, action2
	rospy.init_node('burger_gym', anonymous=True)		
	sub = pose_subscriber = rospy.Subscriber("/odom", Odometry, pose_callback)
	rospy.spin()

if __name__ == '__main__':
	try:
		start()
	except rospy.ROSInterruptException:
		pass
