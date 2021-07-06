"""
This script defines the gym environment class for ContinuousDubinGym and DiscreteDubinGym
Goal Task - 
1. ContinuousDubinGym
	Action Space - linear velocity [0., 0.20], rotational velocity [-2.80, 2.80]
	Observation Space
	Reward
	Reset Condition
2. DiscreteDubinGym
	Action Space - linear velocity {0., 0.20}, rotational velocity {-2.50, -1.25, 0.0, 1.25, 2.50}
	Observation Space
	Reward
	Reset Condition
"""
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
import torch

MAX_STEER = 2.84
MAX_SPEED = 0.22
MIN_SPEED = 0.
THRESHOLD_DISTANCE_2_GOAL = 0.02
GRID = 5.
THETA0 = np.pi/4

# Vehicle parameters
LENGTH = 0.25  # [m]
WIDTH = 0.2  # [m]
BACKTOWHEEL = 0.05  # [m]
WHEEL_LEN = 0.03  # [m]
WHEEL_WIDTH = 0.02  # [m]
TREAD = 0.07  # [m]
WB = 0.15  # [m]

show_animation = True

class ContinuousDubinGym(gym.Env):

	def __init__(self):
		super(DubinGym,self).__init__()
		metadata = {'render.modes': ['console']}
		self.action_space = spaces.Box(np.array([0., -2.84]), np.array([0.22, 2.84]), dtype = np.float16) # max rotational velocity of burger is 2.84 rad/s
		low = np.array([-1.,-1.,-4.])
		high = np.array([1.,1.,4.])
		self.observation_space = spaces.Box(low, high, dtype=np.float16)
		self.target = [0., 0., 1.57]
		self.pose = [0., 0., 1.57]
		self.action = [0., 0.]
		self.traj_x = [self.pose[0]*MAX_X]
		self.traj_y = [self.pose[1]*MAX_Y]
		self.traj_yaw = [self.pose[2]]

	def reset(self): 
		self.pose = np.array([0., 0., 0.])
		x = random.uniform(-1., 1.)
		y = random.choice([-1., 1.])

		self.target[0], self.target[1] = random.choice([[x, y], [y, x]])

		head_to_target = self.get_heading(self.pose, self.target)
		yaw = random.uniform(theta - THETA0, theta + THETA0)

		self.pose[2] = yaw
		self.target[2] = yaw
		self.traj_x = [0.]
		self.traj_y = [0.]
		self.traj_yaw = [self.pose[2]]
		return np.array([(self.target[0] - self.pose[0])/GRID, (self.target[1] - self.pose[1])/GRID, head_to_target - self.pose[2]])

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

		return -1*(abs(crossTrackError)**2 + alongTrackError + 3*abs headingError/1.57)/6

	def check_goal(self):
		if abs(self.pose[0] < GRID) or abs(self.pose[1] < GRID):
			if(abs(self.pose[0]-self.target[0]) < THRESHOLD_DISTANCE_2_GOAL and  abs(self.pose[1]-self.target[1]) < THRESHOLD_DISTANCE_2_GOAL):
				done = True
				reward = 10
				print("Goal Reached!")
			else:
				reward = self.get_reward()
		else:
			done = True
			reward = -1
			print("Outside Range")
			
		return done, reward

	def step(self,action):
		reward = 0
		done = False
		info = {}
		self.action = action
		self.pose = self.update_state(self.pose, action, 0.05) # 0.005 Modify time discretization

		done, reward = self.check_goal()

		return np.array([(self.target[0] - self.pose[0])/GRID, (self.target[1] - self.pose[1])/GRID, head_to_target - self.pose[2]]), reward, done, info     

	def render(self):
		self.traj_x.append(self.pose[0])
		self.traj_y.append(self.pose[1])
		self.traj_yaw.append(self.pose[2])
	  
		plt.cla()
		# for stopping simulation with the esc key.
		plt.gcf().canvas.mpl_connect('key_release_event',
				lambda event: [exit(0) if event.key == 'escape' else None])
		plt.plot(self.traj_x, self.traj_y, "ob", markersize = 2, label="trajectory")
		plt.plot(self.target[0], self.target[1], "xg", label="target")
		self.plot_car()
		plt.axis("equal")
		plt.grid(True)
		plt.title("Simulation")
		plt.pause(0.0001)
		
	def close(self):
		pass

	def update_state(self, state, a, DT):
		lin_velocity = a[0]
		rot_velocity = a[1]

		state[0] = state[0] + lin_velocity * math.cos(state[2]) * DT
		state[1] = state[1] + lin_velocity * math.sin(state[2]) * DT
		state[2] = state[2] + rot_velocity * DT

		return state

	def plot_car(self, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
		# print("Plotting Car")
		x = self.pose[0]
		y = self.pose[1]
		yaw = self.pose[2] 
		steer = self.action[1]

		outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
							[WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

		fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
							 [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

		rr_wheel = np.copy(fr_wheel)

		fl_wheel = np.copy(fr_wheel)
		fl_wheel[1, :] *= -1
		rl_wheel = np.copy(rr_wheel)
		rl_wheel[1, :] *= -1

		Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
						 [-math.sin(yaw), math.cos(yaw)]])
		Rot2 = np.array([[math.cos(steer), math.sin(steer)],
						 [-math.sin(steer), math.cos(steer)]])

		fr_wheel = (fr_wheel.T.dot(Rot2)).T
		fl_wheel = (fl_wheel.T.dot(Rot2)).T
		fr_wheel[0, :] += WB
		fl_wheel[0, :] += WB

		fr_wheel = (fr_wheel.T.dot(Rot1)).T
		fl_wheel = (fl_wheel.T.dot(Rot1)).T

		outline = (outline.T.dot(Rot1)).T
		rr_wheel = (rr_wheel.T.dot(Rot1)).T
		rl_wheel = (rl_wheel.T.dot(Rot1)).T

		outline[0, :] += x
		outline[1, :] += y
		fr_wheel[0, :] += x
		fr_wheel[1, :] += y
		rr_wheel[0, :] += x
		rr_wheel[1, :] += y
		fl_wheel[0, :] += x
		fl_wheel[1, :] += y
		rl_wheel[0, :] += x
		rl_wheel[1, :] += y

		plt.plot(np.array(outline[0, :]).flatten(),
				 np.array(outline[1, :]).flatten(), truckcolor)
		plt.plot(np.array(fr_wheel[0, :]).flatten(),
				 np.array(fr_wheel[1, :]).flatten(), truckcolor)
		plt.plot(np.array(fl_wheel[0, :]).flatten(),
				 np.array(fl_wheel[1, :]).flatten(), truckcolor)
		plt.plot(x, y, "*")

class DiscreteDubinGym(gym.Env):

	def __init__(self, start_point):
		super(DubinGym,self).__init__()
		metadata = {'render.modes': ['console']}
		self.action_space = spaces.Discrete(15) 
		self.actSpace = {
			[0., -2.5], [0., -1.25], [0., 0.], [0., 1.25], [0., 2.5],
			[1.0, -2.5], [1., -1.25], [1., 0.], [1., 1.25], [1., 2.5],
			[2., -2.5], [2., -1.25], [2., 0.], [2., 1.25], [2., 2.5]
		}
		low = np.array([-1.,-1.,-4.])
		high = np.array([1.,1.,4.])
		self.observation_space = spaces.Box(low, high, dtype=np.float32)
		self.target = [1., 0., 1.57]
		self.pose = [0., 0., 1.57]
		self.action = [0., 0.]
		self.traj_x = [self.pose[0]*MAX_X]
		self.traj_y = [self.pose[1]*MAX_Y]
		self.traj_yaw = [self.pose[2]]

	def reset(self):
		self.pose = np.array([0., 0., 0.])
		x = random.uniform(-1., 1.)
		y = random.choice([-1., 1.])

		self.target[0], self.target[1] = random.choice([[x, y], [y, x]])
		head_to_target = self.get_heading(self.pose, self.target)
		yaw = random.uniform(theta - THETA0, theta + THETA0)
		self.pose[2] = yaw
		self.target[2] = yaw

		self.traj_x = [0.]
		self.traj_y = [0.]
		self.traj_yaw = [self.pose[2]]
		return np.array([(self.target[0] - self.pose[0])/GRID, (self.target[1] - self.pose[1])/GRID, head_to_target - self.pose[2]])

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

		return -1*(abs(crossTrackError)**2 + alongTrackError + 3*abs headingError/1.57)/6

	def check_goal(self):
		if abs(self.pose[0] < GRID) or abs(self.pose[1] < GRID):
			if(abs(self.pose[0]-self.target[0])<THRESHOLD_DISTANCE_2_GOAL and  abs(self.pose[1]-self.target[1])<THRESHOLD_DISTANCE_2_GOAL):
				done = True
				reward = 10
				print("Goal Reached!")
			else:
				reward = self.get_reward()
		else:
			done = True
			reward = -1
			print("Outside Range")			
		return done, reward

	def step(self,action):
		reward = 0
		done = False
		info = {}
		self.action = self.actSpace[action]
		self.pose = self.update_state(self.pose, self.action, 0.05) # 0.005 Modify time discretization

		done, reward = self.check_goal()

		return np.array([(self.target[0] - self.pose[0])/GRID, (self.target[1] - self.pose[1])/GRID, head_to_target - self.pose[2]]), reward, done, info     

	def render(self):
		self.traj_x.append(self.pose[0])
		self.traj_y.append(self.pose[1])
		self.traj_yaw.append(self.pose[2])
	  
		plt.cla()
		# for stopping simulation with the esc key.
		plt.gcf().canvas.mpl_connect('key_release_event',
				lambda event: [exit(0) if event.key == 'escape' else None])
		plt.plot(self.traj_x, self.traj_y, "ob", markersize = 2, label="trajectory")
		plt.plot(self.target[0], self.target[1], "xg", label="target")
		self.plot_car()
		plt.axis("equal")
		plt.grid(True)
		plt.title("Simulation")
		plt.pause(0.0001)
		
	def close(self):
		pass

	def update_state(self, state, a, DT):
		lin_velocity = a[0]
		rot_velocity = a[1]

		state[0] = state[0] + lin_velocity * math.cos(state[2]) * DT
		state[1] = state[1] + lin_velocity * math.sin(state[2]) * DT
		state[2] = state[2] + rot_velocity * DT

		return state

	def plot_car(self, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
		# print("Plotting Car")
		x = self.pose[0] 
		y = self.pose[1] 
		yaw = self.pose[2] 
		steer = self.action[1] 

		outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
							[WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

		fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
							 [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

		rr_wheel = np.copy(fr_wheel)

		fl_wheel = np.copy(fr_wheel)
		fl_wheel[1, :] *= -1
		rl_wheel = np.copy(rr_wheel)
		rl_wheel[1, :] *= -1

		Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
						 [-math.sin(yaw), math.cos(yaw)]])
		Rot2 = np.array([[math.cos(steer), math.sin(steer)],
						 [-math.sin(steer), math.cos(steer)]])

		fr_wheel = (fr_wheel.T.dot(Rot2)).T
		fl_wheel = (fl_wheel.T.dot(Rot2)).T
		fr_wheel[0, :] += WB
		fl_wheel[0, :] += WB

		fr_wheel = (fr_wheel.T.dot(Rot1)).T
		fl_wheel = (fl_wheel.T.dot(Rot1)).T

		outline = (outline.T.dot(Rot1)).T
		rr_wheel = (rr_wheel.T.dot(Rot1)).T
		rl_wheel = (rl_wheel.T.dot(Rot1)).T

		outline[0, :] += x
		outline[1, :] += y
		fr_wheel[0, :] += x
		fr_wheel[1, :] += y
		rr_wheel[0, :] += x
		rr_wheel[1, :] += y
		fl_wheel[0, :] += x
		fl_wheel[1, :] += y
		rl_wheel[0, :] += x
		rl_wheel[1, :] += y

		plt.plot(np.array(outline[0, :]).flatten(),
				 np.array(outline[1, :]).flatten(), truckcolor)
		plt.plot(np.array(fr_wheel[0, :]).flatten(),
				 np.array(fr_wheel[1, :]).flatten(), truckcolor)
		plt.plot(np.array(fl_wheel[0, :]).flatten(),
				 np.array(fl_wheel[1, :]).flatten(), truckcolor)

		plt.plot(x, y, "*") 

