import numpy as np
from .gaussian_bandit import GaussianBandit
import sys

class MultiArmBandit(object):

	def __init__(self, num_bandits, bandit_sampling_mean, bandit_sampling_std_dev, bandit_std_dev, step_size, init_val=0):
		self.num_bandits = num_bandits
		self.bandit_sampling_mean = bandit_sampling_mean
		self.bandit_sampling_std_dev = bandit_sampling_std_dev
		self.bandit_std_dev = bandit_std_dev
		self.step_size = step_size
		self.init_val = init_val

		self.bandit_list = []				# Initialize num_bandits and store mean and variance of those bandits in this list


	def init_bandits(self):
		best_bandit_mean = - sys.maxsize

		for i in range(self.num_bandits):
			bandit_mean = np.random.normal(self.bandit_sampling_mean, self.bandit_sampling_std_dev)
			bandit = GaussianBandit(bandit_mean, self.bandit_std_dev, self.init_val, self.step_size)
			
			if bandit_mean > best_bandit_mean:
				self.__set_best_bandit(bandit)
				best_bandit_mean = bandit_mean

			self.bandit_list.append(bandit)

	def __set_best_bandit(self, bandit):
		self.best_bandit = bandit

	def get_best_bandit(self):
		return self.best_bandit
	
	def get_bandits(self):
		return self.bandit_list

	def get_bandit(self, idx):
		return self.bandit_list[idx]
















