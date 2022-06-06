import numpy as np
from .gaussian_bandit import GaussianBandit

class GradientBandit(GaussianBandit):

	def __init__(self, pi, mean=0.0, std_dev=1.0, init_val=0, step_size=None, gradient_step_size = 0.1):
		super(GradientBandit, self).__init__(mean, std_dev, init_val, step_size)
		
		self.preference = 0
		self.gradient_step_size = gradient_step_size
		self.pi = pi

	def __eq__(self, other):
		if isinstance(other, GradientBandit):
			#return self.get_preference() == other.get_preference()
			return self.get_mean() == other.get_mean()

		return False


	def __ne__(self, other):
		if isinstance(other, GradientBandit):
			return self.get_preference() != other.get_preference()

		return True


	def get_preference(self):
		return self.preference

	
	def adjust_preference(self, was_taken, baseline, last_reward):
		# print "inside adjust_preference, was_taken : " + str(was_taken)
		# print "inside adjust_preference, gradient_step_size : " + str(self.gradient_step_size)
		# print "inside adjust_preference, last_reward : " + str(last_reward)
		# print "inside adjust_preference, baseline : " + str(baseline)

		if was_taken:
			self.preference = self.preference + self.gradient_step_size * (last_reward - baseline)*(1 - self.pi)
		else:
			self.preference = self.preference - self.gradient_step_size * (last_reward - baseline)*self.pi


	def adjust_pi(self, logsumexp_sum_of_preferences):
		self.pi = np.exp(self.preference - logsumexp_sum_of_preferences)


	def get_pi(self):
		return self.pi










