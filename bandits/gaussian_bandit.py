import numpy as np

class GaussianBandit(object):

	def __init__(self, mean=0.0, std_dev=1.0, init_val=0, step_size=None):
		self.mean = mean
		self.std_dev = std_dev
		self.q_init = init_val
		self.q = init_val
		self.step_size = step_size			## Step size None implies sample average method
		self.returns = np.array([])
		self.step_size_decay_array = np.array([1])


	def __eq__(self, other):
		if isinstance(other, GaussianBandit):
			return self.get_estimate() == other.get_estimate()

		return False


	def __ne__(self, other):
		if isinstance(other, GaussianBandit):
			return self.get_estimate() != other.get_estimate()

		return True


	def execute(self):
		ret_val = self._get_return()

		self.returns = np.append(self.returns, ret_val)

		self.__add_decay_ratio_to_arr()
		
		self._adjust_estimate()

		return ret_val

	
	def get_estimate(self):
		return self.q


	def get_experience_len(self):
		return self.returns.size


	def get_mean(self):
		return self.mean

	
	def _get_return(self):
		sample_from_normal = np.random.normal(self.mean, self.std_dev, 1)

		return sample_from_normal[0]


	def _adjust_estimate(self):
		if self.step_size is not None:
			self.q = self.__decayed_sum_of_previous_rewards()
		else:
			step_size = (1/float(self.get_experience_len()))
			self.q = self.q + step_size*(self.returns[-1] - self.q)


	def __add_decay_ratio_to_arr(self):
		if self.step_size is not None:
			self.step_size_decay_array = np.append(self.step_size_decay_array, self.step_size_decay_array[-1] * self.step_size)


	def __decayed_sum_of_previous_rewards(self):
		n = np.size(self.returns)

		one_minus_alpha = 1 - self.step_size
		decay_arr_reversed = np.flip(self.step_size_decay_array)[:n]

		decayed_sum = self.step_size_decay_array[-1] * one_minus_alpha * self.q_init + (self.step_size * np.dot(decay_arr_reversed, self.returns))

		return decayed_sum








