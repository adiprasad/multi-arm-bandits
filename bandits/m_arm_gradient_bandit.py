from .m_arm_bandit import MultiArmBandit
from .gradient_bandit import GradientBandit
import sys
import numpy as np
from scipy.special import logsumexp

class MultiArmGradientBandit(MultiArmBandit):
	
	def __init__(self, num_bandits, bandit_sampling_mean, bandit_sampling_std_dev, bandit_std_dev, step_size, gradient_step_size, init_val=0):
		super(MultiArmGradientBandit, self).__init__(num_bandits, bandit_sampling_mean, bandit_sampling_std_dev, bandit_std_dev, step_size, init_val)
		self.gradient_step_size = gradient_step_size


	def init_bandits(self):
		best_bandit_mean = - sys.maxint
		pi = 1/float(self.num_bandits)

		for i in range(self.num_bandits):
			bandit_mean = np.random.normal(self.bandit_sampling_mean, self.bandit_sampling_std_dev)
			bandit = GradientBandit(pi, bandit_mean, self.bandit_std_dev, self.init_val, self.step_size, self.gradient_step_size)

			# print "Gradient bandit #" + str(i) + " initialized"
			# print "Gradient bandit #" + str(i) + " preference " + str(bandit.get_preference())
			
			if bandit_mean > best_bandit_mean:
				self._MultiArmBandit__set_best_bandit(bandit)
				best_bandit_mean = bandit_mean

			#print "Initialized bandit : " + str(i) + "with mean : " + str(bandit_mean)
			self.bandit_list.append(bandit)


	def adjust_preferences(self, idx_of_chosen_bandit, baseline, last_reward):
		for idx, bandit in enumerate(self.bandit_list):
			# print "bandit idx : " + str(idx)
			# print "bandit prob : " + str(bandit.get_pi())
			was_this_idx_taken = (idx_of_chosen_bandit == idx)

			'''
			if was_this_idx_taken:
				print "Inside adjust preferences, Baseline : " + str(baseline) + ", Last reward : " + str(last_reward)
				print "Inside adjust preferences, Bandit idx : " + str(idx) + " was taken, Bandit pi : " + str(bandit.get_pi())
			'''

			#if was_this_idx_taken:
				#print "best idx from m arm gradient bandit : " + str(idx)
				#print "Q* of bandit from m arm : " + str(bandit.get_mean()) 
			# print "was this idx taken" + str(was_this_idx_taken)
			# print "bandit preference before " + str(bandit.get_preference())
			bandit.adjust_preference(was_this_idx_taken, baseline, last_reward)
			# print "bandit preference after " + str(bandit.get_preference())

			# if (np.isnan(bandit.get_preference())):
			# 	sys.exit()
			

	def get_logsumexp_sum_of_preferences(self):
		preference_list = []

		for bandit in self.bandit_list:
			preference_list.append(bandit.get_preference())

		return logsumexp(preference_list)


	def adjust_policy(self):
		logsumexp_sum_of_preferences = self.get_logsumexp_sum_of_preferences()

		for idx, bandit in enumerate(self.bandit_list):
			bandit.adjust_pi(logsumexp_sum_of_preferences)
			#print "Inside adjust policy, after adjustment, Bandit idx : " + str(idx) + " , Bandit pi : " + str(bandit.get_pi())







