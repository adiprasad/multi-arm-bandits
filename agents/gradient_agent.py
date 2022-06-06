from .agents import AbstractAgent
import numpy as np
import sys
import time

class GradientAgent(AbstractAgent):

	def __init__(self, m_arm_bandit, baseline=None):
		super(GradientAgent, self).__init__(m_arm_bandit)
		self.baseline = baseline

	
	def take_action(self):
		best_bandit = self._AbstractAgent__get_best_bandit()
		bandit_idx = self.find_most_probable_bandit_idx()

		#print "best idx from take action : " + str(bandit_idx) 

		bandit = self.m_arm_bandit.get_bandits()[bandit_idx]

		#print "Q* of bandit from take action : " + str(bandit.get_mean()) 

		reward = bandit.execute()

		is_best_action = (best_bandit == bandit)

		self._AbstractAgent__increment_episode_count()
		self._AbstractAgent__append_episode_reward(reward)
		self._AbstractAgent__append_action_type(is_best_action)

		# print "episode : " + str(self.episode_cnt)
		# print "preference sum : " + str(self.m_arm_bandit.get_logsumexp_sum_of_preferences())

		self.__update_policy(bandit_idx)

		return reward


	def __update_policy(self, idx_of_chosen_bandit):
		if self.baseline is None:
			baseline = np.mean(self.rewards_list)
		else:
			baseline = self.baseline

		last_reward = self.rewards_list[-1]

		# print "Idx of chosen bandit : " + str(idx_of_chosen_bandit)
		# print "Last reward : " + str(last_reward)
		# print "Baseline : " + str(baseline)

		self.m_arm_bandit.adjust_preferences(idx_of_chosen_bandit, baseline, last_reward)
		self.m_arm_bandit.adjust_policy()

		#time.sleep(1)


	def find_most_probable_bandit_idx(self):
		bandit_probs = []
		bandits_list = self.m_arm_bandit.get_bandits()

		for bandit_idx, bandit in enumerate(bandits_list):
			bandit_pi = bandit.get_pi()

			bandit_probs.append({"bandit_idx" : bandit_idx, "pi" : bandit_pi})

		bandit_probs.sort(key = lambda x : x["pi"], reverse=True)

		return bandit_probs[0]["bandit_idx"]
