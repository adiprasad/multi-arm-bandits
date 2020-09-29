from agents import AbstractAgent
import numpy as np

class UcbAgent(AbstractAgent):

	def __init__(self, m_arm_bandit, ucb_c):
		super(UcbAgent, self).__init__(m_arm_bandit)
		self.ucb_c = ucb_c

	
	def take_action(self):
		best_bandit = self._AbstractAgent__get_best_bandit()
		bandit = self.find_next_bandit()

		reward = bandit.execute()

		is_best_action = (best_bandit == bandit)

		self._AbstractAgent__increment_episode_count()
		self._AbstractAgent__append_episode_reward(reward)
		self._AbstractAgent__append_action_type(is_best_action)

		return reward


	def find_next_bandit(self):
		bandit_estimates_list = []
		bandits_list = self.m_arm_bandit.get_bandits()

		for bandit in bandits_list:
			bandit_estimate = bandit.get_estimate()
			bandit_ucb_factor, untried_bandit = self.__get_ucb_factor_for_bandit(bandit)

			## Part of algorithm
			if (untried_bandit is True):
				return bandit

			ucb_estimate = bandit_estimate + bandit_ucb_factor

			bandit_estimates_list.append({"bandit" : bandit, "est" : ucb_estimate})

		bandit_estimates_list.sort(key = lambda x : x["est"], reverse=True)

		return bandit_estimates_list[0]["bandit"]


	def __get_ucb_factor_for_bandit(self, bandit):
		numerator = 0
		frac = 0

		num_episodes = self.get_episode_count()
		
		if num_episodes > 0:
			numerator = np.log(num_episodes)
		
		bandit_experience = bandit.get_experience_len()
		untried_bandit = (bandit_experience == 0)
		
		if bandit_experience > 0:
			frac = np.sqrt(numerator/float(bandit_experience))

		return self.ucb_c * frac, untried_bandit
