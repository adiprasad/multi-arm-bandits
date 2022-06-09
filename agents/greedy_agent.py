from .agents import AbstractAgent
import copy

class GreedyAgent(AbstractAgent):

	def __init__(self, m_arm_bandit):
		super(GreedyAgent, self).__init__(m_arm_bandit)

	def take_action(self):
		best_bandit = self._AbstractAgent__get_best_bandit()
		greedy_bandit_idx = self.__find_bandit_greedily()

		greedy_bandit = self.m_arm_bandit.get_bandit(greedy_bandit_idx)

		reward = greedy_bandit.execute()

		is_best_action = (best_bandit == greedy_bandit)

		self._AbstractAgent__increment_episode_count()
		self._AbstractAgent__append_episode_reward(reward)
		self._AbstractAgent__append_action_type(is_best_action)

		return reward
		
		
	def __find_bandit_greedily(self):
		bandits_list = self.m_arm_bandit.get_bandits()

		bandit_list_argsort = sorted(range(len(bandits_list)), key=lambda k : bandits_list[k].get_estimate(), reverse=True)

		return bandit_list_argsort[0]





