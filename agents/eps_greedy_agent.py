from greedy_agent import GreedyAgent
import numpy as np

class EpsGreedyAgent(GreedyAgent):

	def __init__(self, m_arm_bandit, eps):
		super(EpsGreedyAgent, self).__init__(m_arm_bandit)
		self.eps = eps

	def take_action(self):
		best_bandit = self._AbstractAgent__get_best_bandit()

		if self.__should_agent_explore():
			bandit = self.__find_random_bandit()
		else:
			bandit = self._GreedyAgent__find_bandit_greedily()
			
		reward = bandit.execute()
		is_best_action = (bandit == best_bandit)

		self._AbstractAgent__increment_episode_count()
		self._AbstractAgent__append_episode_reward(reward)
		self._AbstractAgent__append_action_type(is_best_action)


	def __find_random_bandit(self):
		bandits_list = self.m_arm_bandit.get_bandits()

		random_bandit = np.random.choice(bandits_list)

		return random_bandit


	def __should_agent_explore(self):
		return (np.random.uniform() <= self.eps)