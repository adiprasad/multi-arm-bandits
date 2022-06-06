from .agents import AbstractAgent

class GreedyAgent(AbstractAgent):

	def __init__(self, m_arm_bandit):
		super(GreedyAgent, self).__init__(m_arm_bandit)

	def take_action(self):
		best_bandit = self._AbstractAgent__get_best_bandit()
		greedy_bandit = self.__find_bandit_greedily()

		reward = greedy_bandit.execute()

		is_best_action = (best_bandit == greedy_bandit)

		self._AbstractAgent__increment_episode_count()
		self._AbstractAgent__append_episode_reward(reward)
		self._AbstractAgent__append_action_type(is_best_action)

		return reward
		
		
	def __find_bandit_greedily(self):
		bandits_list = self.m_arm_bandit.get_bandits()

		bandits_list.sort(key= lambda x : x.get_estimate(), reverse=True)

		return bandits_list[0]





