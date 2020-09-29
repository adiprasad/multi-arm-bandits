from abc import ABCMeta, abstractmethod

class AbstractAgent(object):
	__metaclass__ = ABCMeta

	def __init__(self, m_arm_bandit):
		self.m_arm_bandit = m_arm_bandit
		self.episode_cnt = 0		# Keep track of the number of episodes
		self.rewards_list = []
		self.optimal_action_record_list = []		# For each episode, keep track of whether taken action was optimal or not

	@abstractmethod
	def take_action(self):
		pass

	def get_optimal_action_record_list(self):
		return self.optimal_action_record_list

	def get_episode_count(self):
		return self.episode_cnt

	def get_rewards_list(self):
		return self.rewards_list

	def __get_best_bandit(self):
		return self.m_arm_bandit.get_best_bandit()

	def __increment_episode_count(self):
		self.episode_cnt+=1

	def __append_action_type(self, was_action_optimal):
		self.optimal_action_record_list.append(was_action_optimal)

	def __append_episode_reward(self, reward):
		self.rewards_list.append(reward)





