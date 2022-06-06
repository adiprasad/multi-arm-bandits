import numpy as np
import argparse
from agents import greedy_agent, eps_greedy_agent, ucb_agent, gradient_agent
from bandits.m_arm_bandit import MultiArmBandit
from bandits.m_arm_gradient_bandit import MultiArmGradientBandit
import matplotlib.pyplot as plt
import os

def parse_args():
	parser = argparse.ArgumentParser(description = 'Multi Arm Bandit Experiments')
	
	parser.add_argument('--num-bandits', dest='num_bandits', help='Number of bandits',type=int, required=True)
	parser.add_argument('--bandit-sampling-mean', dest='bandit_sampling_mean', help='Mean of Gaussian sampling bandit Q*(a)',type=float, required=True)
	parser.add_argument('--bandit-sampling-std-dev', dest='bandit_sampling_std_dev', help='Std dev of Gaussian sampling bandit Q*(a)',type=float, required=True)
	parser.add_argument('--bandit-std-dev', dest='bandit_std_dev', help='Std dev in reward/return from Q*(a)',type=float, required=True)
	parser.add_argument('--mode', dest='mode', help='Mode or the algorithm to follow, eps for eps greedy, default for greedy, ucb for upper confidence bound', type=str, required=True)
	parser.add_argument('--eps', dest='eps', help='Exploration probability', type=float, required=False, default = 0.0)
	parser.add_argument('--alpha', dest='alpha', help='Step size, Leave blank for 1/n', type=float, required=False, default = None)
	parser.add_argument('--init-vals', dest='init_val', help='Initial Q estimates', type=float, required=False, default = 0.0)
	parser.add_argument('--ucb-degree-exploration', dest='ucb_c', help='', type=float, required=False, default=1.0)
	parser.add_argument('--gradient-step-size', dest='gradient_alpha', help='', type=float, required=False, default=0.2)
	parser.add_argument('--gradient-agent-reward-baseline', dest='gradient_rewards_baseline', help='', type=float, required=False, default=None)
	parser.add_argument('--num-episodes', dest='num_episodes', help='', type=int, required=True)
	parser.add_argument('--num-trials', dest='num_trials', help='', type=int, required=True)

	args = parser.parse_args()

	return args

'''
Initializes and returns a list of bandits 

Inputs :-

num_bandits : Number of bandits
bandit_sampling_mean : Mean

'''
def init_bandits(num_bandits, bandit_sampling_mean, bandit_sampling_std_dev, bandit_std_dev, alpha, init_val):
	m_arm_bandit = MultiArmBandit(num_bandits, bandit_sampling_mean, bandit_sampling_std_dev, bandit_std_dev, alpha, init_val)
	m_arm_bandit.init_bandits()

	return m_arm_bandit

def init_gradient_bandits(num_bandits, bandit_sampling_mean, bandit_sampling_std_dev, bandit_std_dev, alpha, gradient_step_size, init_val):
	m_arm_bandit = MultiArmGradientBandit(num_bandits, bandit_sampling_mean, bandit_sampling_std_dev, bandit_std_dev, alpha, gradient_step_size, init_val)
	m_arm_bandit.init_bandits()

	#sys.exit()

	return m_arm_bandit

def greedy_algo(m_arm_bandit):
	agent = greedy_agent.GreedyAgent(m_arm_bandit)

	return agent


def eps_greedy_algo(m_arm_bandit, eps):
	agent = eps_greedy_agent.EpsGreedyAgent(m_arm_bandit, eps)

	return agent


def ucb_algo(m_arm_bandit, ucb_c):
	agent = ucb_agent.UcbAgent(m_arm_bandit, ucb_c)

	return agent

def gradient_algo(m_arm_bandit, baseline=None):
	agent = gradient_agent.GradientAgent(m_arm_bandit, baseline)

	return agent


def run_trial(agent, num_episodes):
	for i in range(num_episodes):
		agent.take_action()

	return agent.get_rewards_list(), agent.get_optimal_action_record_list()



def run_experiment(num_bandits, bandit_sampling_mean, bandit_sampling_std_dev, bandit_std_dev, mode, eps, alpha, gradient_step_size, gradient_rewards_baseline, init_val, ucb_c, num_episodes, num_trials):
	experiment_rewards = np.zeros((1, num_episodes))
	bool_optimal_action_record = np.zeros((1, num_episodes), dtype=bool)

	# print "Number of trials"
	# print num_trials

	for trial in range(num_trials):
		#print "Trial " + str(trial)

		if (mode != "gradient"):
			m_arm_bandit = init_bandits(num_bandits, bandit_sampling_mean, bandit_sampling_std_dev, bandit_std_dev, alpha, init_val)
		else:
			m_arm_bandit = init_gradient_bandits(num_bandits, bandit_sampling_mean, bandit_sampling_std_dev, bandit_std_dev, alpha, gradient_step_size, init_val)

		if mode == "greedy":
			agent = greedy_algo(m_arm_bandit)
		elif mode == "eps-greedy":
			agent = eps_greedy_algo(m_arm_bandit, eps)
		elif mode == "ucb":
			agent = ucb_algo(m_arm_bandit, ucb_c)
		elif mode == "gradient":
			agent = gradient_algo(m_arm_bandit, gradient_rewards_baseline)

		rewards_list, optimal_action_record_list = run_trial(agent, num_episodes)

		# print "Rewards list shape"
		# print np.array(rewards_list).shape

		experiment_rewards = np.vstack((experiment_rewards, np.array((rewards_list))))

		if len(optimal_action_record_list) > 0:
			bool_optimal_action_record = np.vstack((bool_optimal_action_record, np.array((optimal_action_record_list))))


		mean_rewards = np.mean(experiment_rewards, axis = 0)
		mean_optimal_action_percentage = np.mean(bool_optimal_action_record, axis = 0) * 100

	return mean_rewards, mean_optimal_action_percentage


def plot_data(data_arr, plot_file_dir, experiment_mode, num_trials, type_of_plot="Rewards"):
		if not os.path.exists(plot_file_dir):
			os.makedirs(plot_file_dir)

		plot_title = experiment_mode + "_" + type_of_plot + "_" + str(num_trials)
		filename = plot_title +'.png'

		filepath = os.path.join(plot_file_dir, filename)

		# x = np.array(range(data_arr.shape[0]))
		# y = np.mean(data_arr, axis=1)
		# e = np.std(data_arr, axis=1)
		# plt.errorbar(x, y, e, linestyle='--', marker='o')

		x = np.array(range(1, len(data_arr) + 1))

		print("Length of x") 
		print(len(x))

		print("Len of datarr")
		print(len(data_arr))

		plt.plot(x, data_arr)

		x_label = "Number of episodes"
		y_label = "Mean " + type_of_plot

		plt.suptitle(plot_title, fontsize=14, fontweight='bold')

		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.savefig(filepath)
		plt.clf()
		plt.close()

		print('Plot saved in {0}'.format(filepath))

if __name__=="__main__":
	args = parse_args()

	print(args)

	rewards, optimal_action_per = run_experiment(args.num_bandits, args.bandit_sampling_mean, args.bandit_sampling_std_dev, args.bandit_std_dev, args.mode, args.eps, args.alpha, args.gradient_alpha, args.gradient_rewards_baseline, args.init_val, args.ucb_c, args.num_episodes, args.num_trials)
	# print "Rewards shape"
	# print rewards.shape
	# print "Optimal action shape"
	# print optimal_action_per.shape

	print(rewards)
	print(optimal_action_per)

	plot_data(rewards, os.path.join(os.getcwd(), "plots"), args.mode, args.num_trials, "Rewards")
	plot_data(optimal_action_per, os.path.join(os.getcwd(), "plots"), args.mode, args.num_trials, "Optimal Action Percentage")














	
