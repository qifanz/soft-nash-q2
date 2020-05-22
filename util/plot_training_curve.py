import pickle

nash_q_rewards = None
data_dir = '../data/'
low_dim = 'LowDimensionPEG/'
snq2_dynamic_prefix = 'snq2/rewards_dynamic_'
snq2_fixed_prefix = 'snq2/rewards_fixed_'
nashq_prefix = 'nashq/rewards_'
import matplotlib.pyplot as plt
import numpy as np

AGGREGATION_FACTOR = 1000

snq2_fixed = None
snq2_dynamic = None
nashq = None


def plot_rewards():
    font_title = {
        'weight': 'normal',
        'size': 14,
    }

    font1 = {
        'weight': 'normal',
        'size': 13,
    }

    font_legend = {
        'weight': 'normal',
        'size': 12,
    }

    plt.figure()
    title = '4*4 PEG - Training curves of different algorithms'
    plt.title(title, fontdict=font_title)
    plt.xlabel('Episodes', font1)
    plt.ylabel('Reward', font1)
    plt.plot(smooth(snq2_dynamic), color='red', label='SNQ2 with uniform prior')
    plt.plot(smooth(snq2_fixed), color='blue', label='SNQ2 with dynamic schedule')
    plt.plot(smooth(nashq), color='green', label='Minimax-Q')
    plt.ylim([-0.1, 0.4])
    plt.legend(prop=font_legend)
    plt.show()


def smooth(rewards):
    rewards = np.array(rewards)
    episodes = len(rewards)
    smoothen_rewards = np.zeros(episodes - AGGREGATION_FACTOR)

    for i in range(0, episodes - AGGREGATION_FACTOR):
        smoothen_rewards[i:i + AGGREGATION_FACTOR] = np.mean(rewards[i:i + AGGREGATION_FACTOR])
    return smoothen_rewards


for i in range(5):
    file_name = data_dir + low_dim + snq2_dynamic_prefix + str(i) + '.pkl'
    f = open(file_name, 'rb')
    if snq2_dynamic is None:
        snq2_dynamic = pickle.load(f)
    else:
        snq2_dynamic = np.add(snq2_dynamic, pickle.load(f))
    f.close()
snq2_dynamic = np.divide(snq2_dynamic, 5)

for i in range(5):
    file_name = '../data/' + low_dim + snq2_fixed_prefix + str(i) + '.pkl'
    f = open(file_name, 'rb')
    if snq2_fixed is None:
        snq2_fixed = pickle.load(f)
    else:
        snq2_fixed = np.add(snq2_fixed, pickle.load(f))
    f.close()
snq2_fixed = np.divide(snq2_fixed, 5)

for i in range(1):
    file_name = '../data/' + low_dim + nashq_prefix + str(i) + '.pkl'
    f = open(file_name, 'rb')
    if nashq is None:
        nashq = pickle.load(f)
    else:
        nashq = np.add(nashq, pickle.load(f))
    f.close()
nashq = np.divide(nashq, 1)

plot_rewards()
