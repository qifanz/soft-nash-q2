import csv

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import matplotlib

plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.dpi'] = 200


def collect(file_name, total_count, frequency, total_states=132):
    data = []
    for i in range(total_count): data.append([])
    data_max = np.zeros(total_count)
    data_min = np.ones(total_count) * 99999
    f = open(file_name, 'r')
    csv_data = csv.reader(f)
    for row in csv_data:
        i = int(int(row[0]) / frequency - 1)
        correct_percentage = int(row[5])
        data[i].append(correct_percentage)
        data_max[i] = max(data_max[i], correct_percentage)
        data_min[i] = min(data_min[i], correct_percentage)
    data_median = []
    for d in data:
        data_median.append(np.median(d))
    data_median = np.divide(data_median, total_states)
    data_min /= total_states
    data_max /= total_states
    f.close()

    x = []
    for i in range(total_count):
        x.append(frequency * i)
    return data_median, data_max, data_min, x





nashq, nashq_max, nashq_min, nashq_x = collect('../data/DeterministicPEG/nashq/log_uniform_dynamic_10000_0.2.csv', 300, 1000 ,total_states=462)
snq2_dynamic, snq2_dynamic_max, snq2_dynamic_min, snq2_dynamic_x = collect(
    '../data/DeterministicPEG/snq2/log_uniform_dynamic_10000_0.2.csv', 300, 1000,total_states=462)

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

fig = plt.figure()
title = '6x6 PEG - Evolution of % of states with Nash Policy'
plt.plot(nashq_x, nashq, color='green', label='MiniMax-Q')
plt.fill_between(nashq_x,nashq_min , nashq_max, color='green', alpha=0.1)
plt.plot(snq2_dynamic_x, snq2_dynamic, color='orange', label='SNQ2 with dynamic scheduling')
plt.fill_between(snq2_dynamic_x, snq2_dynamic_min, snq2_dynamic_max, color='orange', alpha=0.1)


plt.title(title, fontdict=font_title)
plt.ylim((0.5, 1))
plt.xlabel('Episodes', font1)
plt.ylabel('% States with Nash Policy', font1)
axes = plt.axes()
axes.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
plt.legend(prop=font_legend)
plt.show()

