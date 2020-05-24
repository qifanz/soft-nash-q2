import csv

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import matplotlib


plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
total_count = 100
frequency = 2000


def collect(file_name, n_experiences, total_count, frequency,total_states= 132):
    data = np.zeros(total_count)
    data_max = np.zeros(total_count)
    data_min = np.ones(total_count) * 99999
    f = open(file_name, 'r')
    csv_data = csv.reader(f)
    for row in csv_data:
        i = int(int(row[0]) / frequency - 1)
        correct_percentage = int(row[5])
        data[i] += correct_percentage
        data_max[i] = max(data_max[i], correct_percentage)
        data_min[i] = min(data_min[i], correct_percentage)
    data = data / n_experiences
    data /= total_states
    data_min /= total_states
    data_max /= total_states
    f.close()

    x = []
    for i in range(total_count):
        x.append(frequency * i)
    return data, data_max, data_min,x


def collect_single_line(file_name):
    data, data_max, data_min = collect(file_name)
    return np.ones(total_count) * data[-1], np.ones(total_count) * data_max[-1], np.ones(total_count) * data_min[-1]

nashq, nashq_max, nashq_min, nashq_x = collect('../data/LowDimensionPEG/nashq/log0.1.csv',5,100,2000)
snq2, snq2_max, snq2_min,snq2_x = collect('../data/LowDimensionPEG/snq2/log_uniform_fixed_10000_0.1.csv', 5,200,1000)
snq22, snq2_max2, snq2_min2,snq2_x2 = collect('../data/LowDimensionPEG/snq2/log_uniform_fixed_20000_0.1.csv', 5,200,1000)

snq2_dynamic, snq2_dynamic_max, snq2_dynamic_min,snq2_dynamic_x = collect('../data/LowDimensionPEG/snq2/log_uniform_dynamic_10000_0.1.csv', 5,200,1000)
snq2_prior, snq2_prior_max, snq2_prior_min,snq2_prior_x = collect('../data/LowDimensionPEG/snq2/log_quasi-nash_dynamic_10000_0.15_polyak.csv', 5,200,1000)

snq2_h, snq2_h_max, snq2_h_min, snq2_h_x = collect('../data/HighDimensionPEG/snq2/log_uniform_dynamic_30000.csv',2, 120, 5000, total_states=2862)
snq2hfix, snq2hfix_max, snq2hfix_min, snq2hfix_x = collect('../data/HighDimensionPEG/snq2/log_uniform_fixed_30000.csv',1,120,5000,total_states=2862)
nashq_h, nashq_h_max, nashq_h_min, nashq_h_x = collect('../data/HighDimensionPEG/nashq/log.csv',1,56,5000,total_states=2862)
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
title = '4*4 PEG - Evolution of % of states with Nash Policy'
plt.plot(nashq_x, nashq, color='green', label='MiniMax-Q')
plt.fill_between(nashq_x,nashq_min , nashq_max, color='green', alpha=0.3)
#plt.plot(snq2_x, snq2, color='red', label='SNQ2 M=10k')
#plt.plot(snq2_x2, snq22, color='brown', label='SNQ2 M=20k')
plt.plot(snq2_prior_x, snq2_prior, color = 'blue',label='SNQ2 prior')
plt.fill_between(snq2_prior_x, snq2_prior_min, snq2_prior_max, color='blue', alpha=0.3)
plt.plot(snq2_dynamic_x, snq2_dynamic, color='orange', label='SNQ2 with dynamic scheduling')
plt.fill_between(snq2_dynamic_x, snq2_dynamic_min, snq2_dynamic_max, color='orange', alpha=0.3)
#plt.plot(snq2_prior_x, snq2_prior, color='blue', label='SNQ2 with previous experience')
#plt.fill_between(snq2_prior_x, snq2_prior_min, snq2_prior_max, color='blue', alpha=0.3)

plt.title(title, fontdict=font_title)
plt.ylim((0.5,1))
plt.xlabel('Episodes', font1)
plt.ylabel('% States with Nash Policy', font1)
axes = plt.axes()
axes.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
plt.legend(prop=font_legend)
plt.show()


fig = plt.figure()
title = '8*8 PEG - Evolution of % of states with Nash Policy'
plt.plot(snq2_h_x, snq2_h, color='orange', label='SNQ2 with dynamic scheduling')
plt.plot(snq2hfix_x, snq2hfix, color='green', label='SNQ2 M=30k fixed')

plt.plot(nashq_h_x, nashq_h, color = 'blue', label='Nash-Q')

plt.title(title, fontdict=font_title)
plt.ylim((0.35,1))
plt.xlabel('Episodes', font1)
plt.ylabel('% States with Nash Policy', font1)
axes = plt.axes()
axes.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
plt.legend(prop=font_legend)
plt.show()
