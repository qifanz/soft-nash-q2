import csv

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

total_count = 100
frequency = 2000
total_states = 132


def collect(file_name, n_experiences):
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
    return data, data_max, data_min


def collect_single_line(file_name):
    data, data_max, data_min = collect(file_name)
    return np.ones(total_count) * data[-1], np.ones(total_count) * data_max[-1], np.ones(total_count) * data_min[-1]


snq2, snq2_max, snq2_min = collect('../data/LowDimensionPEG/snq2_log_uniform.csv', 6)
snq2_dynamic, snq2_dynamic_max, snq2_dynamic_min = collect('../data/LowDimensionPEG/snq2_log_uniform_30k.csv', 5)
x = []
for i in range(total_count):
    x.append(frequency * i)

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
# plt.plot(x, nash_q, color='green', label='MiniMax-Q')
# plt.fill_between(x, nash_q_min, nash_q_max, color='green', alpha=0.3)
plt.plot(x, snq2, color='red', label='SNQ2 with uniform prior')
plt.fill_between(x, snq2_min, snq2_max, color='red', alpha=0.3)
plt.plot(x, snq2_dynamic, color='blue', label='SNQ2 with previous experience')
plt.fill_between(x, snq2_dynamic_min, snq2_dynamic_max, color='blue', alpha=0.3)

plt.title(title, fontdict=font_title)
# plt.ylim((60,132))
plt.xlabel('Episodes', font1)
plt.ylabel('% States with Nash Policy', font1)
axes = plt.axes()
axes.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
plt.legend(prop=font_legend)
plt.show()
