__author__ = 'Lipman'

from pylab import *
import time
from extractData import extractData

days_list = extractData('chicago_summaries.dly')

num_days = len(days_list)
# print(num_days)
mins = [day[1] for day in days_list]
maxs = [day[2] for day in days_list]
min_min = min(mins)
max_min = max(mins)
min_max = min(maxs)
max_max = max(maxs)

min_spread = max_min - min_min + 1
max_spread = max_max - min_max + 1
print('Min min: ' + str(min_min))
print('Max min: ' + str(max_min))
print('Min max: ' + str(min_max))
print('Max max: ' + str(max_max))

print('Min spread: ' + str(min_spread))
print('Max spread: ' + str(max_spread))

times = [day[0]%365 for day in days_list]
plt.scatter(times, mins, color='blue', s=0.1)
plt.scatter(times, maxs, color='red', s=0.1)
# plt.plot(times, mins, color='green')
# plt.plot(times, maxs, color='yellow')
plt.axes().set_xlabel('time')
plt.axes().set_ylabel('temp')
plt.savefig('graphs/' + str(time.time()) + '.png')

# def plotHistogram(value, color):
#     mu, sigma=100, 15
#     hist, bins=np.histogram(value, bins=50)
#     width = 0.7 * (bins[1] - bins[0])
#     center = (bins[:-1] + bins[1:]) / 2
#     plt.bar(center, hist, align='center', width=width, color=color)
#     plt.savefig('graphs/' + str(time.time()) + '.png')
#
# plotHistogram(maxs, 'blue')
# plotHistogram(mins, 'green')
