import numpy as np

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

timesteps = 10
min_X = np.zeros((num_days-timesteps, timesteps, min_spread))
min_y = np.zeros((num_days-timesteps, min_spread))
max_X = np.zeros((num_days-timesteps, timesteps, min_spread))
max_y = np.zeros((num_days-timesteps, max_spread))
for i in range(timesteps, num_days):
    day = days_list[i]
    example_num = i - timesteps

    min_y_pos = day[1] - min_min
    min_y[example_num, min_y_pos] = 1
    for j in range(timesteps):
        min_X_pos = days_list[example_num + j][1] - min_min
        min_X[example_num, j, min_X_pos] = 1

    max_y_pos = day[2] - max_min
    max_y[example_num, max_y_pos] = 1
    for j in range(timesteps):
        max_X_pos = days_list[example_num + j][2] - max_min
        max_X[example_num, j, max_X_pos] = 1


def get_one_hot_index(vec):
    for i in range(len(vec)):
        if vec[i] == 1:
            return i

# print(min_X[0, :, :])
# print(min_y[0, :])
# print(max_X[0, :, :])
# print(max_y[0, :])
for i in range(timesteps):
    print(get_one_hot_index(min_X[0, i, :]) + min_min)
print(get_one_hot_index(min_y[0, :]) + min_min)
print(days_list[0][2])
