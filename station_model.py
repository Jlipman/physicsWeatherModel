import numpy as np
import h5py
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras.regularizers import l2  # , activity_l2
from keras.layers.advanced_activations import LeakyReLU

with open('chicago_summaries.dly') as file:
    text = file.read()

lines = text.split('\n')[:-1]

'''{ID: (TMIN, TMAX)}'''
days = {}

for line in lines[0:]:
    id = line[0:11]
    year = int(line[11:15])
    month = int(line[15:17])
    element = line[17:21]
    _char = 21
    for i in range(0, 31):
        value = int(line[_char:_char+5])/10
        id = str(year) + str(month).zfill(2) + str(i + 1).zfill(2)
        if element == 'TMIN':
            try:
                days[id] = (value, days[id][1])
            except KeyError:
                days[id] = (value, None)
        elif element == 'TMAX':
            try:
                days[id] = (days[id][0], value)
            except KeyError:
                days[id] = (None, value)
        _char += 8

days_list = []
keys = list(days.keys())
keys.sort()
for day in keys:
    days_list.append((int(day), days[day][0], days[day][1]))


def get_closest(i, days_list, position):
    j = i
    while j >= 0 and days_list[j][position] == -999.9:
        j -= 1
    j_correct = not days_list[j][position] == -999.9
    k = i
    while k < len(days_list) - 1 and days_list[k][position] == -999.9:
        k += 1
    k_correct = not days_list[k][position] == -999.9
    if j_correct and (i - j < k - i or not k_correct):
        return days_list[j][position]
    else:
        return days_list[k][position]

for i, day in enumerate(days_list):
    if day[1] == -999.9:
        closest_min = get_closest(i, days_list, 1)
        days_list[i] = (day[0], closest_min, day[2])
    day = days_list[i]
    if day[2] == -999.9:
        closest_max = get_closest(i, days_list, 2)
        days_list[i] = (day[0], day[1], closest_max)

for i, day in enumerate(days_list):
    days_list[i] = (day[0], round(day[1]), round(day[2]))
    # print(days_list[i])

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
# print('Min min: ' + str(min_min))
# print('Max min: ' + str(max_min))
# print('Min max: ' + str(min_max))
# print('Max max: ' + str(max_max))

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

split = num_days/4

min_train_X = min_X[:-split, :, :]
min_train_y = min_y[:-split, :]
min_test_X = min_X[-split:, :, :]
min_test_y = min_y[-split:, :]

max_train_X = max_X[:-split, :, :]
max_train_y = max_y[:-split, :]
max_test_X = max_X[-split:, :, :]
max_test_y = max_y[-split:, :]

dataset = [min_train_X, min_train_y, min_test_X, min_test_y,
           max_train_X, max_train_y, max_test_X, max_test_y]
names = ['min_train_X', 'min_train_y', 'min_test_X', 'min_test_y',
         'max_train_X', 'max_train_y', 'max_test_X', 'max_test_y']

for data, name in zip(dataset, names):
    h5f = h5py.File(name + '.h5', 'w')
    h5f.create_dataset(name, data=data)
    h5f.close()

print('Build min model...')
model = Sequential()

model.add(LSTM(512, return_sequences=True,
               input_shape=(timesteps, min_spread), init='glorot_normal',
               W_regularizer=l2(0.01)))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))

model.add(LSTM(512, return_sequences=False, init='glorot_normal',
               W_regularizer=l2(0.01)))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))

model.add(Dense(min_spread, W_regularizer=l2(0.01), init='glorot_normal'))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))

model.add(Dense(min_spread, W_regularizer=l2(0.01), init='glorot_normal'))
model.add(Activation('softmax'))

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)

model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
model.fit(min_train_X, min_train_y, batch_size=2048, nb_epoch=1000,
          validation_split=0.1, show_accuracy=True)
