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
    print(days_list[i])

print(len(days_list))
print(keys[0])
print(keys[1])
print(keys[-2])
print(keys[-1])
mins = [day[1] for day in days_list]
maxs = [day[2] for day in days_list]
min_min = min(mins)
max_min = max(mins)
min_max = min(maxs)
max_max = max(maxs)
print('Min min: ' + str(min_min))
print('Max min: ' + str(max_min))
print('Min max: ' + str(min_max))
print('Max max: ' + str(max_max))

print('Min spread: ' + str(max_min - min_min + 1))
print('Max spread: ' + str(max_max - min_max + 1))
