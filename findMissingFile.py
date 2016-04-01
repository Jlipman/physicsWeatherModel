import os
for i in os.listdir('PrecipMaps'):
    file = i.split('_')[1].split('.')[0]
    if not os.path.exists('TempMaps/colormaxmin_' + file + '.gif'):
        print file

for i in os.listdir('TempMaps'):
    file = i.split('_')[1].split('.')[0]
    if not os.path.exists('PrecipMaps/precip_' + file + '.gif'):
        print file