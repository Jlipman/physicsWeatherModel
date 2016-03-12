import urllib.request

def main():
    for year in range(2003, 2011):
        for month in range(1, 13):
            for day in range(1, 32):
                url = "http://www.wpc.ncep.noaa.gov/dailywxmap/htmlimages/colormaxmin_" + str(year) + str(month).zfill(2) + str(day).zfill(2) + ".gif"
                print(url)
                try:
                    urllib.request.urlretrieve(url, "TempMaps/" + url.split('/')[-1])
                except:
                    pass
main()
