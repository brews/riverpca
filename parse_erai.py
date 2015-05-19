#! /usr/bin/env python3

# 2015-01-15
# Parse ERA-Interim files in `ERA_PATH` into 
# winter and spring seasonal averages for pressure heights `LEVEL_TARGETS`. 
# A numpy array is sent to `OUT_FILE`. It's structure is # Dim 1 is [NDJ, FMA],
# dim 2 is levels [700, 500, 250], dim 3 is year, dim 4 is lat, dim 5 is lon.

# This is poorly written but it's run once or twice. It will do for now.

import glob
import numpy as np
import pygrib

ERA_PATH = "./data/ERA-I/"
TARGET_FILE_ID = "ei.moda.an.pl.regn128sc."
LEVEL_TARGETS = [700, 500, 250]
OUT_FILE = "./data/erai.npz"
G_NOT = 9.80665

fls = glob.glob(ERA_PATH + TARGET_FILE_ID + "*")
fls.sort()
year_list = [int(f.split(".")[-1][:4]) for f in fls]
year_min = min(year_list)
year_max = max(year_list)
year_range = np.arange(year_min + 1, year_max + 1)
test = pygrib.open(fls[0])
lat, lon = test.select(name = "Geopotential", level = 500, typeOfLevel = "isobaricInhPa")[0].latlons()
grid_size = lat.shape
test.close()
time = []
# Dim 1 is [NDJ, FMA], dim 2 is levels [700, 500, 250], dim 3 is year, dim 4 is lat, dim 5 is lon:
out = np.empty((2, len(LEVEL_TARGETS), len(year_range), grid_size[0], grid_size[1]))
for j in range(len(year_range)):
    yr = year_range[j]
    time.append(yr)
    ndj = []
    ndj.append(ERA_PATH + TARGET_FILE_ID + str(yr - 1) + "11" + "0100")
    ndj.append(ERA_PATH + TARGET_FILE_ID + str(yr - 1) + "12" + "0100")
    ndj.append(ERA_PATH + TARGET_FILE_ID + str(yr) + "01" + "0100")
    for i in range(len(LEVEL_TARGETS)):
        sandbox = np.empty((3, grid_size[0], grid_size[1]))
        for n in range(len(ndj)):
            fl = pygrib.open(ndj[n])
            sandbox[n] = (fl.select(name = "Geopotential", level = LEVEL_TARGETS[i], typeOfLevel = "isobaricInhPa")[0].values / G_NOT)
            fl.close()
        out[0, i, j] = np.mean(sandbox, 0)
    fma = []
    fma.append(ERA_PATH + TARGET_FILE_ID + str(yr) + "02" + "0100")
    fma.append(ERA_PATH + TARGET_FILE_ID + str(yr) + "03" + "0100")
    fma.append(ERA_PATH + TARGET_FILE_ID + str(yr) + "04" + "0100")
    for i in range(len(LEVEL_TARGETS)):
        sandbox = np.empty((3, grid_size[0], grid_size[1]))
        for n in range(len(fma)):
            fl = pygrib.open(fma[n])
            sandbox[n] = (fl.select(name = "Geopotential", level = LEVEL_TARGETS[i], typeOfLevel = "isobaricInhPa")[0].values / G_NOT)
            fl.close()
        out[1, i, j] = np.mean(sandbox, 0)

np.savez_compressed(OUT_FILE, data = out, lat = lat, lon = lon, time = time)