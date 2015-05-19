#! /usr/bin/env python3

# 2015-01-15
# Parse 20th century reanalysis version 2 netCDF file `TWENTY_PATH` into 
# winter and spring seasonal averages for pressure heights `LEVEL_TARGETS`. 
# A numpy array is sent to `OUT_FILE`. It's structure is # Dim 1 is [NDJ, FMA],
# dim 2 is levels [700, 500, 250], dim 3 is year, dim 4 is lat, dim 5 is lon.

# The netCDF4 file is from ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2/Monthlies/pressure/hgt.mon.mean.nc
    
import datetime
import numpy as np
from netCDF4 import Dataset

TWENTY_PATH = "./data/20th_rean_V2/hgt.mon.mean.nc"
ORIGIN_TIME = datetime.datetime(1800, 1, 1, 0, 0, 0)
LEVEL_TARGETS = [700, 500, 250]
OUT_FILE = "./data/20th_rean_V2.npz"

raw = Dataset(TWENTY_PATH)
dates = [ORIGIN_TIME + datetime.timedelta(hours = x) for x in raw.variables["time"][:]]
year_min = min([i.year for i in dates])
year_max = max([i.year for i in dates])
year_range = np.arange(year_min + 1, year_max)
lon, lat = np.meshgrid(raw.variables["lon"][:], raw.variables["lat"][:])
time = []
# Dim 1 is [NDJ, FMA], dim 2 is levels [700, 500, 250], dim 3 is year, dim 4 is lat, dim 5 is lon:
out = np.empty((2, len(LEVEL_TARGETS), len(year_range), raw.variables["hgt"].shape[-2], raw.variables["hgt"].shape[-1]))
for j in range(len(year_range)):
    yr = year_range[j]
    msk_ndj = []
    msk_fma = []
    time.append(yr)
    for d in range(len(dates)):
        if (dates[d] == datetime.datetime(yr - 1, 11, 1, 0, 0)) or (dates[d] == datetime.datetime(yr - 1, 12, 1, 0, 0)) or (dates[d] == datetime.datetime(yr, 1, 1, 0, 0)):
            msk_ndj.append(d)
        if (dates[d] == datetime.datetime(yr, 2, 1, 0, 0)) or (dates[d] == datetime.datetime(yr, 3, 1, 0, 0)) or (dates[d] == datetime.datetime(yr, 4, 1, 0, 0)):
            msk_fma.append(d)
    for i in range(len(LEVEL_TARGETS)):
        msk_levels = raw.variables["level"][:] == LEVEL_TARGETS[i]
        out[0, i, j, :, :] = np.mean(raw.variables["hgt"][msk_ndj, msk_levels], 0)
        out[1, i, j, :, :] = np.mean(raw.variables["hgt"][msk_fma, msk_levels], 0)
np.savez_compressed(OUT_FILE, data = out, lat = lat, lon = lon, time = time)