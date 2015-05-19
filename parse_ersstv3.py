#! /usr/bin/env python3
# 2015-01-21

# Grab and parse the ERSST V3b dataset.

import datetime
import numpy as np
from netCDF4 import Dataset

ERSST_PATH = "./data/ERSST_V3b/sst.mnmean.nc"
OUT_FILE = "./data/ersst.npz"
ORIGIN_TIME = datetime.datetime(1800, 1, 1, 0, 0, 0)

raw = Dataset(ERSST_PATH)
dates = [ORIGIN_TIME + datetime.timedelta(days = x) for x in raw.variables["time"][:]]
lat = raw.variables["lat"]
lon = raw.variables["lon"]
lon, lat = np.meshgrid(lon[:], lat[:])
sst = raw.variables["sst"]
time = []

# year_min = min([i.year for i in dates])
year_min = min([i.year for i in dates])
year_max = max([i.year for i in dates])
year_range = np.arange(year_min + 1, year_max)

out = np.empty((2, len(year_range), raw.variables["sst"].shape[-2], raw.variables["sst"].shape[-1]))
for j in range(len(year_range)):
    yr = year_range[j]
    time.append(yr)
    msk_ndj = []
    msk_fma = []
    for d in range(len(dates)):
        if (dates[d] == datetime.datetime(yr - 1, 11, 1, 0, 0)) or (dates[d] == datetime.datetime(yr - 1, 12, 1, 0, 0)) or (dates[d] == datetime.datetime(yr, 1, 1, 0, 0)):
            msk_ndj.append(d)
        if (dates[d] == datetime.datetime(yr, 2, 1, 0, 0)) or (dates[d] == datetime.datetime(yr, 3, 1, 0, 0)) or (dates[d] == datetime.datetime(yr, 4, 1, 0, 0)):
            msk_fma.append(d)
    out[0, j, :, :] = np.mean(raw.variables["sst"][msk_ndj], 0)
    out[1, j, :, :] = np.mean(raw.variables["sst"][msk_fma], 0)

landmask = out[0, 0] == 0
np.savez_compressed(OUT_FILE, data = out, lat = lat, lon = lon, time = time, landmask = landmask)


# Plot on map for sanity checks.
# plt.figure()
# m = Basemap(projection = "robin", lon_0= 180, resolution = "c")
# x, y = m(lon, lat)
# m.drawcoastlines()
# m.drawparallels(np.arange(-90.,120.,30.))
# m.drawmeridians(np.arange(0.,360.,60.))
# m.drawmapboundary(fill_color = "0.7")
# m.pcolormesh(x, y, sst[:][0])
# cb = m.colorbar()
# cb.set_label("SST (C)")
# plt.title("NDJ ERSST V3b")
# plt.show()