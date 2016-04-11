#! /usr/bin/env python3
# 2016-04-11

# Grab and parse NOAA OI V2 SST dataset.    

import datetime
import numpy as np
from netCDF4 import Dataset

OI_PATH = "./data/NOAA_OI_V2/sst.mnmean.nc"
OI_LAND_PATH = "./data/NOAA_OI_V2/lsmask.nc"
OUT_FILE = "./data/oisst.npz"
ORIGIN_TIME = datetime.datetime(1800, 1, 1, 0, 0, 0)

def main():
    raw = Dataset(OI_PATH)
    dates = [ORIGIN_TIME + datetime.timedelta(days = x) for x in raw.variables["time"][:]]
    lat = raw.variables["lat"]
    lon = raw.variables["lon"]
    lon, lat = np.meshgrid(lon[:], lat[:])
    lnd = Dataset(OI_LAND_PATH)
    msk = lnd.variables["mask"][:][0]
    time = []

    # year_min = min([i.year for i in dates])
    year_min = 1982
    year_max = max([i.year for i in dates])
    year_range = np.arange(year_min + 1, year_max)

    out = np.empty((4, len(year_range), raw.variables["sst"].shape[-2], raw.variables["sst"].shape[-1]))
    for j in range(len(year_range)):
        yr = year_range[j]
        time.append(yr)
        msk_djf = []
        msk_mam = []
        msk_son = [] # Antecedent fall
        msk_jja = [] # Antecedent summer
        for d in range(len(dates)):
            if (dates[d] == datetime.datetime(yr - 1, 12, 1, 0, 0)) or (dates[d] == datetime.datetime(yr, 1, 1, 0, 0)) or (dates[d] == datetime.datetime(yr, 2, 1, 0, 0)):
                msk_djf.append(d)
            if (dates[d] == datetime.datetime(yr, 3, 1, 0, 0)) or (dates[d] == datetime.datetime(yr, 4, 1, 0, 0)) or (dates[d] == datetime.datetime(yr, 5, 1, 0, 0)):
                msk_mam.append(d)
            if (dates[d] == datetime.datetime(yr - 1, 9, 1, 0, 0)) or (dates[d] == datetime.datetime(yr - 1, 10, 1, 0, 0)) or (dates[d] == datetime.datetime(yr - 1, 11, 1, 0, 0)):
                msk_son.append(d)
            if (dates[d] == datetime.datetime(yr - 1, 6, 1, 0, 0)) or (dates[d] == datetime.datetime(yr - 1, 7, 1, 0, 0)) or (dates[d] == datetime.datetime(yr - 1, 8, 1, 0, 0)):
                msk_jja.append(d)
        out[0, j, :, :] = np.mean(raw.variables["sst"][msk_djf], 0)
        out[1, j, :, :] = np.mean(raw.variables["sst"][msk_mam], 0)
        out[2, j, :, :] = np.mean(raw.variables["sst"][msk_son], 0)
        out[3, j, :, :] = np.mean(raw.variables["sst"][msk_jja], 0)
    np.savez_compressed(OUT_FILE, data = out, lat = lat, lon = lon, time = time, landmask = ~(msk == True))

# # Plot on map for sanity checks.
# sst_masked = np.ma.masked_array(out[1, 0], ~(msk == True))
# plt.figure()
# m = Basemap(projection = "robin", lon_0= 180, resolution = "c")
# lon, lat = np.meshgrid(lon[:], lat[:])
# x, y = m(lon, latmesh)
# m.drawcoastlines()
# m.drawparallels(np.arange(-90.,120.,30.))
# m.drawmeridians(np.arange(0.,360.,60.))
# m.drawmapboundary(fill_color = "0.7")
# m.pcolormesh(x, y, sst_masked)
# cb = m.colorbar()
# cb.set_label("SST (C)")
# plt.title("NDJ NOAA OI SST V2")
# plt.show()

if __name__ == '__main__':
    main()
