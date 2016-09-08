#! /usr/bin/env python3
# 2016-06-03

# Grab and parse the ERSST V3b dataset.

import datetime
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import xarray as xr
import scipy.signal as signal

ERSST_PATH = "./data/ERSST_V3b/sst.mnmean.nc"
OUT_FILE = "./data/ersst.npz"
ORIGIN_TIME = datetime.datetime(1800, 1, 1, 0, 0, 0)
OUT_NC = './data/ersstv3b_season.nc'

def nandetrend(x, **kwargs):
    """Detrending of ndarray that contains nans"""
    assert len(x.shape) == 3
    nan_mask = np.isnan(x)
    x[nan_mask] = 0
    x_detrend = signal.detrend(x, **kwargs)
    x_detrend[nan_mask] = np.nan
    return x_detrend

def main():
    raw = Dataset(ERSST_PATH)
    dates = [ORIGIN_TIME + datetime.timedelta(days = x) for x in raw.variables["time"][:]]
    lon, lat = np.meshgrid(raw.variables["lon"], raw.variables["lat"])
    sst_detrend = nandetrend(raw.variables["sst"][:].filled(np.nan), type = 'linear', axis = 0)
    time = []

    year_min = min([i.year for i in dates])
    year_max = max([i.year for i in dates])
    year_range = np.arange(year_min + 1, year_max)

    out = np.empty((4, len(year_range), sst_detrend.shape[-2], sst_detrend.shape[-1]))
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
        out[0, j, :, :] = np.nanmean(sst_detrend[msk_djf], 0)
        out[1, j, :, :] = np.nanmean(sst_detrend[msk_mam], 0)
        out[2, j, :, :] = np.nanmean(sst_detrend[msk_son], 0)
        out[3, j, :, :] = np.nanmean(sst_detrend[msk_jja], 0)

    ds = xr.Dataset({'sst': (['season', 'wy', 'lat', 'lon'], out)},
                    coords = {'lon': (['lon'], raw.variables['lon'][:]),
                              'lat': (['lat'], raw.variables['lat'][:]),
                              'wy': pd.date_range(str(year_range[0])+'-01-01', str(year_range[-1])+'-01-01', freq = 'AS'),
                              'season': ['DJF', 'MAM', 'SON-1', 'JJA-1']})
    ds.attrs['units'] = raw.variables["sst"].units

    ds.to_netcdf(OUT_NC, format = 'NETCDF4', engine = 'netcdf4', 
        mode = 'w', encoding = {'sst': {'zlib': True}})

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

if __name__ == '__main__':
    main()