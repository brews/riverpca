"""Parse geopotential height and SST fields
"""

import numpy as np
import xarray as xr
import scipy.signal as signal

# OPeNDAP URLs. Change these to local paths if you don't want to download these.
MONTHLY_HGT = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/20thC_ReanV2/Monthlies/pressure/hgt.mon.mean.nc'
MONTHLY_SST = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/noaa.ersst.v3/sst.mnmean.nc'

def tcr(outpath=None, inpath=MONTHLY_HGT):
    """Parse TCR fields into local netCDF file
    """
    ds = load_seasonal(inpath)
    ds = (ds[['hgt']].sel(level = 500)
                     .squeeze()
                     .rename({'hgt': 'z500'}))
    ds.coords['wy'] = ('time', get_wy_labels(ds))
    ds.coords['season'] = ('time', get_season_labels(ds))
    if outpath:
        ds.to_netcdf(outpath, format = 'NETCDF4', engine = 'netcdf4',
                     mode = 'w', encoding = {'z500': {'zlib': True}})
    else:
        return ds

def ersst(outpath=None, inpath=MONTHLY_SST):
    """Parse ERSST fields into local netCDF file
    """
    ds = load_seasonal(inpath)
    ds = ds[['sst']].squeeze()
    # This is a bit clunky. Detrend and fill any gridpoint that has NAN at 
    # anytime with np.nan for the entire analysis.
    sstarr = nandetrend(ds['sst'].values)
    assert ds['sst'].dims.index('time') == 0
    sstarr[:, np.isnan(sstarr).any(axis = 0)] = np.nan
    ds['sst'] = (ds['sst'].dims, sstarr)

    ds.coords['wy'] = ('time', get_wy_labels(ds))
    ds.coords['season'] = ('time', get_season_labels(ds))
    if outpath:
        ds.to_netcdf(outpath, format = 'NETCDF4', engine = 'netcdf4',
                     mode = 'w', encoding = {'sst': {'zlib': True}})
    else:
        return ds

def load_seasonal(path):
    """Read data and average into 3-month seasons
    """
    with xr.open_dataset(path) as d:
        out = d.resample('QS-DEC', dim = 'time', how = 'mean', 
                         label = 'left', keep_attrs = True)
    return out

def get_wy_labels(ds):
    """Get wateryear for Dataset time dimension
    """
    yr = ds['time.year']
    mon = ds['time.month']
    wy = yr.copy()
    wy[np.in1d(mon, [6, 9, 12])] += 1
    return wy.values

def get_season_labels(ds):
    """Get season labels for Dataset, for plotting
    """
    key = {6: 'JJA-1', 9: 'SON-1', 12: 'DJF', 3: 'MAM'}
    converted = [key[x] for x in ds['time.month'].values]
    return converted

def nandetrend(x, **kwargs):
    """Detrend ndarray that contains nans
    """
    nan_mask = np.isnan(x)
    x[nan_mask] = 0 
    x_detrend = signal.detrend(x, type = 'linear', axis = 0)
    x_detrend[nan_mask] = np.nan
    return x_detrend
