""" Special plotting and stats functions of varying quality
"""

import sqlite3
import string
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats as stats

import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.path as mplpath
import seaborn as sns

import cartopy
import cartopy.util
import cartopy.crs as ccrs
from mpl_toolkits.basemap import Basemap


def pointfield_statistic(field, pc_df, stat_fun, stat_name='statistic', units=''):
    """Calculate point-field statistic hypothesis test

    Args:
    field : An xarray DataArray with dimensions for season, PC, lat, lon.
    pc_df : A pandas DataFrame of principal components.
    stat_fun : A function, f(s, a), that performs a hypothesis test when 
        given a 1D array-like time series ('s') and a 2D (lat, lon) array-like
         field ('a'). The function must return two arrays with the same shape
          as 'a'. The first gives the statistic. The second gives a p-value 
        for the hypothesis test.
    stat_name : A string giving the name of the statistic returned by 
        'stat_fun'. Default is simply 'statistic'.
    units : Optional string that gives the units for the statistic. This will 
        be passed on to the 'units' attribute of the statistic in the returned 
        DataSet.

    Returns:
    An xarray dataset with variables for the statistic and pvalue of the 
    hypothesis test.

    This might work better as a DataSet subclass.
    """
    season = ['JJA-1', 'SON-1', 'DJF', 'MAM']
    pc_dim = ['PC1', 'PC2']
    out_lat = field['lat'].values
    out_lon = field['lon'].values
    out_shape = (len(pc_dim), len(season), len(out_lat), len(out_lon))
    out = xr.Dataset({stat_name: (['PC','season', 'lat', 'lon'], np.zeros(out_shape)),
                      'pvalue': (['PC','season', 'lat', 'lon'], np.zeros(out_shape))},
                     coords = {'PC': pc_dim,
                               'season': season,
                               'lon': out_lon,
                               'lat': out_lat})
    for pc_i in pc_dim:
        for seas_i in season:
            s, pvalue = stat_fun(pc_df[pc_i].values, field[field['season'] == seas_i].values)
            out[stat_name].loc[dict(PC = pc_i, season= seas_i)] = s
            out['pvalue'].loc[dict(PC = pc_i, season= seas_i)] = pvalue
    if units:
        out[stat_name].attrs['units'] = units
    return out


def plot_pointfield_statistic(ds, map_type, stat_name, sig_alpha=0.05, plotfun=None, **kwargs):
    """Generic plotting function for field statistic manuscript figures
    """
    assert map_type in ['north_hemisphere', 'global']
    assert plotfun in ['contourf', 'pcolormesh']

    proj = {'north_hemisphere': ccrs.LambertAzimuthalEqualArea(central_longitude = -160, central_latitude = 90),
            'global': ccrs.Robinson(central_longitude = -160)}
    subplot_kwargs = {'north_hemisphere': {'adjust': {'bottom': 0.1},
                                         'add_axes': [0.17, 0.1, 0.7, 0.01],
                                         'colorbar': {'orientation': 'horizontal'}},
                      'global': {'adjust': {'bottom': 0.1},
                                 'add_axes': [0.17, 0.1, 0.7, 0.01],
                                 'colorbar': {'orientation': 'horizontal'}}
                     }

    pc_dim = ds['PC'].values
    season = ds['season'].values

    alpha = [0, sig_alpha, 1]

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mplpath.Path(verts * radius + center)

    plot_meta = [{'season': 'JJA-1',
                  'PC':     'PC1',
                  'title':  'a) PC1: JJA'},
                 {'season': 'SON-1',
                  'PC':     'PC1',
                  'title':  'b) PC1: SON'},
                 {'season': 'DJF',
                  'PC':     'PC1',
                  'title':  'c) PC1: DJF'},
                 {'season': 'MAM',
                  'PC':     'PC1',
                  'title':  'd) PC1: MAM'},
                 {'season': 'JJA-1',
                  'PC':     'PC2',
                  'title':  'e) PC2: JJA'},
                 {'season': 'SON-1',
                  'PC':     'PC2',
                  'title':  'f) PC2: SON'},
                 {'season': 'DJF',
                  'PC':     'PC2',
                  'title':  'g) PC2: DJF'},
                 {'season': 'MAM',
                  'PC':     'PC2',
                  'title':  'h) PC2: MAM'}]

    fig = plt.figure(figsize = (7.48031, 4.52756))
    for i in range(len(pc_dim) * len(season)):
        i_season = plot_meta[i]['season']
        i_pc = plot_meta[i]['PC']
        i_title = plot_meta[i]['title']
        ax = fig.add_subplot(len(pc_dim), len(season), i + 1, 
                             projection = proj[map_type])
        ax.gridlines(color = '#696969',
                     linewidth = 0.5)
        ds_crop = ds.sel(season = i_season, PC = i_pc)
        if map_type == 'north_hemisphere':
            ax.coastlines(color = '#696969', linewidth = 1)
            ax.set_boundary(circle, transform = ax.transAxes)
            ds_crop = ds_crop.sel(lat = slice(90, 0))
        else:
            ax.set_extent([100, 300, -90, 90])
            ax.add_feature(cartopy.feature.LAND, 
                           edgecolor = '#696969', facecolor = '#696969', 
                           zorder = 0)
        stat_crop = ds_crop[stat_name]
        p_crop = ds_crop['pvalue']
        p_cyc, lon_cyc = cartopy.util.add_cyclic_point(p_crop.values, 
                                                       coord = p_crop['lon'].values, 
                                                       axis = -1)
        stat_cyc, lon_cyc = cartopy.util.add_cyclic_point(stat_crop.values, 
                                                       coord = stat_crop['lon'].values, 
                                                       axis = -1)
        # For whatever reason, this screws with our NAN masks, so to correct:
        stat_cyc = np.ma.masked_where(np.isnan(stat_cyc), stat_cyc)

        ctf1 = None
        if plotfun == 'pcolormesh':
            ctf1 = ax.pcolormesh(lon_cyc, stat_crop.lat.values, stat_cyc, 
                               cmap = plt.cm.RdBu, 
                               transform = ccrs.PlateCarree(), **kwargs)
        elif plotfun == 'contourf':
            ctf1 = ax.contourf(lon_cyc, stat_crop.lat.values, stat_cyc, 
                               cmap = plt.cm.RdBu, 
                               transform = ccrs.PlateCarree(), **kwargs)

        ctf2 = ax.contourf(lon_cyc, p_crop.lat.values, p_cyc, alpha, 
                           colors = 'none', 
                           hatches = ['....', None], 
                           transform = ccrs.PlateCarree())
        ax.set_title(i_title, loc = 'left')
        ax.outline_patch.set_edgecolor('none')
        
    fig.tight_layout()

    fig.subplots_adjust(**subplot_kwargs[map_type]['adjust'])
    cax = fig.add_axes(subplot_kwargs[map_type]['add_axes'])
    cb = plt.colorbar(ctf1, cax = cax, **subplot_kwargs[map_type]['colorbar'])
    cb.set_label(stat_name)
    return fig

def check_wy(dbpath, yearlow, yearhigh, westoflon=-104, eastoflon=-125):
    """Get USGS sites in sqllitedb have complete obs over time and area
    """
    #yearlow and yearhigh should be in wateryear
    # This is a very poor way of doing this.
    expected_obs_no = ((yearhigh - yearlow) + 1)
    candidate_pass = []
    conn = sqlite3.connect(dbpath)
    c = conn.cursor()
    c.execute('SELECT "STATION ID" FROM StationInfo WHERE LONG_GAGE < ? AND LONG_GAGE > ?', (westoflon, eastoflon))
    candidate_ids = c.fetchall()
    for station in candidate_ids:
        c.execute("SELECT year_nu, count_nu FROM StationWY WHERE site_no=? AND year_nu >= ? AND year_nu <= ?", (station[0], yearlow, yearhigh))
        myc = c.fetchall()
        # Check that we have all years:
        if len(myc) != expected_obs_no:
            continue
        # Check that no years are missing days:
        exam = [False] * expected_obs_no
        for i in range(expected_obs_no):
            if myc[i][1] in [365, 366]:
                exam[i] = True
        if not all(exam):
            continue
        candidate_pass.append(station[0])
    conn.close()
    return candidate_pass

def adj_gamma_kstest(x):
    """Get p-values of KS-test for adjusted-gamma fit on series x
    """
    # Use with pandas aggregate()?
    x_adj = x + 0.000001
    gfit = stats.gamma.fit(x_adj, floc = 0) # This should deal with 0 values.
    return stats.kstest(x_adj, lambda a: stats.gamma.cdf(a, *gfit))[1] # [1] should return pvalue.

def spigamma(x):
    """Transform data like SPI after fitting a gamma function
    """
    # Use with pandas transform().
    zero_mask = x == 0
    q = zero_mask.sum()/len(x)
    g_shape, g_loc, g_scale = stats.gamma.fit(x[~zero_mask], floc = 0)
    g_fit = q + (1 - q) * stats.gamma.cdf(x, a = g_shape, scale = g_scale)
    return stats.norm.ppf(g_fit)

def zscore(x):
    """Standardize data into a Z-score
    """
    # Use with pandas transform(), like spigamma.
    return (x - x.mean()) / x.std()

def plot_northtest(x, nmodes=10):
    """Screeplot `nmodes` leading modes from EOFS solver instance x
    """
    fig = plt.figure(figsize = (3.74016, 4.52756))
    frac_var = x.varianceFraction(nmodes)
    err = x.northTest(nmodes, vfscaled = True)
    plt.errorbar(np.arange(nmodes) + 1, frac_var, yerr = err, fmt = "o")
    plt.xlim(0.5, nmodes + 0.5)
    plt.xlabel("Component")
    plt.ylabel("Fraction of variance")
    return fig

def plot_pc(x, yr, nmodes=10):
    """Plot nmodes leading PCs from EOFS solver x over corresponding array of years, yr
    """
    pc = x.pcs(npcs = nmodes, pcscaling = 1)
    frac_var = x.varianceFraction(nmodes)
    fig, axes = plt.subplots(figsize = (3.74016, 4.52756), nrows = nmodes, 
                             ncols = 1, sharex = True, sharey = True)
    for i in range(nmodes):
        axes.flat[i].plot(yr, pc[:, i], "-o")
        title_str = "PC " + str(i + 1) + " (" + str(np.round(frac_var[i] * 100, 1)) + "%)" 
        axes.flat[i].set_title(title_str)
    return fig

def plot_eof(x, lat, lon, nmodes=10, figure_size=(7.48031, 4.52756)):
    """Plot covariance map for nmodes EOFS of EOFs solver instance x
    """
    eof = x.eofsAsCovariance(neofs = nmodes)
    frac_var = x.varianceFraction(nmodes)
    fig, axes = plt.subplots(figsize = figure_size,
                             nrows = 1, ncols = nmodes, 
                             sharex = True, sharey = True)
    eof_min = np.floor(eof.min())
    eof_max = np.ceil(eof.max())
    for i in range(nmodes):
        m = Basemap(ax = axes.flat[i], width = 2000000, height = 2300000, 
                    resolution = 'l', projection = 'stere', 
                    lat_ts = 40.0, 
                    lat_0 = 40.0, lon_0 = -114.0)
        x, y = m(lon, lat)
        m.drawmapboundary(color = 'none')
        m.drawcoastlines(linewidth = 1, color = "#696969")
        m.drawstates(linewidth = 1, color = "#696969")
        m.drawcountries(linewidth = 1, color = "#696969")
        parallels = np.arange(0., 81, 10)
        m.drawparallels(parallels, #labels = [True, False, False, False],
                        color = "#333333", fontsize = 6, linewidth = 0.5)
        meridians = np.arange(10., 351., 10)
        m.drawmeridians(meridians, #labels = [False, False, False, True], 
                        color = "#333333", fontsize = 6, linewidth = 0.5)
        ctf1 = m.scatter(x, y, s = 50, c = eof[i].squeeze(),
                         vmin = eof_min, vmax = eof_max,
                         cmap = plt.cm.RdBu, edgecolor = "k", linewidth = 0.75)
        percent_var = int(np.round(frac_var[i] * 100, 1)) 
        title_str = string.ascii_lowercase[i] + ") PC" + str(i + 1) + " (" + str(percent_var) + "%)"
        axes.flat[i].set_title(title_str, loc = "left")
    fig.subplots_adjust(bottom = 0.15)
    cax = fig.add_axes([0.17, 0.1, 0.7, 0.01])
    cb = plt.colorbar(ctf1, ticks = np.linspace(eof_min, eof_max, 5),
                      cax = cax, orientation = 'horizontal')
    cb.set_label('Covariance')
    return fig

def plot_gagesmap(lat, lon):
    """Create a simple point map of USGS gages in the Western US, given lat/lon
    """
    fig = plt.figure(figsize = (3.74016, 4.52756))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(width = 2000000, height = 2300000, 
                resolution = 'l', projection = 'stere', 
                lat_ts = 40.0, 
                lat_0 = 40.0, lon_0 = -114.0)
    m.drawcoastlines(color = "#333333")
    m.drawstates(linewidth = 0.7, color = "#333333")
    m.drawcountries(color = "#333333")
    m.shadedrelief()
    parallels = np.arange(0., 81, 10)
    m.drawparallels(parallels, labels = [True, False, True, False], color = "#333333")
    meridians = np.arange(10., 351., 10)
    m.drawmeridians(meridians, labels = [False, True, False, True], color = "#333333")
    m.scatter(lon, lat, s = 30, lw = 1, marker = 'o', latlon = True, facecolors = "none", edgecolors='r')
    plt.title("n = " + str(len(lon)))
    return fig

def pearson_corr(x, field):
    """Pearson correlation with 2-sided t-test

    Parameters:
        x: ndarray
            A 1D array time series.
        field: ndarray
            A 3D array of field values. The first dimension of the array needs 
            to be time.

    Returns: (ndarray, ndarray)
        Two ndarrays. A 2D array of Pearson correlation values and a 2D array of p-values.

    Notes:
        The p-values returned by this function are from a two-sided Student's 
        t-distribution. The test is against the null hypothesis that the 
        correlation is not significantly different from "0".
        This function could use some more work.
    """
    field = field.copy()
    f_oldshape = field.shape
    field.shape = (f_oldshape[0], f_oldshape[1] * f_oldshape[2])
    n = len(x)
    df = n - 2
    r = ((x[:, np.newaxis] * field).sum(axis = 0) - n * x.mean() * field.mean(axis = 0)) / (np.sqrt(np.sum(x**2) - n * x.mean()**2) * np.sqrt(np.sum(field**2, axis = 0) - n * field.mean(axis = 0)**2))
    t = r * np.sqrt(df/(1 - r**2))
    p = stats.betai(0.5*df, 0.5, df/(df+t*t))
    r.shape = (f_oldshape[1], f_oldshape[2])
    p.shape = r.shape
    return r, p

def cut_divisions(x):
    """Return bool array indicating whether observations are in the 'high' composite
    """
    return (x > 0)

def ttest(x, msk):
    """Apply bool mask to x and Welch's t-test the mask results and it's inverse
    """
    t, p = stats.ttest_ind(x[msk], x[~msk], equal_var = False)
    return t, p

def composite_ttest(x, field):
    """2-sided t-test for composites

    Parameters:
        x: ndarray
            A 1D array time series.
        field: ndarray
            A 3D array of field values. The first dimension of the array needs 
            to be time.

    Returns: (ndarray, ndarray)
        A 2D array of difference and a 2D array of p-values.

    Notes:
        The p-values returned by this function are from a two-sided Student's 
    """
    divisions_high = cut_divisions(x)
    t, p = ttest(field, divisions_high)
    dif = np.mean(field[divisions_high], 0) - np.mean(field[~divisions_high], 0)
    return dif, p

# If I have some time it might be worth wrapping funcs below into a special 
# class with plot methods.

def pointfield_corr(**kwargs):
    """ Point correlation with significance test
    """
    out = pointfield_statistic(**kwargs, 
                                     stat_fun = pearson_corr,
                                     stat_name = 'Correlation')
    return out

def plot_pointfield_corr(ds, map_type):
    """Plot point correlation maps
    """
    out = plot_pointfield_statistic(ds = ds,
                                          map_type = map_type,
                                          stat_name = 'Correlation',
                                          plotfun = 'pcolormesh',
                                          vmin = -1, vmax = 1)
    return out

def pointfield_ttest(**kwargs):
    """ Point correlation with significance test
    """
    out = pointfield_statistic(**kwargs, 
                                     stat_fun = composite_ttest)
    return out

def plot_pointfield_ttest(ds, map_type, **kwargs):
    """Plot composite t-test maps
    """
    out = plot_pointfield_statistic(ds = ds,
                                          map_type = map_type,
                                          plotfun = 'pcolormesh',
                                          **kwargs)
    return out
