# Copyright 2015 S. B. Malevich <malevich@email.arizona.edu>
# 2015-01-09

# Random collections of often-used code for the project.

import sqlite3
import string
import datetime
import itertools
import random
import xarray as xr
import numpy as np
import pandas as pd
# from numpy.linalg import svd
import scipy.stats as stats
import seaborn as sns
import pylab as plt
import matplotlib as mpl
import cartopy.util
import cartopy.crs as ccrs
import matplotlib.path as mplpath
from mpl_toolkits.basemap import Basemap
from calendar import monthrange

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
            s, pvalue = stat_fun(pc_df[pc_i].values, field.sel(season = seas_i).values)
            out[stat_name].loc[dict(PC = pc_i, season= seas_i)] = s
            out['pvalue'].loc[dict(PC = pc_i, season= seas_i)] = pvalue
    if units:
        out[stat_name].attrs['units'] = units
    return out


def plot_pointfield_statistic(ds, map_type, stat_name, sig_alpha=0.05, **kwargs):
    """
    """
    assert map_type in ['north_hemisphere', 'global']

    proj = {'north_hemisphere': ccrs.LambertAzimuthalEqualArea(central_longitude = -160,
                                                               central_latitude = 90),
            'global': ccrs.Robinson(central_longitude = -160)}
    subplot_kwargs = {'north_hemisphere': {'adjust': {'right': 1 - 0.15},
                                         'add_axes': [1 - 0.17, 0.1, 0.017, 0.7],
                                         'colorbar': {'orientation': 'vertical'}},
                      'global': {'adjust': {'bottom': 0.15},
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
                 {'season': 'JJA-1',
                  'PC':     'PC2',
                  'title':  'e) PC2: JJA'},
                 {'season': 'SON-1',
                  'PC':     'PC1',
                  'title':  'b) PC1: SON'},
                 {'season': 'SON-1',
                  'PC':     'PC2',
                  'title':  'f) PC2: SON'},
                 {'season': 'DJF',
                  'PC':     'PC1',
                  'title':  'c) PC1: DJF'},
                 {'season': 'DJF',
                  'PC':     'PC2',
                  'title':  'g) PC2: DJF'},
                 {'season': 'MAM',
                  'PC':     'PC1',
                  'title':  'd) PC1: MAM'},
                 {'season': 'MAM',
                  'PC':     'PC2',
                  'title':  'h) PC2: MAM'}]

    fig = plt.figure(figsize = (5.5, 8))
    for i in range(len(pc_dim) * len(season)):
        i_season = plot_meta[i]['season']
        i_pc = plot_meta[i]['PC']
        i_title = plot_meta[i]['title']
        ax = fig.add_subplot(len(season), len(pc_dim), i + 1, projection = proj[map_type])
        ax.coastlines()
        ax.gridlines(linestyle = 'dotted', color = '#696969', linewidth = 0.5)
        ds_crop = ds.sel(season = i_season, PC = i_pc)
        if map_type == 'north_hemisphere':
            ax.set_boundary(circle, transform = ax.transAxes)
            ds_crop = ds_crop.sel(lat = slice(90, 0))
        dif_crop = ds_crop[stat_name]
        p_crop = ds_crop['pvalue']
        p_cyc, lon_cyc = cartopy.util.add_cyclic_point(p_crop.values, 
                                                       coord = p_crop['lon'].values, 
                                                       axis = -1)

        ctf1 = (ds_crop[stat_name].plot
                                  .pcolormesh(ax = ax,
                                              cmap = plt.cm.RdBu,
                                              transform = ccrs.PlateCarree(), 
                                              add_colorbar = False,
                                              add_labels = False,
                                              **kwargs))
        ctf2 = ax.contourf(lon_cyc, p_crop.lat.values, p_cyc, alpha, 
                           colors = 'none', 
                           hatches = ['....', None], 
                           transform = ccrs.PlateCarree())
        ax.set_title(i_title, loc = 'left')
        
    fig.tight_layout()

    fig.subplots_adjust(**subplot_kwargs[map_type]['adjust'])
    cax = fig.add_axes(subplot_kwargs[map_type]['add_axes'])
    cb = plt.colorbar(ctf1, cax = cax, **subplot_kwargs[map_type]['colorbar'])
    cb.set_label(stat_name)
    
    return fig

class Date(datetime.date):
    # TODO: Is this used?
    def __init__(self, *args, **kwargs):
        super().__init__()
        if self.month in (10, 11, 12):
            self.wateryear = self.year + 1
        else:
            self.wateryear = self.year

def check_monthly(dbpath, yearlow, yearhigh, westoflon=-104, eastoflon=-125):
    """Test that each of the USGS stations in the sqllite database have complete observations over the period we are interested in"""
    expected_obs_no = 12 * ((yearhigh - yearlow) + 1)
    candidate_pass = []
    conn = sqlite3.connect(dbpath)
    c = conn.cursor()
    c.execute("SELECT stationid FROM StationInfo WHERE longage < ? AND longage > ?", (westoflon, eastoflon))
    candidate_ids = c.fetchall()
    for station in candidate_ids:
        c.execute("SELECT month, year, count FROM StationMonthly WHERE stationid=? AND year >= ? AND year <= ?", (station[0], yearlow, yearhigh))
        myc = c.fetchall()
        # Check that we have all months:
        if len(myc) != expected_obs_no:
            continue
        # Check that no months are missing days:
        exam = [False] * expected_obs_no
        for i in range(expected_obs_no):
            if monthrange(myc[i][1], myc[i][0])[1] == myc[i][2]:
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
    gfit = stats.gamma.fit(x_adj, floc = 0) # This should deal with 0 values?
    return stats.kstest(x_adj, lambda a: stats.gamma.cdf(a, *gfit))[1] # [1] should return pvalue.

def spigamma(x):
    """Transform data like SPI after fitting a gamma function"""
    # Use with pandas transform().
    zero_mask = x == 0
    q = zero_mask.sum()/len(x)
    g_shape, g_loc, g_scale = stats.gamma.fit(x[~zero_mask], floc = 0)
    g_fit = q + (1 - q) * stats.gamma.cdf(x, a = g_shape, scale = g_scale)
    return stats.norm.ppf(g_fit)

def zscore(x):
    """Standardize data into a Z-score"""
    # Use with pandas transform(), like spigamma.
    return (x - x.mean()) / x.std()

def trender(x):
    """Get the linear trend coef of a data series."""
    # Use with pandas apply()
    A = np.vstack([np.arange(len(x)), np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, x)[0]
    return m

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    """ Varimax rotation of a 'loadings matrix', Phi. Output is (rotated loadings, rotation matrix). This is loosely based on the R stats library's varimax()"""
    pass
    # p,k = Phi.shape
    # R = np.eye(k)
    # d=0
    # for i in xrange(q):
    #     d_old = d
    #     Lambda = np.dot(Phi, R)
    #     b = np.dot(Phi.T, np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, diag(diag(np.dot(Lambda.T,Lambda)))))
    #     u,s,vh = svd(b)
    #     R = np.dot(u,vh)
    #     d = np.sum(s)
    #     if d_old!=0 and d/d_old < 1 + tol:
    #         break
    # return (np.dot(Phi, R), R)

def plot_northtest(x, nmodes=10):
    """Screeplot `nmodes` leading modes from EOFS solver instance `x`"""
    fig = plt.figure(figsize = (3, 3))
    frac_var = x.varianceFraction(nmodes)
    err = x.northTest(nmodes, vfscaled = True)
    plt.errorbar(np.arange(nmodes) + 1, frac_var, yerr = err, fmt = "o")
    plt.xlim(0.5, nmodes + 0.5)
    plt.xlabel("Component")
    plt.ylabel("Fraction of variance")
    return fig

def plot_pc(x, yr, nmodes=10):
    """Plot `nmodes` leading PCs from EOFS solver instance `x` over corresponding array of years, `yr`"""
    pc = x.pcs(npcs = nmodes, pcscaling = 1)
    frac_var = x.varianceFraction(nmodes)
    fig, axes = plt.subplots(figsize = (9.5, 6.5), nrows = nmodes, ncols = 1, sharex = True, sharey = True)
    for i in range(nmodes):
        axes.flat[i].plot(yr, pc[:, i], "-o")
        axes.flat[i].set_title("PC " + str(i + 1) + " (" + str(np.round(frac_var[i] * 100, 1)) + "%)")
    return fig

def plot_trendmap(x, lat, lon):
    """Point map colored to show the trend of western US gages"""
    fig = plt.figure(figsize = (4, 6))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(width = 2000000, height = 2300000, 
                resolution = 'l', projection = 'stere', 
                lat_ts = 40.0, 
                lat_0 = 40.0, lon_0 = -114.0)
    m.drawcoastlines(color = "#333333")
    m.drawstates(linewidth = 0.7, color = "#333333")
    m.drawcountries(color = "#333333")
    m.scatter(lon, lat, c = x.tolist(), s = 30, latlon = True, edgecolors='k')
    m.colorbar()
    plt.title("Trend coef")
    return fig

def plot_vectormap(coef1, coef2, lat, lon):
    """See Fig. 2 of Quadrelli and Wallace, 2004"""
    fig = plt.figure(figsize = (5, 4))
    m = Basemap(width = 2000000, height = 2300000, 
                resolution = 'l', projection = 'stere', 
                lat_ts = 40.0, 
                lat_0 = 40.0, lon_0 = -114.0)
    x, y = m(lon, lat)
    m.drawcoastlines(linewidth = 0.7, color = "#696969")
    m.drawstates(linewidth = 0.7, color = "#696969")
    m.drawcountries(linewidth = 0.7, color = "#696969")
    m.quiver(x, y, coef1, coef2, scale = 10)
    m.scatter(x, y, facecolors = "none", edgecolor = "k")
    return fig

def plot_eof(x, lat, lon, nmodes=10, figure_size=(9.5, 6.5)):
    """Plot covariance map for `nmodes` EOFS of EOFS solver instance `x`."""
    eof = x.eofsAsCovariance(neofs = nmodes)
    frac_var = x.varianceFraction(nmodes)
    fig, axes = plt.subplots(figsize = figure_size,
                             nrows = nmodes, ncols = 1, 
                             sharex = True, sharey = True)
    eof_min = np.floor(eof.min())
    eof_max = np.ceil(eof.max())
    for i in range(nmodes):
        m = Basemap(ax = axes.flat[i], width = 2000000, height = 2300000, 
                    resolution = 'l', projection = 'stere', 
                    lat_ts = 40.0, 
                    lat_0 = 40.0, lon_0 = -114.0)
        x, y = m(lon, lat)
        m.drawcoastlines(linewidth = 0.7, color = "#696969")
        m.drawstates(linewidth = 0.7, color = "#696969")
        m.drawcountries(linewidth = 0.7, color = "#696969")
        parallels = np.arange(0., 81, 10)
        m.drawparallels(parallels, labels = [True, False, False, False],
                        color = "#333333")
        meridians = np.arange(10., 351., 10)
        m.drawmeridians(meridians, labels = [False, False, False, True], 
                        color = "#333333")
        ctf1 = m.scatter(x, y, s = 50, c = eof[i].squeeze(),
                         vmin = eof_min, vmax = eof_max,
                         cmap = plt.cm.RdBu, edgecolor = "k", lw = 0.75)
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
    """Create a simple sample map of USGS gages in the Western US, given lat/lon"""
    fig = plt.figure(figsize = (4, 6))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(width = 2000000, height = 2300000, 
                resolution = 'l', projection = 'stere', 
                lat_ts = 40.0, 
                lat_0 = 40.0, lon_0 = -114.0)
    # m.drawmapboundary()
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

def plot_pearson(x, field, lat, lon, nmodes=10, alpha=[0.05], msk = False, world_map = False):
    """Plot pearson correlations of "nmodes."""
    pc = x.pcs(npcs = nmodes, pcscaling = 1)
    fig, axes = plt.subplots(figsize = (9.5, 6.5), nrows = nmodes, ncols = 1)
    divs = np.linspace(-1, 1, 11)
    for i in range(nmodes):
        r, p = pearson_corr(pc[:, i], field.copy())
        if np.any(msk):
            r = np.ma.masked_array(r, msk)
            p = np.ma.masked_array(p, msk)
        m = None
        if world_map:
            m = Basemap(ax = axes.flat[i], projection = "robin", lon_0 = 180, resolution = "c")
        else:
            m = Basemap(ax = axes.flat[i], projection = 'npstere', boundinglat = 20, lon_0 = 210, resolution='c')
        x, y = m(lon, lat)
        m.drawcoastlines(color = "#696969")
        m.drawparallels(np.arange(-90, 110, 20), color = "#696969")
        m.drawmeridians(np.arange(0, 360, 60), color = "#696969")
        ctf1 = m.contourf(x, y, r, divs, cmap = plt.cm.RdBu)
        ctf2 = m.contour(x, y, p, alpha, colors = "k", linewidths = 1.5)
        cb = m.colorbar(ctf1)
        cb.set_label("r")
        axes.flat[i].set_title("PC " + str(i + 1))
    return fig

# def pearson_corr(x, field):
#     """Pearson correlation between a 1D time series `x` and field array `f` (with 3 dimensions, the first is time)"""
#     field = field.copy()
#     f_oldshape = field.shape
#     field.shape = (f_oldshape[0], f_oldshape[1] * f_oldshape[2])
#     r = np.empty((f_oldshape[1] * f_oldshape[2]))
#     p = np.empty(r.shape)
#     for i in range(field.shape[1]):
#         r[i], p[i] = stats.pearsonr(x, field[:, i])
#     r.shape = (f_oldshape[1], f_oldshape[2])
#     p.shape = r.shape
#     return r, p

def pearson_corr(x, field):
    """Pearson correlation with 2-sided t-test

    Parameters:
        x: ndarray
            A 1D array time series.
        field: ndarray
            A 3D array of field values. The first dimension of the array needs 
            to be time.

    Returns: (ndarray, ndarray)
        A 2D array of Pearson correlation values and a 2D array of p-values.

    Notes:
        The p-values returned by this function are from a two-sided Student's 
        t-distribution. The test is against the null hypothesis that the 
        correlation is not significantly different from "0".
    """
    field = field.copy()
    f_oldshape = field.shape
    field.shape = (f_oldshape[0], f_oldshape[1] * f_oldshape[2])
    n = len(x)
    df = n - 2
    r = ((x[:, np.newaxis] * field).sum(axis = 0) - n * x.mean() * field.mean(axis = 0)) / (np.sqrt(np.sum(x**2) - n * x.mean()**2) * np.sqrt(np.sum(field**2, axis = 0) - n * field.mean(axis = 0)**2))
    # TODO: Need to ensure that R is between -1 and 1. This is from float-point rounding errors.
    r[r > 1] = 1
    r[r < -1] = -1
    t = r * np.sqrt(df/(1 - r**2))
    p = stats.betai(0.5*df, 0.5, df/(df+t*t))
    # p = 1 - (stats.t.cdf(abs(t), df = df) - stats.t.cdf(-abs(t), df = df))
    r.shape = (f_oldshape[1], f_oldshape[2])
    p.shape = r.shape
    return r, p

def globalweight_proportion(field, lat):
    """Calculate the globally-weighted proportion of a binary field

    Parameters:
        field: ndarray
            A 2D array of boolean values along an regularly-spaced grid across 
            a sphere.
        lat: ndarray
            A 2D array of latitude values (in decimal degrees) which 
            correspond to values in `field`.

    Returns: float
        The proportion of True values in `field` over the entire area of the 
        field.

    Notes:
        This applies sqrt(cos(lat)) weight to field values before calculating 
        the proportion.
    """
    weights = np.sqrt(np.cos(np.deg2rad(lat)))
    prop = (field * weights).sum()/(np.ones(field.shape) * weights).sum()
    return prop

def pearson_fieldsig_test(x, field, lat, local_alpha=0.05, nruns=500):
    """pass"""
    target_r, target_p = pearson_corr(x, field)
    rejectnull_field = target_p <= local_alpha
    target_proportion = globalweight_proportion(rejectnull_field, lat)
    noise_proportion = np.zeros(nruns)
    noise = np.random.normal(loc = np.mean(x), scale = np.std(x), 
                             size = (nruns, len(x)))
    for i in range(nruns):
        noise_r, noise_p = pearson_corr(noise[i], field)
        rejectnull_field = noise_p <= local_alpha
        noise_proportion[i] = globalweight_proportion(rejectnull_field, lat)
    return noise_proportion, target_proportion

def random_combination(iterable, r):
    """Random selection from itertools.combinations(iterable, r)"""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

def ttest_fieldsig_test(x, field, lat, local_alpha=0.05, nruns=500):
    """pass"""
    dif, target_p = composite_ttest(x, field)
    n = len(x)
    n_high = np.sum(cut_divisions(x))
    n_low = n - n_high

    rejectnull_field = target_p <= local_alpha
    target_proportion = globalweight_proportion(rejectnull_field, lat)
    noise_proportion = np.zeros(nruns)
    id_list = list(range(n))
    for i in range(nruns):
        high_targets = list(random_combination(id_list, n_high))
        high_mask = np.zeros(x.shape, dtype = bool)
        high_mask[high_targets] = True
        test_t, test_p = ttest(field, high_mask)
        rejectnull_field = test_p <= local_alpha
        noise_proportion[i] = globalweight_proportion(rejectnull_field, lat)
    return noise_proportion, target_proportion

def cut_divisions(x):
    """Return bool array indicating whether observations are in the 'high' composite"""
    return (x > 0)

def ttest(x, msk):
    """Apply bool mask to x and Welch's t-test the mask results and it's inverse"""
    t, p = stats.ttest_ind(x[msk], x[~msk], equal_var = False)
    return (t, p)

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

def plot_ttest(x, field, lat, lon, nmodes=10, alpha=0.05, msk=False, world_map=False, figure_size=(9.5, 6.5)):
    """Plot composite t-tests fields of nmodes."""
    pc = x.pcs(npcs = nmodes, pcscaling = 1)
    fig, axes = plt.subplots(figsize = figure_size, nrows = nmodes, ncols = 1)
    dif = np.empty((nmodes,) + field.shape[1:])
    p = np.empty(dif.shape)
    for i in range(nmodes):
        dif[i], p[i] = composite_ttest(pc[:, i], field)
        if np.any(msk):
            p[i] = np.ma.masked_array(p[i], msk)
    # max_dif = np.round(np.max([np.abs(np.floor(dif.min())), np.ceil(dif.max())]))
    # divs = np.linspace(-max_dif, max_dif, 11)
    for i in range(nmodes):
        sig_points = np.ma.masked_array(p[i], ~(p[i] <= alpha))
        m = None
        if world_map:
            m = Basemap(ax = axes.flat[i], projection = "robin", lon_0 = 180, resolution = "c")
        else:
            m = Basemap(ax = axes.flat[i], projection = 'npstere', boundinglat = 20, lon_0 = 210, resolution='c')
        xlon, ylat = m(lon, lat)
        m.drawcoastlines(color = "#696969")
        m.drawparallels(np.arange(-90, 110, 20), color = "#696969")
        m.drawmeridians(np.arange(0, 360, 60), color = "#696969")
        m.contourf(xlon, ylat, sig_points, 0, colors = mpl.colors.rgb2hex(sns.color_palette("colorblind")[-1]))
        axes.flat[i].set_title(string.ascii_lowercase[i] + ") PC" + str(i + 1), loc = "left")
    fig.tight_layout()
    return fig

def plot_fieldsig(x, star, alpha=0.2, nbins=25):
    """pass"""
    x.sort()
    fig = plt.figure(figsize = (3, 3))
    plt.hist(x, bins = nbins, histtype = "stepfilled")
    plt.axvline(x = x[- alpha * len(x)], linestyle = ":", color = "black")
    plt.axvline(x = star, color = "red")
    # plt.annotate('', xy = (star, 35), 
                # xytext = (star, 60), 
                # arrowprops = dict(facecolor = 'blue', shrink = 0.05))
    plt.xlabel('Proportion significant')
    plt.ylabel('Count')
    plt.xlim(0, np.max(np.max(x), star))
    return fig

def plot_hgt_composite(field, lat, lon, msk, use_ax = None, plot_title=None, 
                       contour_divs=None, fill_min=None, fill_max=None):
    """Plot a hgt composite given a mask"""
    if use_ax is None:
        use_ax = plt.gca()
    field_mean = field[msk].mean(axis = 0)
    zonal_mean = field_mean.mean(axis = 1)
    eddy_mean = field_mean - zonal_mean[:, np.newaxis]
    m = Basemap(ax = use_ax, projection = 'npstere', boundinglat = 20,
                lon_0 = 210, resolution='c')
    m.drawcoastlines(color = "#696969")
    m.drawparallels(np.arange(-90, 110, 20), color = "#696969")
    m.drawmeridians(np.arange(0, 360, 60), color = "#696969")
    ctf = m.contourf(lon, lat, field_mean, latlon = True, 
                     V = contour_divs)
    m.colorbar(ctf)
    # pcol = m.pcolormesh(lon, lat, eddy_mean, latlon = True, 
    #                     cmap = plt.cm.RdBu, 
    #                     vmin = fill_min, vmax = fill_max)
    ct = m.contour(lon, lat, field_mean, colors = "k", latlon = True, 
                     V = contour_divs)
    # plt.clabel(ct, fontsize = 10, fmt = "%.0f")
    # cb = m.colorbar(pcol)
    if plot_title:
        use_ax.set_title(plot_title)

def plot_many_hgt_composites(pc, yr, **kwargs):
    """Plot multiple hgt composites based on +/- phases of a PC
    """
    fig, axes = plt.subplots(figsize = (6.5, 9.5), nrows = 2, ncols = 1)
    msks = [pc > 0, pc < 0]
    signs = ["+", "-"]
    intervals = np.arange(0, kwargs["field"].max() + 60, 60)
    for i in range(2):
        target_mask = msks[i]
        year_str = "\n" + ", ".join(str(v) for v in list(yr[target_mask]))
        # year_str = ""
        title_str = signs[i] + "PC" + year_str
        plot_hgt_composite(msk = target_mask, use_ax = axes.flat[i], 
                           plot_title = title_str, contour_divs = intervals, **kwargs)
    return fig
