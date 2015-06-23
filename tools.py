# Copyright 2015 S. B. Malevich <malevich@email.arizona.edu>
# 2015-01-09

# Random collections of often-used code for the project.

import sqlite3
import string
from calendar import monthrange
import datetime
import numpy as np
import pandas as pd
# from numpy.linalg import svd
import scipy.stats as stats
import pylab as plt
from mpl_toolkits.basemap import Basemap

class Date(datetime.date):
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

def spigamma(x):
    """Transform data like SPI after fitting a gamma function"""
    # Use with pandas transform().
    zero_mask = x == 0
    q = zero_mask.sum()/len(x)
    g_shape, g_loc, g_scale = stats.gamma.fit(x[~zero_mask], floc = 0)
    g_fit = q + (1 - q) * stats.gamma.cdf(x, a = g_shape, scale = g_scale)
    return stats.norm.ppf(g_fit)

def trender(x):
    """Get the linear trend coef of a data series."""
    # Use with pandas apply()
    A = np.vstack([np.arange(len(x)), np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, x)[0]
    return m

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    """ Varimax rotation of a 'loadings matrix', Phi. Output is (rotated loadings, rotation matrix). This is loosely based on the R stats library's varimax()."""
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

def plot_eof(x, lat, lon, nmodes=10):
    """Plot covariance map for `nmodes` EOFS of EOFS solver instance `x`."""
    eof = x.eofsAsCovariance(neofs = nmodes)
    frac_var = x.varianceFraction(nmodes)
    fig, axes = plt.subplots(figsize = (9.5, 6.5), nrows = nmodes, ncols = 1)
    divs = np.linspace(np.floor(eof.min()), np.ceil(eof.max()), 21)
    for i in range(nmodes):
        m = Basemap(ax = axes.flat[i], width = 2000000, height = 2300000, 
                    resolution = 'l', projection = 'stere', 
                    lat_ts = 40.0, 
                    lat_0 = 40.0, lon_0 = -114.0)
        x, y = m(lon, lat)
        m.drawcoastlines(linewidth = 0.7, color = "#696969")
        m.drawstates(linewidth = 0.7, color = "#696969")
        m.drawcountries(linewidth = 0.7, color = "#696969")
        ctf1 = m.contourf(x, y, eof[i].squeeze(), divs, cmap = plt.cm.RdBu_r, tri = True)
        cb = m.colorbar(ctf1)
        cb.set_label("cov")
        m.scatter(x, y, facecolors = "none", edgecolor = "k")
        axes.flat[i].set_title("EOF " + str(i + 1) + " (" + str(np.round(frac_var[i] * 100, 1)) + "%)")
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
    m.scatter(lon, lat, s = 30, marker = 'o', latlon = True, facecolors = "none", edgecolors='r')
    plt.title("n = " + str(len(lon)))
    return fig

def plot_pearson(x, field, lat, lon, nmodes=10, alpha=[0.05], msk = False, world_map = False):
    """Plot pearson correlations of "nmodes."""
    pc = x.pcs(npcs = nmodes, pcscaling = 1)
    fig, axes = plt.subplots(figsize = (9.5, 6.5), nrows = nmodes, ncols = 1)
    divs = np.linspace(-1, 1, 21)
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
        ctf1 = m.contourf(x, y, r, divs, cmap = plt.cm.RdBu_r)
        ctf2 = m.contour(x, y, p, alpha, colors = "k")
        cb = m.colorbar(ctf1)
        cb.set_label("r")
        axes.flat[i].set_title("PC " + str(i + 1))
    return fig

def pearson_corr(x, field):
    """Pearson correlation between a 1D time series `x` and field array `f` (with 3 dimensions, the first is time)"""
    f_oldshape = field.shape
    field.shape = (f_oldshape[0], f_oldshape[1] * f_oldshape[2])
    r = np.empty((f_oldshape[1] * f_oldshape[2]))
    p = np.empty(r.shape)
    for i in range(field.shape[1]):
        r[i], p[i] = stats.pearsonr(x, field[:, i])
    r.shape = (f_oldshape[1], f_oldshape[2])
    p.shape = r.shape
    return r, p

# def ttest_proportion_sig(a, b, alpha):
#     """Returns the proportion of significant Welch's t-test results"""
#     #TODO: This needs to give percent of area that is significant, weighting
#     #      the grid by it's latitude.
#     t, p = stats.ttest_ind(a, b, equal_var = False)
#     # return (p <= alpha).sum()/p.size, t, p
#     pass

# def build_composite(series, yr, field, divisions=2):
#     """Return SST or Z field composites for extreme ends of a time series"""
#     labs = list(string.ascii_lowercase)[:divisions]
#     quarts = pd.qcut(series, np.linspace(0, 1, divisions + 1), labels = labs)
#     composite = {"low": field[(quarts == labs[0])],
#                  "high": field[(quarts == labs[-1])]}
#     years = {"low": yr[(quarts == labs[0])],
#              "high": yr[(quarts == labs[-1])]}
#     return composite, years

# def composite_mc_ttest(series, yr, field, alpha=0.1, nruns=500, divisions=4):
#     """Field significance for Welch's t-test of field composites"""
#     composite, years = field_composite(series, yr, field)
#     proportion_sig, t, p = ttest_proportion_sig(composite["high"], composite["low"], alpha = alpha)
#     low_bound = len(years["high"])
#     high_bound = low_bound + len(years["low"])
#     mc_props = np.zeros(nruns)
#     for i in range(nruns):
#         draw = np.random.permutation(field.shape[0])
#         population_a = field[draw[:low_bound]]
#         population_b = field[draw[low_bound:high_bound]]
#         mc_props[i] = ttest_proportion_sig(population_a, population_b, alpha)
#     return p, stats.percentileofscore(mc_props, proportion_sig, kind = "rank")


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

    # wings = np.max([np.absolute([x.min() for x in ), np.absolute(composit_anoms.max())])
#     # Composite map of NDJ mean 500 mb height anomalies.
#     erai = np.load(ERAI_PATH)
#     msk_time = (erai["time"] >= SHORTCALYEAR_LOW + 1) & (erai["time"] <= SHORTCALYEAR_HIGH)
#     hgts = erai["data"][0, 1, msk_time]
# 
#     # First dim is PC, second is 0 -> low, 1 -> high:
#     quarts = pd.qcut(series, [0, 0.33, 0.66, 1], labels = ["low", "mid", "high"])
#     composite_low = np.mean(hgts[(quarts == "low")], 0)
#     composite_high = np.mean(hgts[(quarts == "high")], 0)
# 
#     wings = np.max([np.absolute(composit_anoms.min()), np.absolute(composit_anoms.max())])
#     divs = np.linspace(-wings, wings, 21)
#     divs2 = np.linspace(-wings, wings, 11)
#     fig, axes = plt.subplots(figsize = (8, 7.5), nrows = 2, ncols = 1)
#     for i in range(2):
#         m = Basemap(ax = axes.flat[i], projection = 'npstere', boundinglat = 20, lon_0 = 210, resolution='c')
#         x, y = m(field["lon"], field["lat"])
#         m.drawcoastlines(color = "#696969")
#         m.drawparallels(np.arange(-90, 110, 20), color = "#696969")
#         m.drawmeridians(np.arange(0, 360, 60), color = "#696969")
#     #     m.contour(x, y, composit_anoms[0, i], divs2, colors = "k")
#         pcol = m.contourf(x, y, composit_anoms[0, i], divs, cmap = plt.cm.RdBu_r)
#         cb = m.colorbar(pcol)
#         cb.set_label("Anomaly mean (m)")
#         axes.flat[i].set_title(("Low", "High")[i]+" PC 1" + " ERA-I 500mb NDJ composit")
#     plt.show()
#         
#     fig, axes = plt.subplots(figsize = (8, 7.5), nrows = 2, ncols = 1)
#     for i in range(2):
#         m = Basemap(ax = axes.flat[i], projection = 'npstere', boundinglat = 20, lon_0 = 210, resolution='c')
#         x, y = m(field["lon"], field["lat"])
#         m.drawcoastlines(color = "#696969")
#         m.drawparallels(np.arange(-90, 110, 20), color = "#696969")
#         m.drawmeridians(np.arange(0, 360, 60), color = "#696969")
#     #     m.contour(x, y, composit_anoms[1, i], divs2, colors = "k")
#         pcol = m.contourf(x, y, composit_anoms[1, i], divs, cmap = plt.cm.RdBu_r)
#         cb = m.colorbar(pcol)
#         cb.set_label("Anomaly mean (m)")    
#         axes.flat[i].set_title(("Low", "High")[i]+" PC 2" + " ERA-I 500mb NDJ composit")
#     plt.show()
# 
#     quarts = pd.qcut(pc[:, 0], [0, 0.33, 0.66, 1], labels = ["low", "mid", "high"])
#     print("PC1 low+high:")
#     print(yr[quarts == "low"])
#     print(yr[quarts == "high"])
#     quarts = pd.qcut(pc[:, 1], [0, 0.33, 0.66, 1], labels = ["low", "mid", "high"])
#     print("PC2 low+high:")
#     print(yr[quarts == "low"])
#     print(yr[quarts == "high"])

def globalweight_proportion(field, lat):
    """Calculate the globally-weighted proportion of a binary field

    Parameters:
        field: ndarray
            A 2D array of boolean values along an equally-spaced grid across a sphere.
        lat: ndarray
            A 2D array of latitude values (in decimal degrees) which correspond to values in `field`.

    Returns: float
        The proportion of True values in `field` over the entire area of the field.

    Notes:
        This applies sqrt(cos(lat)) weight to field values before calculating the proportion.
    """
    weights = np.sqrt(np.cos(np.deg2rad(lat)))
    prop = (field * weights).sum()/(np.ones(field.shape) * weights).sum()
    return prop