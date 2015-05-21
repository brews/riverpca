# Copyright 2015 S. B. Malevich <malevich@email.arizona.edu>
# 2015-01-09

# Random collections of often-used code for the project.

import sqlite3
from calendar import monthrange
import numpy as np
import pandas as pd
from numpy.linalg import svd
import scipy.stats as stats
import pylab as plt
from mpl_toolkits.basemap import Basemap

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
    p,k = Phi.shape
    R = np.eye(k)
    d=0
    for i in xrange(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        b = np.dot(Phi.T, np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, diag(diag(np.dot(Lambda.T,Lambda)))))
        u,s,vh = svd(b)
        R = np.dot(u,vh)
        d = np.sum(s)
        if d_old!=0 and d/d_old < 1 + tol:
            break
    return (np.dot(Phi, R), R)

def plot_trendmap(x, lat, lon):
    """Point map colored to show the trend of western US gages"""
    f = plt.figure(figsize = (4, 6))
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
    return f

def plot_gagesmap(lat, lon):
    """Create a simple sample map of USGS gages in the Western US, given lat/lon"""
    f = plt.figure(figsize = (4, 6))
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
    return f

def ttest_proportion_sig(a, b, alpha):
    """Returns the proportion of significant Welch's t-test results"""
    #TODO: This needs to give percent of area that is significant, weighting
    #      the grid by it's latitude.
    t, p = stats.ttest_ind(a, b, equal_var = False)
    # return (p <= alpha).sum()/p.size, t, p
    pass

def field_composite(series, yr, divisions=4):
    """Return SST or Z field composites for extreme ends of a time series"""
    quarts = pd.qcut(series, np.linspace(0, 1, divisions), labels = ["low", "mid", "high"])
    composite = {"low": field[(quarts == "low")], "high": field[(quarts == "high")]}
    years = {"low": yr[(quarts == "low")], "high": yr[(quarts == "high")]}
    return composite, years

def composite_mc_ttest(series, yr, field, alpha=0.1, nruns=500, divisions=4):
    """Field significance for Welch's t-test of field composites"""
    composite, years = field_composite(series, yr)
    proportion_sig, t, p = ttest_proportion_sig(composite["high"], composite["low"], alpha = alpha)
    low_bound = len(years["high"])
    high_bound = low_bound + len(years["low"])
    mc_props = np.zeros(nruns)
    for i in range(nruns):
        draw = np.random.permutation(field.shape[0])
        population_a = field[draw[:low_bound]]
        population_b = field[draw[low_bound:high_bound]]
        mc_props[i] = ttest_proportion_sig(population_a, population_b, alpha)
    return p, stats.percentileofscore(mc_props, proportion_sig, kind = "rank")

def plot_field_composite(series, field):
    """Compare SST or Z field composites for extreme terciles of a time series"""
    pass
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

